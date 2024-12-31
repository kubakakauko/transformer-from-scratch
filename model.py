import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    # d_model: represents the model size; 512
    # given a number we want to get the same vector every time, this is what embedding does
    # - its mapping between numbers and vector of size 512 (in this case)

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )  # a form of dictionary: maps numbers to a same vector every time; this vector is learnt by the model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # this is from the paper


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # size of the embedding model: 512
        self.seq_len = seq_len  # maximum length of the sentence
        self.dropout = nn.Dropout(dropout)  # makes the model overfit less

        # we will need vectors of seq_len * d_model. (we will need 512 rows and seq_len columns)

        pe = torch.zeros(seq_len, d_model)

        # create a vector of shame (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # corresponds to pos in the formula

        # create denominator of the formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # corresponds with the dividing part of the formula

        # Apply the sin to even positions and cos to odd positions.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # now we must add the batch dimension to this tensor so that we can apply it to the whole sentences
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # we register it as a buffer so that it gets saved from the tensor along with the state of the model.
        # otherwise this would be lost.
        self.register_buffer("pe", pe)

    # Add positional encoding to the input embeddings and apply dropout to the result.
    # allows the model to learn positional information while preventing overfitting.
    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    # alfa/gamma is multiplied
    # beta/bias is added
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std * self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # we need to make sure that the d_model is devisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        # define matrixes in which we will store w, q, k, v
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # before applyind the softmax, we apply the mask so that the softmax can later replace the values with 0
        # we mask all the words that we want to hide (e.g padding values)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # we will also use the second part of the return for visualisation of the attention
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # mask helps by hiding some words so that the model does not interact with them
        # we do it by setting their values to something small before it goes to the softmax
        # after small values go thorugh the softmax they become 0.
        query = self.w_q(q)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # devide them to smaller matixes so that we have the heads
        # we keep the batch dimesion, because we dont want to split the sentence. We want to split the embeddings into h parts.
        # we want to keep the seq dim
        # we want to split the d_model into 2 smaller dimension which is h,d_k
        # we transpose because we want the h dimension to be 2nd dimension instead the 3rd
        # This way each head sees the entire sentence (query.shape[1], d_k)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        # each head will see each word in the sentence but a smaller part of the embeddings
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Now we need to calculate the attention using the attention formula
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        # self.h * self.d_k = d_model because how we defined self.d_k = d_model // h

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply x by Wo which is our output matix
        return self.w_o(x)


# Now we need to build the connection which will go from add & norm and go to the other add & norm skipping Feed Forward.
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # in the paper Attetion is all you need, they first apply the sublayer and then the norm however both work
        return x + self.dropout(sublayer(self.norm(x)))


# All of the sublayers is combined into a bigger block which is repeated N times. That combining block is called the Encoder block.
# where the output of the preveous block is sent to the next one and the output of the last encoder block is sent to the Decoder.


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # 2 residual (skip) conections
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        # apply the mask so that padding words dont interract with the model.

        # for the first residual connection we send the intput to the MultiHeadAttentionBlock and also skip right to the Add & LayerNormalization
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # feed worward
        x = self.residual_connections[1](x, self.feed_forward_block)

        # what this does is combines the feed forward and X (output of the preveous layer) then apply ResidualConnection
        # all of the above defines our Encoder block.
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Initializes the Encoder with a list of layers and a LayerNormalization instance.

        Args:
            layers (nn.ModuleList): A list of encoder layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Defines the forward pass of the Encoder.

        Args:
            x: Input tensor.
            mask: Mask tensor to prevent attention to certain positions.

        Returns:
            Tensor: The normalized output tensor after processing through all layers.
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # we have two seperate masks because we are dealing with a translation task.
        # one mask is for the source language, e.g English and the other e.g Italian.
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, trg_mask)
        )
        # cross attention which is the second residal connection
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.project(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

        # define 3 methods, encode, decode and project.

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, trg_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.projection_layer(x)


# the naming is based of the translation task this model will be used for however this transformer can be used for any other taks,
# we can change the naming convention below for clarity.
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # creathe the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attendion_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attendion_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

        # create the encoder and the decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initializes the parameters with random values
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
