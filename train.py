from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path
from dataset import (  # If you are not using causal_mask, remove or comment out
    BilingualDataset,
    causal_mask,
)
from model import build_transformer


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output reuse it for every token we get in the decoder
    encoder_output = model.encode(source, source_mask)

    # inferencing
    # for first itteration
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the trarget(decoder input)
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:, -1])
        # select the token with max probability (because greedy search)
        _, next_word = torch.max(prob, dim=1)

        # get the word and append it back
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


# validation loop
def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer,
    num_examples=2,
):
    # tells pytorch that we are in evaluation mode
    model.eval()
    count = 0
    source_texts = []  # Use plural for clarity
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            assert encoder_input.size(0) == 1

            # Forward pass
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]

            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Save to the lists
            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_out_text)

            print_msg("-" * 80)
            print_msg(f"SOURCE: {src_text}")
            print_msg(f"TARGET: {tgt_text}")
            print_msg(f"PRED: {model_out_text}")

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    """
    Generator yielding one sentence at a time from dataset ds in the given language.
    """
    for item in ds:
        yield item["translation"][
            lang
        ]  # Each item has translation dict: {lang_src:..., lang_tgt:...}


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        # Create the base Tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Create a separate trainer
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )

        # Train
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # Save
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    # 1) Load dataset as a single split. So it’s a plain Dataset, not DatasetDict/IterableDataset
    ds_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # 2) Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 3) Split into train/val sets
    # If ds_raw is an actual Dataset, we can do:
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size

    train_ds_raw, val_ds_raw = torch.utils.data.random_split(
        ds_raw, [train_size, val_size]
    )

    # 4) Wrap with BilingualDataset
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # Show approximate max length (purely for debugging)
    max_len_src = 0
    max_len_tgt = 0
    for item in train_ds_raw:
        src_sent = item["translation"][config["lang_src"]]
        tgt_sent = item["translation"][config["lang_tgt"]]
        src_ids = tokenizer_src.encode(src_sent).ids
        tgt_ids = tokenizer_tgt.encode(tgt_sent).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length src: {max_len_src}, tgt: {max_len_tgt}")

    # 5) Create Dataloaders
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Example: build a transformer with given source/target vocab sizes
    and pass relevant config values.
    """
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Get data + tokenizers
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Build model (pass config plus the two vocab sizes)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # TensorBoard
    writer = SummaryWriter(config["experiment_name"])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # Possibly restore from checkpoint
    initial_epoch = 0
    global_step = 0
    if config.get("preload"):
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preload model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    # Loss function
    pad_id = tokenizer_src.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)

    # Training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)

            # Forward pass
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_tgt_len)

            # Labels
            label = batch["label"].to(device)  # (B, seq_len)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            # Logging
            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backprop + update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            run_validation(
                model,
                val_loader,
                tokenizer_src,
                tokenizer_tgt,
                config["seq_len"],
                device,
                lambda msg: batch_iterator.write(msg),
                global_step,
                writer,
            )

            global_step += 1

        # Save checkpoint each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = get_config()
    train_model(config)  # corrected function name
