import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import math
import torch.nn.functional as F
from torch.optim import Adam
import os
import sys

LANG = "fi"

# CORPORA = {
#     "train": "/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.2007.en.shuffled",  # "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/train.eng",
#     "dev": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/dev.eng",
#     "test": "/mnt/storage/swexler/thesis-wexler/examples/french-data-7-mil-512-filtered/test.eng",
# }


device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if __name__ == "__main__"
    else torch.device("cpu")
)


class FileBasedLMData(IterableDataset):
    def __init__(self, filepath, max_length, lang_code=1, eos=2):
        self.filepath = filepath
        self.max_length = max_length
        self.lang_code = lang_code
        self.eos = eos

    def parse_line(self, line):
        return [x + 3 for x in line.strip().encode("utf-8")]

    def __iter__(self):
        for line in open(self.filepath, "r"):
            tokens = [self.lang_code] + self.parse_line(line) + [self.eos]
            chunk = tokens[: self.max_length]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, target_ids


class StreamingDatasetLMData(IterableDataset):
    def __init__(self, name, lang, split, max_length, lang_code=1, eos=2):
        self.max_length = max_length
        self.ds = load_dataset(name, lang, split=split, streaming=True)
        self.lang_code = lang_code
        self.eos = eos

    def parse_line(self, line):
        return [x + 3 for x in line.strip().encode("utf-8")]

    def __iter__(self):
        for doc in self.ds:
            line = doc["translation"][LANG]
            tokens = [self.lang_code] + self.parse_line(line) + [self.eos]
            chunk = tokens[: self.max_length]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, target_ids


# combining function - mainly does padding
def collate_causal_lm(batch, pad_token_id=0):
    inputs, targets = zip(*batch)
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    target_ids = pad_sequence(
        targets, batch_first=True, padding_value=-100
    )  # -100 is ignored in loss by default
    return input_ids, target_ids


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_len
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids):
        input_ids = input_ids[:, : self.max_len]
        x = self.token_embedding(input_ids)  # (B, T, d_model)

        x = self.positional_encoding(x)

        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float("-inf"))

        x = self.transformer(x, mask=causal_mask)  # Only self-attention
        return self.lm_head(x)  # (B, T, vocab_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        try:
            x = x + self.pe[:, : x.size(1)]
        except RuntimeError:
            msg = f"{x.shape}\n"
            msg += f"{self.pe[:, : x.size(1)].shape}\n"
            raise Exception(msg)
        return self.dropout(x)


def get_total_lines(filepath):
    with open(filepath, "rb") as f:
        return sum(1 for _ in f)


@torch.no_grad()
def evaluate(model, dataloader, device=device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(
            input_ids
        )  # (B, T, vocab_size) = batch size rows, # of tokens length, vocab size depth (logit for each letter)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="sum"
        )

        # Compute predictions and accuracy
        preds = logits.argmax(
            dim=-1
        )  # (B, T) = batch size rows, # of tokens length where each item is most probable char
        mask = (
            target_ids != -100
        )  # an example doesn't know its padded --> ignore padding positions
        correct = (
            preds == target_ids
        ) & mask  # Boolean 2d array of each char where item is True if prediction is correct
        # & mask part allows it to ignore masked tokens

        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()

    model.train()
    avg_loss = total_loss / total_tokens
    accuracy = (
        total_correct / total_tokens
    )  # % of next tokens the model predicted correctly
    return avg_loss, accuracy


# does training loop and updates saved model if applicable during evaluation periods
def train(
    model,
    dataloader,
    optimizer,
    val_dataloader=None,
    val_interval=1000,
    training_steps=500000,
    model_dir=None,
):
    model.train()
    best_val_loss = None
    data_iter = iter(dataloader)  # dataloader is batched
    total_loss = 0
    for global_step in tqdm(range(training_steps)):

        try:
            input_ids, target_ids = next(data_iter)  # these are 2d bc they are batched
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, target_ids = next(data_iter)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_ids.view(-1)
        )  # loss between predictions and target IDs {-log prob of correct token} --> doing for a whole sentence at once
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if val_dataloader is not None and global_step % val_interval == 0:
            val_loss, val_acc = evaluate(model, val_dataloader)
            if model_dir is not None and (
                best_val_loss is None or val_loss < best_val_loss
            ):
                print("Saving new best model.")
                best_val_loss = val_loss
                os.makedirs(model_dir, exist_ok=True)
                checkpoint_path = os.path.join(model_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "step": global_step,
                    },
                    checkpoint_path,
                )

            print(
                f"[Step {global_step}] Validation loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}"
            )
            sys.stdout.flush()


def initialize_model(max_length):
    vocab_size = 259  # 256 + pad + lang_code + eos
    model = DecoderOnlyTransformer(
        vocab_size,
        d_model=1024,
        nhead=16,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_len=max_length,
    )
    return model


def main():
    max_length = 1024
    pad, lang_code, eos = 0, 1, 2
    model_dir = f"experiments/{LANG}-autocomplete-v1"
    train_dataset = StreamingDatasetLMData(
        "wmt19", f"{LANG}-en", "train", 1024, lang_code, eos
    )
    val_dataset = StreamingDatasetLMData(
        "wmt19", f"{LANG}-en", "validation", 1024, lang_code, eos
    )
    test_dataset = StreamingDatasetLMData(
        "wmt19", f"{LANG}-en", "validation", 1024, lang_code, eos
    )
    # train_dataset = FileBasedLMData(
    #     CORPORA["train"], max_length=max_length, lang_code=lang_code, eos=eos
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=pad),
    )
    # val_dataset = FileBasedLMData(CORPORA["dev"], max_length=max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=pad),
    )
    # test_dataset = FileBasedLMData(CORPORA["test"], max_length=max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: collate_causal_lm(batch, pad_token_id=0),
    )

    model = initialize_model(max_length)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    train(
        model,
        train_loader,
        optimizer,
        val_loader,
        model_dir=model_dir,
        training_steps=500000,
        val_interval=1000,
    )


if __name__ == "__main__":
    main()
