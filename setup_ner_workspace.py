#!/usr/bin/env python3
"""Create a clean biomedical NER BiLSTM-CRF workspace and optionally start training.

Usage:
  python setup_ner_workspace.py --write-only
  python setup_ner_workspace.py --epochs 3 --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FILES = {}

FILES["requirements.txt"] = r'''torch>=2.2
datasets>=2.19
tqdm>=4.66
seqeval>=1.2.2
'''

FILES["README.md"] = r'''# Biomedical NER BiLSTM-CRF Workspace

A small, readable biomedical Named Entity Recognition (NER) project for pharmacy/medical research coursework. It trains a **custom PyTorch BiLSTM-CRF** model on the Hugging Face `tner/bc5cdr` dataset, which contains Chemical/Drug and Disease entities.

This is intentionally **not SOTA**. It is designed to be understandable, runnable from a terminal/Termux/Colab, and easy to discuss in a graduation research project.

## Quick start

```bash
python setup_ner_workspace.py --write-only
python -m pip install -r requirements.txt
python train.py --epochs 3 --batch-size 16 --max-train-samples 4000
python predict.py --checkpoint checkpoints/best_model.pt
```

Or let the master script write files, install packages, and start training:

```bash
python setup_ner_workspace.py --epochs 3 --batch-size 16 --max-train-samples 4000
```

## Termux notes

Install Python and required build tools first if needed:

```bash
pkg update
pkg install python clang rust git
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On low-memory devices, use smaller settings:

```bash
python train.py --epochs 2 --batch-size 4 --embedding-dim 64 --hidden-dim 128 --max-train-samples 1000
```

## Colab notes

Upload this folder or clone your repository, then run:

```python
!pip install -r requirements.txt
!python train.py --epochs 5 --batch-size 32
!python predict.py --checkpoint checkpoints/best_model.pt
```

## Files

- `setup_ner_workspace.py` — one-command master script that writes all project files.
- `data.py` — downloads `tner/bc5cdr`, builds vocab/tag maps, pads batches.
- `model.py` — custom BiLSTM-CRF architecture.
- `metrics.py` — entity-level precision, recall, and F1.
- `train.py` — training/evaluation/checkpoint pipeline.
- `predict.py` — interactive CLI inference.
'''

FILES["data.py"] = r'''from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def normalize_token(token: str, lowercase: bool = True) -> str:
    token = token.strip()
    if lowercase:
        token = token.lower()
    if re.fullmatch(r"\d+([.,]\d+)?", token):
        return "<NUM>"
    return token


@dataclass
class NERExample:
    tokens: List[str]
    tags: List[str]


class BiomedicalNERDataset(Dataset):
    def __init__(self, examples: List[NERExample], word2idx: Dict[str, int], tag2idx: Dict[str, int], lowercase: bool = True):
        self.examples = examples
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        token_ids = [self.word2idx.get(normalize_token(tok, self.lowercase), self.word2idx[UNK_TOKEN]) for tok in ex.tokens]
        tag_ids = [self.tag2idx[tag] for tag in ex.tags]
        return {
            "tokens": ex.tokens,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "length": len(token_ids),
        }


def _get_tag_names(split) -> List[str]:
    feature = split.features["tags"] if "tags" in split.features else split.features["ner_tags"]
    if hasattr(feature.feature, "names"):
        return list(feature.feature.names)
    raise ValueError("Could not read tag names from dataset features.")


def load_bc5cdr_examples(dataset_name: str = "tner/bc5cdr") -> Tuple[dict, List[str]]:
    raw = load_dataset(dataset_name)
    first_split = raw["train"]
    tag_names = _get_tag_names(first_split)
    token_col = "tokens"
    tag_col = "tags" if "tags" in first_split.column_names else "ner_tags"
    examples = {}
    for split_name, split in raw.items():
        rows = []
        for row in split:
            rows.append(NERExample(tokens=list(row[token_col]), tags=[tag_names[i] for i in row[tag_col]]))
        examples[split_name] = rows
    return examples, tag_names


def build_maps(train_examples: Iterable[NERExample], tag_names: List[str], min_freq: int = 1, lowercase: bool = True):
    counts = Counter()
    for ex in train_examples:
        counts.update(normalize_token(tok, lowercase) for tok in ex.tokens)
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counts.most_common():
        if count >= min_freq and word not in word2idx:
            word2idx[word] = len(word2idx)

    tag2idx = {PAD_TAG: 0}
    for tag in tag_names:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
    tag2idx[START_TAG] = len(tag2idx)
    tag2idx[STOP_TAG] = len(tag2idx)
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return word2idx, tag2idx, idx2tag


def collate_batch(batch, pad_word_idx: int, pad_tag_idx: int):
    max_len = max(item["length"] for item in batch)
    token_ids = torch.full((len(batch), max_len), pad_word_idx, dtype=torch.long)
    tag_ids = torch.full((len(batch), max_len), pad_tag_idx, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    tokens = []
    lengths = []
    for i, item in enumerate(batch):
        length = item["length"]
        token_ids[i, :length] = item["token_ids"]
        tag_ids[i, :length] = item["tag_ids"]
        mask[i, :length] = True
        tokens.append(item["tokens"])
        lengths.append(length)
    return {"tokens": tokens, "token_ids": token_ids, "tag_ids": tag_ids, "mask": mask, "lengths": lengths}


def make_dataloaders(batch_size: int = 16, min_freq: int = 1, lowercase: bool = True, dataset_name: str = "tner/bc5cdr", max_train_samples: int | None = None):
    examples, tag_names = load_bc5cdr_examples(dataset_name)
    if max_train_samples:
        examples["train"] = examples["train"][:max_train_samples]
    word2idx, tag2idx, idx2tag = build_maps(examples["train"], tag_names, min_freq, lowercase)
    datasets = {name: BiomedicalNERDataset(rows, word2idx, tag2idx, lowercase) for name, rows in examples.items()}
    collate = lambda b: collate_batch(b, word2idx[PAD_TOKEN], tag2idx[PAD_TAG])
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), collate_fn=collate)
        for name, ds in datasets.items()
    }
    return loaders, word2idx, tag2idx, idx2tag
'''

FILES["model.py"] = r'''from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, pad_tag_idx: int, start_tag_idx: int, stop_tag_idx: int, embedding_dim: int = 100, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even because the BiLSTM is bidirectional.")
        self.tagset_size = tagset_size
        self.pad_tag_idx = pad_tag_idx
        self.start_tag_idx = start_tag_idx
        self.stop_tag_idx = stop_tag_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.transitions = nn.Parameter(torch.empty(tagset_size, tagset_size))
        nn.init.xavier_uniform_(self.transitions)
        # Transition scores use [from_tag, to_tag]. Forbid impossible paths while
        # still allowing START -> real_tag and real_tag -> STOP transitions.
        self.transitions.data[:, start_tag_idx] = -10000.0
        self.transitions.data[stop_tag_idx, :] = -10000.0
        self.transitions.data[:, pad_tag_idx] = -10000.0
        self.transitions.data[pad_tag_idx, :] = -10000.0

    def _emissions(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.dropout(self.embedding(token_ids))
        lstm_out, _ = self.lstm(embeds)
        return self.hidden2tag(self.dropout(lstm_out))

    def _log_sum_exp(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        max_score, _ = tensor.max(dim)
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score.unsqueeze(dim)), dim))

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = emissions.new_full((batch_size, self.tagset_size), -10000.0)
        score[:, self.start_tag_idx] = 0.0
        for t in range(seq_len):
            next_score = self._log_sum_exp(score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1), dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score = score + self.transitions[:, self.stop_tag_idx].unsqueeze(0)
        return self._log_sum_exp(score, dim=1)

    def _score_gold_sequence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = emissions.new_zeros(batch_size)
        previous_tags = emissions.new_full((batch_size,), self.start_tag_idx, dtype=torch.long)
        for t in range(seq_len):
            current_tags = tags[:, t]
            emit = emissions[torch.arange(batch_size, device=emissions.device), t, current_tags]
            trans = self.transitions[previous_tags, current_tags]
            score += (emit + trans) * mask[:, t]
            previous_tags = torch.where(mask[:, t], current_tags, previous_tags)
        score += self.transitions[previous_tags, self.stop_tag_idx]
        return score

    def neg_log_likelihood(self, token_ids: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = self._emissions(token_ids)
        partition = self._compute_log_partition(emissions, mask)
        gold = self._score_gold_sequence(emissions, tags, mask)
        return torch.mean(partition - gold)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self._emissions(token_ids)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, _ = emissions.shape
        score = emissions.new_full((batch_size, self.tagset_size), -10000.0)
        score[:, self.start_tag_idx] = 0.0
        history = []
        for t in range(seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_path = next_score.max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_path)
        score = score + self.transitions[:, self.stop_tag_idx].unsqueeze(0)
        best_last_score, best_last_tag = score.max(dim=1)
        del best_last_score
        decoded = []
        lengths = mask.long().sum(dim=1).tolist()
        for i in range(batch_size):
            tag = best_last_tag[i]
            path = []
            for hist in reversed(history[: lengths[i]]):
                path.append(int(tag.item()))
                tag = hist[i][tag]
            decoded.append(list(reversed(path)))
        return decoded
'''

FILES["metrics.py"] = r'''from __future__ import annotations

from typing import Iterable, List, Set, Tuple


def bio_spans(tags: List[str]) -> Set[Tuple[str, int, int]]:
    spans = set()
    start = None
    ent_type = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if ent_type is not None:
                spans.add((ent_type, start, i - 1))
            ent_type = tag[2:]
            start = i
        elif tag.startswith("I-") and ent_type == tag[2:]:
            continue
        else:
            if ent_type is not None:
                spans.add((ent_type, start, i - 1))
            ent_type = None
            start = None
    return spans


def precision_recall_f1(true_sequences: Iterable[List[str]], pred_sequences: Iterable[List[str]]):
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_sequences, pred_sequences):
        true_spans = bio_spans(true_tags)
        pred_spans = bio_spans(pred_tags)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
'''

FILES["train.py"] = r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data import PAD_TAG, START_TAG, STOP_TAG, make_dataloaders
from metrics import precision_recall_f1
from model import BiLSTMCRF


def evaluate(model, loader, idx2tag, device):
    model.eval()
    true_sequences, pred_sequences = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            mask = batch["mask"].to(device)
            total_loss += model.neg_log_likelihood(token_ids, tag_ids, mask).item()
            decoded = model(token_ids, mask)
            for gold, pred, length in zip(tag_ids.cpu().tolist(), decoded, batch["lengths"]):
                true_sequences.append([idx2tag[i] for i in gold[:length]])
                pred_sequences.append([idx2tag[i] for i in pred[:length]])
    scores = precision_recall_f1(true_sequences, pred_sequences)
    scores["loss"] = total_loss / max(1, len(loader))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Train a biomedical NER BiLSTM-CRF model.")
    parser.add_argument("--dataset", default="tner/bc5cdr")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loaders, word2idx, tag2idx, idx2tag = make_dataloaders(args.batch_size, args.min_freq, True, args.dataset, args.max_train_samples)
    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        pad_tag_idx=tag2idx[PAD_TAG],
        start_tag_idx=tag2idx[START_TAG],
        stop_tag_idx=tag2idx[STOP_TAG],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            token_ids = batch["token_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(token_ids, tag_ids, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        split = "validation" if "validation" in loaders else "dev"
        scores = evaluate(model, loaders[split], idx2tag, device)
        print(f"Epoch {epoch}: train_loss={total_loss / len(loaders['train']):.4f} val_loss={scores['loss']:.4f} P={scores['precision']:.4f} R={scores['recall']:.4f} F1={scores['f1']:.4f}")
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            torch.save({
                "model_state": model.state_dict(),
                "word2idx": word2idx,
                "tag2idx": tag2idx,
                "idx2tag": idx2tag,
                "args": vars(args),
                "best_f1": best_f1,
            }, checkpoint_dir / "best_model.pt")
            print(f"Saved new best checkpoint to {checkpoint_dir / 'best_model.pt'}")

    test_split = "test" if "test" in loaders else split
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_scores = evaluate(model, loaders[test_split], idx2tag, device)
    print("Final test metrics:", json.dumps(test_scores, indent=2))


if __name__ == "__main__":
    main()
'''

FILES["predict.py"] = r'''from __future__ import annotations

import argparse
import re

import torch

from data import PAD_TAG, START_TAG, STOP_TAG, UNK_TOKEN, normalize_token
from model import BiLSTMCRF


def simple_tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", text)


def collect_entities(tokens, tags):
    entities = {"Chemical/Drug": [], "Disease/Case": []}
    current_type, current_tokens = None, []
    def flush():
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            label = "Chemical/Drug" if current_type.lower().startswith("chem") else "Disease/Case"
            entities[label].append(" ".join(current_tokens))
        current_type, current_tokens = None, []
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            current_type = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            flush()
    flush()
    return entities


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    word2idx = checkpoint["word2idx"]
    tag2idx = checkpoint["tag2idx"]
    idx2tag = {int(k): v for k, v in checkpoint["idx2tag"].items()}
    args = checkpoint.get("args", {})
    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        pad_tag_idx=tag2idx[PAD_TAG],
        start_tag_idx=tag2idx[START_TAG],
        stop_tag_idx=tag2idx[STOP_TAG],
        embedding_dim=args.get("embedding_dim", 100),
        hidden_dim=args.get("hidden_dim", 256),
        dropout=args.get("dropout", 0.25),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, word2idx, idx2tag


def predict_sentence(model, word2idx, idx2tag, sentence, device):
    tokens = simple_tokenize(sentence)
    ids = [word2idx.get(normalize_token(tok), word2idx[UNK_TOKEN]) for tok in tokens]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    with torch.no_grad():
        pred_ids = model(token_ids, mask)[0]
    tags = [idx2tag[i] for i in pred_ids]
    return tokens, tags, collect_entities(tokens, tags)


def print_result(sentence, entities):
    print(f"Sentence: {sentence}")
    print("----------------------------------------")
    print(f"[Chemical/Drug] -> {', '.join(entities['Chemical/Drug']) if entities['Chemical/Drug'] else 'None'}")
    print(f"[Disease/Case]  -> {', '.join(entities['Disease/Case']) if entities['Disease/Case'] else 'None'}")


def main():
    parser = argparse.ArgumentParser(description="Interactive biomedical NER CLI.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word2idx, idx2tag = load_model(args.checkpoint, device)
    print("Biomedical NER is ready. Type a sentence, or 'quit' to exit.")
    while True:
        sentence = input("\nEnter sentence: ").strip()
        if sentence.lower() in {"q", "quit", "exit"}:
            break
        if not sentence:
            continue
        _, _, entities = predict_sentence(model, word2idx, idx2tag, sentence, device)
        print_result(sentence, entities)


if __name__ == "__main__":
    main()
'''


def write_files() -> None:
    for filename, content in FILES.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"wrote {filename}")


def install_requirements() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_training(args) -> None:
    cmd = [
        sys.executable, "train.py",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--embedding-dim", str(args.embedding_dim),
        "--hidden-dim", str(args.hidden_dim),
    ]
    if args.max_train_samples is not None:
        cmd += ["--max-train-samples", str(args.max_train_samples)]
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write and optionally train the biomedical NER workspace.")
    parser.add_argument("--write-only", action="store_true", help="Only generate files; do not install or train.")
    parser.add_argument("--skip-install", action="store_true", help="Do not run pip install before training.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()

    write_files()
    if args.write_only:
        print("Done. Next: python -m pip install -r requirements.txt && python train.py")
        return
    if not args.skip_install:
        install_requirements()
    run_training(args)


if __name__ == "__main__":
    main()
