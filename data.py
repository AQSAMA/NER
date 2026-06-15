from __future__ import annotations

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
