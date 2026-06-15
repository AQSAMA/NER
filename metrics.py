from __future__ import annotations

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
