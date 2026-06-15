from __future__ import annotations

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
