from __future__ import annotations

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
