from __future__ import annotations

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
