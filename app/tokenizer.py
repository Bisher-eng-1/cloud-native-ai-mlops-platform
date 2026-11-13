from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9_]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class TextTokenizer:
    """Lightweight deterministic word/sub-token tokenizer saved as plain JSON."""

    def __init__(self, vocab: dict[str, int], max_seq_len: int = 48) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_id = vocab[PAD_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    @classmethod
    def fit(cls, texts: list[str], max_vocab_size: int = 2048, max_seq_len: int = 48) -> TextTokenizer:
        counts: dict[str, int] = {}
        for text in texts:
            for token in cls.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, _ in sorted_tokens[: max(0, max_vocab_size - 2)]:
            vocab[token] = len(vocab)
        return cls(vocab=vocab, max_seq_len=max_seq_len)

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray, int]:
        tokens = self.tokenize(text)[: self.max_seq_len]
        token_ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]
        valid_len = len(token_ids)
        attention_mask = [1] * valid_len + [0] * (self.max_seq_len - valid_len)
        padded_ids = token_ids + [self.pad_id] * (self.max_seq_len - valid_len)
        return (
            np.asarray(padded_ids, dtype=np.int64),
            np.asarray(attention_mask, dtype=np.int64),
            valid_len,
        )

    def encode_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        ids_list = []
        mask_list = []
        lengths = []
        for text in texts:
            ids, mask, length = self.encode(text)
            ids_list.append(ids)
            mask_list.append(mask)
            lengths.append(length)
        return np.stack(ids_list, axis=0), np.stack(mask_list, axis=0), lengths

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"max_seq_len": self.max_seq_len, "vocab": self.vocab}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TextTokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab=payload["vocab"], max_seq_len=int(payload["max_seq_len"]))
