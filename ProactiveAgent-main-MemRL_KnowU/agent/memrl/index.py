from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class TfidfIndex:
    def __init__(self) -> None:
        self.documents: list[dict[str, float]] = []
        self.doc_ids: list[str] = []
        self.idf: dict[str, float] = {}
        self.postings: dict[str, set[int]] = {}

    def build(self, doc_pairs: Iterable[tuple[str, str]]) -> None:
        pairs = list(doc_pairs)
        self.doc_ids = [doc_id for doc_id, _ in pairs]
        tokenized = [tokenize(text) for _, text in pairs]
        doc_freq: Counter[str] = Counter()
        self.postings = {}
        for idx, tokens in enumerate(tokenized):
            doc_freq.update(set(tokens))
            for token in set(tokens):
                self.postings.setdefault(token, set()).add(idx)
        total = max(len(tokenized), 1)
        self.idf = {
            token: math.log((1 + total) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }
        self.documents = [self._vectorize_tokens(tokens) for tokens in tokenized]

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(sum(counts.values()), 1)
        return {
            token: (count / total) * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }

    def vectorize(self, text: str) -> dict[str, float]:
        return self._vectorize_tokens(tokenize(text))

    def search(self, text: str, *, topk: int = 8) -> list[tuple[str, float]]:
        query_tokens = tokenize(text)
        query = self._vectorize_tokens(query_tokens)
        candidate_indices: set[int] = set()
        for token in set(query_tokens):
            candidate_indices.update(self.postings.get(token, set()))
        if not candidate_indices:
            candidate_indices = set(range(len(self.doc_ids)))
        scored = [
            (self.doc_ids[idx], cosine_similarity(query, self.documents[idx]))
            for idx in candidate_indices
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topk]
