"""Bounded-memory character TF-IDF retrieval over the complete S2/S3 corpus."""

from __future__ import annotations

import heapq
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from src.blocking.base import Blocker, CANDIDATE_COLUMNS
from src.data.blocking_data import iter_source
from src.features.normalization import normalize_business_address, normalize_business_name

LOG = logging.getLogger(__name__)


class CharTfidfBlocker(Blocker):
    """Hash character n-grams, compute corpus IDF, then stream top-K sparse cosines.

    Hashing bounds vocabulary memory. A hash collision can slightly alter scores;
    n_features is configurable. IDF is measured over every S2/S3 record, and
    every S2/S3 record is searched at retrieval time. No true links are inserted.
    """

    method = "char_tfidf"

    def __init__(self, *, field: str = "business_name", ngram_range=(2, 5),
                 min_df: int = 2, n_features: int = 2**18,
                 normalization: dict | None = None, chunk_size: int = 50_000,
                 query_batch_size: int = 16):
        if field not in {"business_name", "business_address"}:
            raise ValueError("field must be business_name or business_address")
        if not (1 <= ngram_range[0] <= ngram_range[1]):
            raise ValueError("invalid ngram_range")
        if min_df < 1 or n_features < 2 or chunk_size < 1 or query_batch_size < 1:
            raise ValueError("blocking sizes must be positive")
        self.field = field
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.n_features = n_features
        self.normalization = normalization or {}
        self.chunk_size = chunk_size
        self.query_batch_size = query_batch_size
        self.vectorizer = HashingVectorizer(
            analyzer="char", ngram_range=self.ngram_range,
            n_features=n_features, alternate_sign=False, norm=None,
            lowercase=False, dtype=np.float32,
        )
        self.sources: tuple[Path, Path] | None = None
        self.idf: np.ndarray | None = None
        self.corpus_count = 0

    def _normalized(self, frame: pd.DataFrame) -> list[str]:
        fn = normalize_business_name if self.field == "business_name" else normalize_business_address
        return [fn(value, self.normalization) for value in frame[self.field]]

    def _weighted(self, texts: list[str]):
        if self.idf is None:
            raise RuntimeError("Call fit() first")
        matrix = self.vectorizer.transform(texts).tocsr()
        matrix = matrix.multiply(self.idf).tocsr()
        l2_normalize(matrix, norm="l2", copy=False)
        return matrix

    def fit(self, source2: Path, source3: Path) -> "CharTfidfBlocker":
        self.sources = (Path(source2), Path(source3))
        document_frequency = np.zeros(self.n_features, dtype=np.int64)
        count = 0
        for path in self.sources:
            for chunk in iter_source(path, self.chunk_size):
                matrix = self.vectorizer.transform(self._normalized(chunk))
                document_frequency += np.asarray(matrix.astype(bool).sum(axis=0)).ravel()
                count += len(chunk)
                if count % 500_000 == 0:
                    LOG.info("TF-IDF document frequencies: %s candidate records", f"{count:,}")
        self.idf = (np.log((1 + count) / (1 + document_frequency)) + 1).astype(np.float32)
        self.idf[document_frequency < self.min_df] = 0.0
        self.corpus_count = count
        LOG.info("Fitted corpus IDF on %s S2/S3 records", f"{count:,}")
        return self

    def retrieve(self, source1: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
        if self.sources is None or self.idf is None:
            raise RuntimeError("Call fit() before retrieve()")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if source1["entity_id"].duplicated().any():
            raise ValueError("Duplicate S1 IDs in retrieval input")
        query_ids = source1["entity_id"].astype(str).tolist()
        query_matrix = self._weighted(self._normalized(source1))
        heaps: list[list[tuple[float, str]]] = [[] for _ in query_ids]
        seen = 0
        for path in self.sources:
            for chunk in iter_source(path, self.chunk_size):
                candidate_ids = chunk["entity_id"].astype(str).to_numpy()
                candidate_matrix = self._weighted(self._normalized(chunk))
                candidate_transpose = candidate_matrix.T.tocsr()
                for start in range(0, len(query_ids), self.query_batch_size):
                    # At most query_batch_size x chunk_size scores exist here.
                    # Sparse multiplication avoids a full S1 x corpus matrix.
                    scores = (query_matrix[start:start + self.query_batch_size]
                              @ candidate_transpose).tocsr()
                    for offset in range(scores.shape[0]):
                        row = start + offset
                        heap = heaps[row]
                        begin, end = scores.indptr[offset:offset + 2]
                        values = scores.data[begin:end]
                        columns = scores.indices[begin:end]
                        if len(values) > top_k:
                            chosen = np.argpartition(values, -top_k)[-top_k:]
                            values, columns = values[chosen], columns[chosen]
                        for col, score in zip(columns, values):
                            item = (float(score), str(candidate_ids[col]))
                            if len(heap) < top_k:
                                heapq.heappush(heap, item)
                            elif item > heap[0]:
                                heapq.heapreplace(heap, item)
                seen += len(chunk)
                if seen % 500_000 == 0:
                    LOG.info("Searched %s candidate records", f"{seen:,}")
        rows = []
        for entity_id, heap in zip(query_ids, heaps):
            for rank, (score, candidate_id) in enumerate(sorted(heap, reverse=True), 1):
                rows.append((entity_id, candidate_id, score, self.method, rank))
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
