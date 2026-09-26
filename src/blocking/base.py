"""Common blocker interface and candidate schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_id",
                     "blocking_score", "blocking_method", "rank"]


class Blocker(ABC):
    method: str

    @abstractmethod
    def fit(self, source2: Path, source3: Path) -> "Blocker":
        """Prepare representations from the complete candidate sources."""

    @abstractmethod
    def retrieve(self, source1: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
        """Return ranked rows with CANDIDATE_COLUMNS."""


def union_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine later blocker outputs; retain the best score per ID pair."""
    if not frames:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return (combined.sort_values("blocking_score", ascending=False)
            .drop_duplicates(["source1_entity_id", "candidate_entity_id"])
            .reset_index(drop=True))
