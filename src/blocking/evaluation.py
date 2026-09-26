"""Blocking recall and candidate-volume measurements against training truth."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

MAIN_METRIC_COLUMNS = [
    "K", "candidate_recall", "full_hit_rate", "avg_candidates", "missed_true_links",
]
DIAGNOSTIC_COLUMNS = [
    "non_singleton_full_hit_rate", "median_candidates", "max_candidates",
    "reduction_ratio", "true_links", "retrieved_true_links", "affected_s1_entities",
    "s1_entities", "non_singleton_s1_entities", "candidate_corpus_records",
]


def format_comparison(comparison: pd.DataFrame, *, include_diagnostics: bool = False) -> pd.DataFrame:
    """Select and round reported columns; diagnostics remain opt-in for future reports."""
    columns = MAIN_METRIC_COLUMNS + (DIAGNOSTIC_COLUMNS if include_diagnostics else [])
    reported = comparison.loc[:, columns].copy()
    reported["candidate_recall"] = reported["candidate_recall"].round(4)
    reported["full_hit_rate"] = reported["full_hit_rate"].round(4)
    reported["avg_candidates"] = reported["avg_candidates"].round(1)
    reported["K"] = reported["K"].astype("int64")
    reported["missed_true_links"] = reported["missed_true_links"].astype("int64")
    return reported


def format_comparison_tsv(comparison: pd.DataFrame) -> pd.DataFrame:
    """Render the compact table with the requested fixed decimal places."""
    rendered = format_comparison(comparison)
    rendered["candidate_recall"] = rendered["candidate_recall"].map(lambda value: f"{value:.4f}")
    rendered["full_hit_rate"] = rendered["full_hit_rate"].map(lambda value: f"{value:.4f}")
    rendered["avg_candidates"] = rendered["avg_candidates"].map(lambda value: f"{value:.1f}")
    return rendered


def evaluate_top_k(candidates: pd.DataFrame, truth: dict[str, set[str]],
                   corpus_size: int, k_values: list[int]):
    if corpus_size < 1 or not truth:
        raise ValueError("Need a nonempty candidate corpus and S1 truth set")
    if candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise ValueError("Duplicate candidate pair in blocker output")
    if not set(candidates["source1_entity_id"]).issubset(truth):
        raise ValueError("Candidate output includes S1 IDs outside evaluation set")
    total_true = sum(map(len, truth.values()))
    nonsingleton = sum(bool(links) for links in truth.values())
    comparison = []
    missed_rows = []
    for k in k_values:
        selected = candidates.loc[candidates["rank"] <= k]
        found = defaultdict(set)
        for s1_id, candidate_id in zip(selected["source1_entity_id"], selected["candidate_entity_id"]):
            found[str(s1_id)].add(str(candidate_id))
        retrieved_true = sum(len(links & found[s1_id]) for s1_id, links in truth.items())
        full_hits = sum(links <= found[s1_id] for s1_id, links in truth.items())
        nonsingleton_full_hits = sum(bool(links) and links <= found[s1_id]
                                     for s1_id, links in truth.items())
        counts = [len(found[s1_id]) for s1_id in truth]
        for s1_id, links in truth.items():
            for candidate_id in sorted(links - found[s1_id]):
                missed_rows.append({"K": k, "source1_entity_id": s1_id,
                                    "candidate_entity_id": candidate_id})
        comparison.append({
            "K": k,
            "candidate_recall": retrieved_true / total_true if total_true else None,
            "full_hit_rate": full_hits / len(truth),
            "non_singleton_full_hit_rate": (nonsingleton_full_hits / nonsingleton
                                            if nonsingleton else None),
            "avg_candidates": sum(counts) / len(counts),
            "median_candidates": float(pd.Series(counts).median()),
            "max_candidates": max(counts),
            "reduction_ratio": 1 - sum(counts) / (len(truth) * corpus_size),
            "true_links": total_true,
            "retrieved_true_links": retrieved_true,
            "missed_true_links": total_true - retrieved_true,
            "affected_s1_entities": sum(bool(links - found[s1_id]) for s1_id, links in truth.items()),
            "s1_entities": len(truth),
            "non_singleton_s1_entities": nonsingleton,
            "candidate_corpus_records": corpus_size,
        })
    return pd.DataFrame(comparison), pd.DataFrame(
        missed_rows, columns=["K", "source1_entity_id", "candidate_entity_id"])
