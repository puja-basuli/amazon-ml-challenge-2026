#!/usr/bin/env python3
"""Benchmark multi-pass blocking on sampled train S1 vs the full train S2/S3 corpus."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.index import build_index, open_index, source_path
from src.pipelines.baseline import (
    PeakMemorySampler,
    ROUTE_NAMES,
    compress_candidates_many,
    load_sampled_anchors,
    load_truth,
    route_candidates,
)

LOG = logging.getLogger(__name__)


def _partition(entity_id: str) -> str:
    fold = zlib.crc32((entity_id + "fold").encode()) % 10
    return "calibration" if fold == 0 else "validation" if fold == 1 else "train"


def summarize_counts(counts: list[int], corpus_size: int) -> dict:
    values = np.asarray(counts, dtype=np.int64)
    total = int(values.sum())
    return {
        "s1_entities": int(len(values)),
        "total_candidates": total,
        "average_candidates_per_s1": float(values.mean()) if len(values) else 0.0,
        "p95_candidates_per_s1": float(np.percentile(values, 95)) if len(values) else 0.0,
        "p99_candidates_per_s1": float(np.percentile(values, 99)) if len(values) else 0.0,
        "zero_candidate_s1": int((values == 0).sum()),
        "over_100_candidates_s1": int((values > 100).sum()),
        "over_1000_candidates_s1": int((values > 1000).sum()),
        "candidate_reduction_ratio": 1 - total / (len(values) * corpus_size) if len(values) and corpus_size else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/student_resource/dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline")
    parser.add_argument("--index-path", type=Path, default=None,
                        help="reuse an existing compatible SQLite training index")
    parser.add_argument("--sample-divisor", type=int, default=1000)
    parser.add_argument("--partition", choices=("all", "train", "calibration", "validation"),
                        default="all",
                        help="optionally benchmark one deterministic training fold")
    parser.add_argument("--per-key-values", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--final-limits", type=int, nargs="+", default=[80, 120, 200])
    args = parser.parse_args()
    if args.sample_divisor < 1 or any(x < 1 for x in args.per_key_values + args.final_limits):
        parser.error("sample divisor and all candidate limits must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    db_path = args.index_path or output / "train_candidates.sqlite"
    memory = PeakMemorySampler()
    memory.start()
    started = time.perf_counter()
    if args.index_path is not None and not db_path.exists():
        parser.error(f"index does not exist: {db_path}")
    build_index(dataset, "train", db_path)
    index_seconds = time.perf_counter() - started
    anchors = load_sampled_anchors(dataset, args.sample_divisor)
    if args.partition != "all":
        anchors = [row for row in anchors if _partition(row[0]) == args.partition]
    truth = load_truth(dataset, {row[0] for row in anchors})
    conn = open_index(db_path)
    corpus_size = int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    conn.close()
    results = {}
    total_truth_links = sum(len(links) for links in truth.values())
    source_true = {prefix[:2]: sum(sum(value.startswith(prefix) for value in links)
                                   for links in truth.values())
                   for prefix in ("S2-", "S3-")}
    for per_key in args.per_key_values:
        counts_by_limit = {limit: [] for limit in args.final_limits}
        raw_union_links = 0
        source_route_counts = {route: Counter() for route in ROUTE_NAMES}
        source_route_hits = {route: Counter() for route in ROUTE_NAMES}
        route_unique_link_hits = Counter()
        route_unique_candidate_hits = Counter()
        source_recalled = {limit: Counter() for limit in args.final_limits}
        full_hits = Counter()
        conn = open_index(db_path)
        for number, anchor in enumerate(anchors, 1):
            actual = truth[anchor[0]]
            route_map = route_candidates(conn, anchor, per_key=per_key)
            for route, rows in route_map.items():
                candidate_ids = set(rows)
                source_route_counts[route]["candidates"] += len(candidate_ids)
                for prefix in ("S2-", "S3-"):
                    source_route_counts[route][prefix[:2]] += sum(x.startswith(prefix) for x in candidate_ids)
                    source_route_hits[route][prefix[:2]] += sum(x in candidate_ids for x in actual if x.startswith(prefix))
            union = {entity_id for rows in route_map.values() for entity_id in rows}
            raw_union_links += len(actual & union)
            for route, rows in route_map.items():
                other_candidates = set().union(*(set(values) for other, values in route_map.items() if other != route))
                unique_candidates = set(rows) - other_candidates
                route_unique_candidate_hits[route] += len(unique_candidates)
                route_unique_link_hits[route] += len(actual & unique_candidates)
            candidates_by_limit = compress_candidates_many(anchor, route_map, args.final_limits)
            for limit in args.final_limits:
                candidates = candidates_by_limit[limit]
                candidate_ids = {row[0] for row in candidates}
                counts_by_limit[limit].append(len(candidate_ids))
                for prefix in ("S2-", "S3-"):
                    source_recalled[limit][prefix[:2]] += sum(x in candidate_ids for x in actual if x.startswith(prefix))
                if actual and actual <= candidate_ids:
                    full_hits[limit] += 1
            if number % 500 == 0:
                LOG.info("per_key=%s: benchmarked %s/%s sampled S1", per_key, f"{number:,}", f"{len(anchors):,}")
        conn.close()
        for limit in args.final_limits:
            key = f"per_key_{per_key}_final_{limit}"
            candidate_stats = summarize_counts(counts_by_limit[limit], corpus_size)
            recalled = sum(source_recalled[limit].values())
            candidate_stats.update({
                "raw_union_blocking_recall": raw_union_links / max(1, total_truth_links),
                "compressed_blocking_recall": recalled / max(1, total_truth_links),
                "source_blocking_recall": {
                    source: source_recalled[limit][source] / max(1, source_true[source])
                    for source in ("S2", "S3")
                },
                "full_hit_rate_non_singletons": full_hits[limit] / max(1, sum(bool(x) for x in truth.values())),
                "source_true_links": source_true,
                "source_recalled_links": dict(source_recalled[limit]),
            })
            results[key] = candidate_stats
        for route in ROUTE_NAMES:
            results[f"per_key_{per_key}_route_{route}"] = {
                "raw_candidates_per_s1": source_route_counts[route]["candidates"] / max(1, len(anchors)),
                "raw_candidate_true_link_recall": sum(source_route_hits[route].values()) / max(1, total_truth_links),
                "source_candidate_recall": {
                    source: source_route_hits[route][source] / max(1, source_true[source])
                    for source in ("S2", "S3")
                },
                "unique_candidates_added_per_s1": route_unique_candidate_hits[route] / max(1, len(anchors)),
                "unique_true_links_added": route_unique_link_hits[route],
            }
    results["sampled_s1"] = len(anchors)
    results["full_training_candidate_corpus"] = corpus_size
    results["sample_divisor"] = args.sample_divisor
    results["partition"] = args.partition
    results["per_key_values"] = args.per_key_values
    results["final_limits"] = args.final_limits
    results["index_runtime_seconds"] = index_seconds
    results["runtime_seconds"] = time.perf_counter() - started
    results["peak_memory_bytes"] = memory.stop()
    result_path = output / "blocking_benchmark.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    LOG.info("Saved blocking benchmark: %s", result_path)


if __name__ == "__main__":
    main()
