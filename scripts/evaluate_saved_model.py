#!/usr/bin/env python3
"""Evaluate a saved baseline model on the deterministic validation fold."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.index import build_index, open_index
from src.pipelines.baseline import (
    PeakMemorySampler, ROUTE_NAMES, candidate_features, compress_candidates_many,
    load_sampled_anchors, load_truth, macro_f05, macro_f1, route_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/student_resource/dataset")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-divisor", type=int, default=1000)
    parser.add_argument("--per-key", type=int, default=500)
    parser.add_argument("--caps", type=int, nargs="+", default=[2400, 3200, 4000])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    started = time.perf_counter()
    memory = PeakMemorySampler()
    memory.start()
    anchors = [row for row in load_sampled_anchors(dataset, args.sample_divisor)
               if zlib.crc32((row[0] + "fold").encode()) % 10 == 1]
    truth = load_truth(dataset, {row[0] for row in anchors})
    model = joblib.load(model_path)
    model.set_params(n_jobs=1)
    conn = open_index(index_path)
    scored = []
    counts = {cap: [] for cap in args.caps}
    try:
        for number, anchor in enumerate(anchors, 1):
            routes = route_candidates(conn, anchor, per_key=args.per_key)
            by_cap = compress_candidates_many(anchor, routes, args.caps)
            candidates = by_cap[max(args.caps)]
            features = candidate_features(anchor, candidates)
            probs = (model.predict_proba(pd.DataFrame(features, columns=model.feature_name_))[:, 1]
                     if len(candidates) else np.empty(0))
            # Caps are nested prefixes of the same relevance-sorted union.
            candidate_pos = {row[0]: idx for idx, row in enumerate(candidates)}
            for cap in args.caps:
                counts[cap].append(len(by_cap[cap]))
            probability_by_id = {row[0]: float(probability)
                                 for row, probability in zip(candidates, probs)}
            scored.append((anchor[0], truth[anchor[0]], by_cap, probability_by_id))
            if number % 50 == 0:
                logging.info("Scored %s/%s validation anchors", number, len(anchors))
    finally:
        conn.close()
    thresholds = sorted(set([round(x, 3) for x in np.arange(0.05, 1.0, 0.05)]
                            + [round(x, 4) for x in np.arange(0.90, 1.0, 0.01)]
                            + [0.991, 0.9925, 0.995, 0.9975, 0.999, 0.9995, 0.9999]))
    report = {"validation_s1": len(anchors), "per_key": args.per_key,
              "runtime_seconds": time.perf_counter() - started, "by_cap": {}}
    for cap in args.caps:
        cap_truth = {entity_id: actual for entity_id, actual, *_ in scored}
        cap_recall = 0
        total_truth = sum(map(len, cap_truth.values()))
        for entity_id, actual, by_cap, _ in scored:
            cap_recall += sum(row[0] in actual for row in by_cap[cap])
        best = {"f1": (-1, None), "f05": (-1, None)}
        for threshold in thresholds:
            predictions = {}
            for entity_id, _, by_cap, probability_by_id in scored:
                predictions[entity_id] = {row[0] for row in by_cap[cap]
                                          if probability_by_id.get(row[0], 0.0) >= threshold}
            f1 = macro_f1(cap_truth, predictions)
            f05 = macro_f05(cap_truth, predictions)
            if f1 > best["f1"][0]:
                best["f1"] = (f1, threshold)
            if f05 > best["f05"][0]:
                best["f05"] = (f05, threshold)
        report["by_cap"][str(cap)] = {
            "candidate_recall": cap_recall / max(1, total_truth),
            "average_candidates": float(np.mean(counts[cap])),
            "p95_candidates": float(np.percentile(counts[cap], 95)),
            "best_macro_f1": best["f1"][0], "best_f1_threshold": best["f1"][1],
            "best_macro_f05": best["f05"][0], "best_f05_threshold": best["f05"][1],
        }
    report["peak_memory_bytes"] = memory.stop()
    report["runtime_seconds"] = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
