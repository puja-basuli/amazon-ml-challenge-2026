#!/usr/bin/env python3
"""Run a few real prediction batches through the configured Windows process pool."""

from __future__ import annotations

import json
import sys
import time
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.index import iter_records, source_path
from src.pipelines.baseline import _init_prediction_worker, _predict_batch


def main() -> None:
    output = ROOT / "outputs/baseline_recovered"
    dataset = ROOT / "data/raw/student_resource/dataset"
    metrics = json.loads((output / "baseline_metrics.json").read_text())
    records = iter_records(source_path(dataset, "test", 1))
    batches = [list(islice(records, 50)) for _ in range(4)]
    started = time.perf_counter()
    with get_context("spawn").Pool(
        4,
        initializer=_init_prediction_worker,
        initargs=(output / "test_candidates.sqlite", output / "baseline_model.joblib",
                  metrics["threshold"], metrics["per_key"], metrics["final_limit"]),
    ) as pool:
        for i, result in enumerate(pool.imap(_predict_batch, batches, chunksize=1), 1):
            print(f"batch={i} rows={len(result)} seconds={time.perf_counter()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
