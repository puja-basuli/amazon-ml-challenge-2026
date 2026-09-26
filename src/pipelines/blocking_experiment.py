"""Config-driven blocking run with exact full-corpus retrieval and reusable artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from src.blocking.char_tfidf import CharTfidfBlocker
from src.blocking.evaluation import evaluate_top_k, format_comparison, format_comparison_tsv
from src.data.blocking_data import load_ground_truth, load_source1_sample, source_path

LOG = logging.getLogger(__name__)
PIPELINE_VERSION = 1
BLOCKER_CODE = [
    "src/features/normalization.py", "src/data/blocking_data.py",
    "src/blocking/base.py", "src/blocking/char_tfidf.py",
]


def _signature(paths: list[Path]) -> dict:
    return {str(path.resolve()): {"bytes": path.stat().st_size,
                                  "modified_ns": path.stat().st_mtime_ns}
            for path in paths}


def run_blocking_experiment(config: dict, root: Path) -> dict:
    experiment_name = config.get("experiment_name", "")
    if not isinstance(experiment_name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", experiment_name):
        raise ValueError("experiment_name must contain only letters, numbers, underscore, or hyphen")
    dataset_value = Path(config.get("dataset", "data/raw/student_resource/dataset"))
    dataset = dataset_value if dataset_value.is_absolute() else root / dataset_value
    method = config.get("blocking", {}).get("method")
    if method != "char_tfidf":
        raise ValueError("Supported blocking method: char_tfidf")
    settings = config["blocking"]
    ks = sorted(set(int(k) for k in settings.get("k_values", [5, 10, 20, 50, 100, 200])))
    if not ks or ks[0] < 1:
        raise ValueError("blocking.k_values must contain positive integers")
    chunk_size = int(settings.get("chunk_size", 50_000))
    divisor = int(settings.get("s1_sample_divisor", 4_000))
    output_root = root / "outputs/experiments" / experiment_name
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / run_id
    source_paths = [source_path(dataset, source) for source in (1, 2, 3)]
    truth_path = dataset / "train/train_ground_truth.tsv"
    inputs = source_paths + [truth_path]
    code_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in BLOCKER_CODE}
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "config": config,
        "input_files": _signature(inputs),
        "blocker_code_sha256": code_hashes,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scikit_learn": sklearn.__version__, "pyarrow": pyarrow.__version__},
    }
    source1, full_s1_count = load_source1_sample(source_paths[0], divisor=divisor,
                                                  chunk_size=chunk_size)
    truth = load_ground_truth(truth_path, set(source1["entity_id"].astype(str)), chunk_size)
    LOG.info("Evaluating %s sampled S1 records from %s total", f"{len(source1):,}",
             f"{full_s1_count:,}")
    blocker = CharTfidfBlocker(
        field=settings.get("field", "business_name"),
        ngram_range=settings.get("ngram_range", [2, 5]),
        min_df=int(settings.get("min_df", 2)),
        n_features=int(settings.get("n_features", 2**18)),
        normalization=settings.get("normalization", {}),
        chunk_size=chunk_size,
        query_batch_size=int(settings.get("query_batch_size", 64)),
    )
    blocker.fit(source_paths[1], source_paths[2])
    candidates = blocker.retrieve(source1, top_k=max(ks))
    corpus_count = blocker.corpus_count
    comparison, missed = evaluate_top_k(candidates, truth, corpus_count, ks)
    reported_comparison = format_comparison(comparison)
    result = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "sampled_s1_records": len(source1),
        "all_train_s1_records": full_s1_count,
        "candidate_corpus_records": corpus_count,
        "comparison": reported_comparison.to_dict(orient="records"),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "candidates.parquet"
    candidates.to_parquet(artifact_path, index=False)
    result["candidate_artifact"] = str(artifact_path)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    format_comparison_tsv(comparison).to_csv(output / "metrics.csv", index=False)
    format_comparison_tsv(comparison).to_csv(output / "comparison.tsv", sep="\t", index=False)
    missed.to_csv(output / "missed_pairs.tsv", sep="\t", index=False)
    (output / "manifest.json").write_text(json.dumps(signature, indent=2) + "\n")
    LOG.info("Saved candidates and evaluation under %s", output)
    return result
