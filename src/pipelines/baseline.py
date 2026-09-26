"""First complete candidate, matcher, validation and submission workflow."""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
import zlib
from collections import Counter
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from src.data.index import build_index, iter_records, open_index, source_path
from src.features.text import (
    FEATURE_NAMES, approximate_grams, extract_geography, features,
    informative_tokens, jaccard, keys, length_ratio, normalize, normalize_country,
    numbers, numeric_components, tokens,
)

LOG = logging.getLogger(__name__)
_WORKER = {}


class PeakMemorySampler:
    """Sample this process and its children to record peak working set bytes."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.process = psutil.Process()
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        total = 0
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = [self.process]
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        self.peak_bytes = max(self.peak_bytes, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()
        return self.peak_bytes


def sample_anchor(entity_id: str, divisor: int) -> bool:
    return zlib.crc32(entity_id.encode("utf-8")) % divisor == 0


def load_sampled_anchors(dataset: Path, divisor: int) -> list[tuple[str, str, str, str]]:
    anchors = [row for row in iter_records(source_path(dataset, "train", 1))
               if sample_anchor(row[0], divisor)]
    if not anchors:
        raise ValueError("No training S1 records selected; decrease --sample-divisor")
    LOG.info("Selected %s training S1 records", f"{len(anchors):,}")
    return anchors


def load_truth(dataset: Path, wanted_ids: set[str]) -> dict[str, set[str]]:
    path = dataset / "train" / "train_ground_truth.tsv"
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or []) != {"source1_entity_id", "matched_entity_ids"}:
            raise ValueError(f"Unexpected ground-truth columns: {reader.fieldnames}")
        for row in reader:
            entity_id = row["source1_entity_id"]
            if entity_id in wanted_ids:
                if entity_id in result:
                    raise ValueError(f"Duplicate ground-truth row for {entity_id}")
                result[entity_id] = {value.strip() for value in row["matched_entity_ids"].split(",")
                                     if value.strip()}
    if result.keys() != wanted_ids:
        raise ValueError(f"Ground truth missing {len(wanted_ids - result.keys())} selected S1 IDs")
    return result


ROUTE_NAMES = (
    "country_state_pin", "country_state_name", "country_pin_name", "country_city_name",
    "country_rare_name", "country_address_token", "country_numeric",
    "approximate_name", "approximate_address",
)


def route_candidates(conn, anchor: tuple[str, str, str, str], *, per_key: int = 30,
                     routes: set[str] | None = None, approximate_trigger: int = 20
                     ) -> dict[str, dict[str, tuple[str, str, str, str]]]:
    """Fetch bounded per-source postings for each independent blocking route."""
    enabled = set(ROUTE_NAMES) if routes is None else set(routes)
    unknown = enabled - set(ROUTE_NAMES)
    if unknown:
        raise ValueError(f"Unknown blocking routes: {sorted(unknown)}")
    country = normalize_country(anchor[3])
    state, postal, city = extract_geography(anchor[2], anchor[3])
    name_token = next(iter(informative_tokens(anchor[1], name=True, limit=1)), "")
    address_token = next(iter(informative_tokens(anchor[2], name=False, limit=1)), "")
    numbers = numeric_components(anchor[2], limit=2)
    name_grams = approximate_grams(anchor[1], name=True, limit=3)
    address_grams = approximate_grams(anchor[2], name=False, limit=3)

    specs = []
    if "country_state_pin" in enabled and country and state and postal:
        specs.append(("country_state_pin", [("country_key", country), ("state_key", state), ("pin_key", postal)], ["source"]))
    if "country_state_name" in enabled and country and state and name_token:
        specs.append(("country_state_name", [("country_key", country), ("state_key", state), ("name_token", name_token)], ["source"]))
    if "country_pin_name" in enabled and country and postal and name_token:
        specs.append(("country_pin_name", [("country_key", country), ("pin_key", postal), ("name_token", name_token)], ["source"]))
    if "country_city_name" in enabled and country and city and name_token:
        specs.append(("country_city_name", [("country_key", country), ("city_key", city), ("name_token", name_token)], ["source"]))
    if "country_rare_name" in enabled and name_token:
        filters = [("country_key", country), ("name_token", name_token)] if country else [("name_token", name_token)]
        specs.append(("country_rare_name", filters, ["source"]))
    if "country_address_token" in enabled and address_token:
        filters = [("country_key", country), ("address_token", address_token)] if country else [("address_token", address_token)]
        specs.append(("country_address_token", filters, ["source"]))
    if "country_numeric" in enabled:
        filters = [("country_key", country)] if country else []
        for number in numbers:
            specs.append(("country_numeric", filters + [("number1", number)], ["source"]))
            specs.append(("country_numeric", filters + [("number2", number)], ["source"]))

    results: dict[str, dict[str, tuple[str, str, str, str]]] = {name: {} for name in ROUTE_NAMES}

    def retrieve(route: str, filters: list[tuple[str, str]], cap: int) -> None:
        if not filters or any(not value for _, value in filters):
            return
        where = " AND ".join(f"{field}=?" for field, _ in filters)
        sql = ("SELECT entity_id, name, address, country, source FROM records WHERE "
               + where + " AND source=? LIMIT ?")
        parameters = [value for _, value in filters]
        for source in ("S2", "S3"):
            rows = conn.execute(sql, (*parameters, source, cap)).fetchall()
            for entity_id, name, address, row_country, row_source in rows:
                results[route][entity_id] = (entity_id, name, address, row_country)

    for route, filters, _ in specs:
        retrieve(route, filters, per_key)

    # Weak numeric/address-token hits must not suppress the approximate fallback.
    # Otherwise common street numbers can make the "exact" union look large even
    # when no geography/name route retrieved anything useful.
    core_routes = {"country_state_pin", "country_state_name", "country_pin_name",
                   "country_city_name", "country_rare_name"}
    core_count = len({entity_id for route, rows in results.items()
                      if route in core_routes for entity_id in rows})
    if core_count < approximate_trigger:
        for i, gram in enumerate(name_grams, 1):
            if "approximate_name" in enabled:
                filters = [("country_key", country), (f"name_gram{i}", gram)] if country else [(f"name_gram{i}", gram)]
                retrieve("approximate_name", filters, per_key)
        for i, gram in enumerate(address_grams, 1):
            if "approximate_address" in enabled:
                filters = [("country_key", country), (f"address_gram{i}", gram)] if country else [(f"address_gram{i}", gram)]
                retrieve("approximate_address", filters, per_key)
    return results


def _candidate_rank(anchor, row) -> tuple[float, float, str]:
    name_a, name_b = normalize(anchor[1], name=True), normalize(row[1], name=True)
    addr_a, addr_b = normalize(anchor[2]), normalize(row[2])
    geo_a = extract_geography(anchor[2], anchor[3])
    geo_b = extract_geography(row[2], row[3])
    name_score = max(fuzz.ratio(name_a, name_b), fuzz.token_set_ratio(name_a, name_b))
    address_score = max(fuzz.ratio(addr_a, addr_b), fuzz.token_set_ratio(addr_a, addr_b))
    geo_bonus = 18 * bool(geo_a[1] and geo_a[1] == geo_b[1]) + 8 * bool(geo_a[0] and geo_a[0] == geo_b[0])
    geo_bonus += 5 * bool(geo_a[2] and geo_a[2] == geo_b[2])
    return (0.58 * name_score + 0.27 * address_score + geo_bonus,
            name_score + address_score, row[0])


def compress_candidates(anchor, by_route, final_limit: int = 120):
    union = {}
    for rows in by_route.values():
        union.update(rows)
    if len(union) <= final_limit:
        return list(union.values())
    scored = sorted(((*_candidate_rank(anchor, row), row) for row in union.values()),
                    key=lambda item: (-item[0], -item[1], item[2]))
    quota = final_limit // 2
    selected = []
    selected_ids = set()
    for source in ("S2-", "S3-"):
        for item in (value for value in scored if value[2].startswith(source)):
            if sum(r[0].startswith(source) for r in selected) >= quota:
                break
            selected.append(item[3])
            selected_ids.add(item[2])
    for item in scored:
        if len(selected) >= final_limit:
            break
        if item[2] not in selected_ids:
            selected.append(item[3])
            selected_ids.add(item[2])
    return selected


def generate_candidates(conn, anchor: tuple[str, str, str, str], *, per_key: int = 30,
                        final_limit: int = 120) -> list[tuple[str, str, str, str]]:
    """Return the compressed union that the classifier actually scores."""
    return compress_candidates(anchor, route_candidates(conn, anchor, per_key=per_key), final_limit)


def candidate_features(anchor, candidates):
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    # Batch string comparisons in RapidFuzz's C implementation. This avoids
    # millions of Python-level scorer calls on the full test set.
    name_a = normalize(anchor[1], name=True)
    addr_a = normalize(anchor[2])
    name_b = [normalize(row[1], name=True) for row in candidates]
    addr_b = [normalize(row[2]) for row in candidates]
    name_ratio = process.cdist([name_a], name_b, scorer=fuzz.ratio, workers=1,
                               dtype=np.float32)[0] / 100
    name_token_ratio = process.cdist([name_a], name_b, scorer=fuzz.token_set_ratio,
                                     workers=1, dtype=np.float32)[0] / 100
    name_jaro = process.cdist([name_a], name_b, scorer=JaroWinkler.normalized_similarity,
                              workers=1, dtype=np.float32)[0]
    addr_ratio = process.cdist([addr_a], addr_b, scorer=fuzz.ratio, workers=1,
                               dtype=np.float32)[0] / 100
    addr_token_ratio = process.cdist([addr_a], addr_b, scorer=fuzz.token_set_ratio,
                                     workers=1, dtype=np.float32)[0] / 100

    name_tokens_a, addr_tokens_a = tokens(name_a), tokens(addr_a)
    numbers_a = numbers(addr_a)
    postal_a = {value for value in numbers_a if 5 <= len(value) <= 6}
    geo_a = extract_geography(anchor[2], anchor[3])
    country_a = normalize_country(anchor[3])
    rows = []
    for i, candidate in enumerate(candidates):
        country_b = normalize_country(candidate[3])
        name_tokens_b, addr_tokens_b = tokens(name_b[i]), tokens(addr_b[i])
        numbers_b = numbers(addr_b[i])
        postal_b = {value for value in numbers_b if 5 <= len(value) <= 6}
        geo_b = extract_geography(candidate[2], candidate[3])
        name_jaccard = jaccard(name_tokens_a, name_tokens_b)
        addr_jaccard = jaccard(addr_tokens_a, addr_tokens_b)
        rows.append((
            name_ratio[i], name_token_ratio[i], name_jaro[i], name_jaccard,
            addr_ratio[i], addr_token_ratio[i], addr_jaccard,
            jaccard(numbers_a, numbers_b), float(bool(numbers_a & numbers_b)),
            float(bool(numbers_a and numbers_b and not numbers_a & numbers_b)),
            length_ratio(name_a, name_b[i]), length_ratio(addr_a, addr_b[i]),
            float(bool(name_a) and name_a == name_b[i]),
            float(bool(addr_a) and addr_a == addr_b[i]),
            float(bool(anchor[3]) and country_a == country_b),
            float(candidate[0].startswith("S3-")), float(not name_a or not name_b[i]),
            float(not addr_a or not addr_b[i]), float(bool(postal_a & postal_b)),
            float(bool(geo_a[0]) and geo_a[0] == geo_b[0]),
            float(bool(geo_a[1]) and geo_a[1] == geo_b[1]),
            float(bool(geo_a[2]) and geo_a[2] == geo_b[2]),
            name_jaccard, addr_jaccard,
            float(bool(anchor[3]) and bool(candidate[3]) and country_a != country_b),
            float(bool(geo_a[0]) and bool(geo_b[0]) and geo_a[0] != geo_b[0]),
            float(bool(geo_a[1]) and bool(geo_b[1]) and geo_a[1] != geo_b[1]),
            float(bool(geo_a[2]) and bool(geo_b[2]) and geo_a[2] != geo_b[2]),
        ))
    return np.asarray(rows, dtype=np.float32)


def macro_f05(truth: dict[str, set[str]], predictions: dict[str, set[str]]) -> float:
    scores = []
    for entity_id, actual in truth.items():
        predicted = predictions.get(entity_id, set())
        if not actual and not predicted:
            scores.append(1.0)
        elif not actual or not predicted:
            scores.append(0.0)
        else:
            common = len(actual & predicted)
            precision = common / len(predicted)
            recall = common / len(actual)
            scores.append(1.25 * precision * recall / (0.25 * precision + recall)
                          if common else 0.0)
    return float(np.mean(scores))


def _fit_model(X: np.ndarray, y: np.ndarray, requested_device: str):
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    base_params = dict(n_estimators=160, learning_rate=0.05, num_leaves=15,
                       max_depth=6, min_child_samples=40, random_state=2026,
                       verbosity=-1, n_jobs=4)
    devices = ["gpu", "cpu"] if requested_device == "auto" else [requested_device]
    last_error = None
    for device in devices:
        try:
            model = LGBMClassifier(**base_params, device_type=device)
            model.fit(pd.DataFrame(X, columns=FEATURE_NAMES), y)
            LOG.info("Fitted LightGBM using %s", device)
            return model, device
        except (LightGBMError, RuntimeError, ValueError) as error:
            last_error = error
            if device == "gpu" and requested_device == "auto":
                LOG.warning("GPU LightGBM unavailable; falling back to CPU: %s", error)
                continue
            raise
    raise RuntimeError(f"Could not fit LightGBM: {last_error}")


def train(dataset: Path, output: Path, *, sample_divisor: int = 100,
          per_key: int = 30, final_limit: int = 50, device: str = "auto") -> dict:
    run_started = time.perf_counter()
    memory = PeakMemorySampler()
    memory.start()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "train_candidates.sqlite"
    index_started = time.perf_counter()
    build_index(dataset, "train", index_path)
    index_seconds = time.perf_counter() - index_started
    anchors = load_sampled_anchors(dataset, sample_divisor)
    truth = load_truth(dataset, {row[0] for row in anchors})
    training_x, training_y = [], []
    calibration = []
    validation = []
    recall_counts = Counter()
    conn = open_index(index_path)
    try:
        for number, anchor in enumerate(anchors, 1):
            candidates = generate_candidates(conn, anchor, per_key=per_key, final_limit=final_limit)
            vector = candidate_features(anchor, candidates)
            actual = truth[anchor[0]]
            retrieved = {row[0] for row in candidates}
            fold = zlib.crc32((anchor[0] + "fold").encode()) % 10
            partition = "calibration" if fold == 0 else "validation" if fold == 1 else "train"
            recall_counts[f"{partition}_links"] += len(actual)
            recall_counts[f"{partition}_recalled"] += len(actual & retrieved)
            recall_counts[f"{partition}_s1"] += 1
            recall_counts[f"{partition}_candidates"] += len(candidates)
            if partition == "calibration":
                calibration.append((anchor[0], actual, candidates, vector))
            elif partition == "validation":
                validation.append((anchor[0], actual, candidates, vector))
            elif candidates:
                labels = np.array([int(row[0] in actual) for row in candidates], dtype=np.int8)
                # Include every retrieved positive and the highest-similarity
                # negatives from each source, rather than arbitrary early rows.
                negatives_by_source = []
                for prefix in ("S2-", "S3-"):
                    source_negatives = [i for i, row in enumerate(candidates)
                                        if row[0].startswith(prefix) and labels[i] == 0]
                    source_negatives.sort(key=lambda i: _candidate_rank(anchor, candidates[i]), reverse=True)
                    negatives_by_source.extend(source_negatives[:6])
                negatives = np.asarray(negatives_by_source, dtype=int)
                chosen = np.concatenate([np.flatnonzero(labels == 1), negatives])
                training_x.append(vector[chosen])
                training_y.append(labels[chosen])
            if number % 2_000 == 0:
                LOG.info("Processed %s sampled S1 anchors", f"{number:,}")
    finally:
        conn.close()
    if not training_x:
        raise ValueError("No training candidates; adjust blocking keys")
    X = np.vstack(training_x)
    y = np.concatenate(training_y)
    if len(np.unique(y)) != 2:
        raise ValueError("Training pairs need both classes; adjust blocking/sample size")
    model, selected_device = _fit_model(X, y, device)
    LOG.info("Fitted LightGBM on %s pairs (%s positives)", f"{len(y):,}", f"{int(y.sum()):,}")
    def score_anchors(group):
        scored = []
        for entity_id, actual, candidates, vector in group:
            probabilities = (model.predict_proba(pd.DataFrame(vector, columns=FEATURE_NAMES))[:, 1]
                             if len(candidates) else np.array([]))
            scored.append((entity_id, actual, candidates, probabilities))
        return scored

    scored_calibration = score_anchors(calibration)
    scored_validation = score_anchors(validation)
    calibration_truth = {entity_id: actual for entity_id, actual, _, _ in scored_calibration}
    valid_truth = {entity_id: actual for entity_id, actual, _, _ in scored_validation}
    thresholds = np.arange(0.05, 0.951, 0.05)
    threshold_scores = {}
    for threshold in thresholds:
        predictions = {entity_id: {row[0] for row, probability in zip(candidates, probabilities)
                                   if probability >= threshold}
                       for entity_id, _, candidates, probabilities in scored_calibration}
        threshold_scores[round(float(threshold), 2)] = macro_f05(calibration_truth, predictions)
    best_threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], value))
    best_predictions = {entity_id: {row[0] for row, probability in zip(candidates, probabilities)
                                    if probability >= best_threshold}
                        for entity_id, _, candidates, probabilities in scored_validation}
    true_positive_links = sum(len(valid_truth[entity_id] & predicted)
                              for entity_id, predicted in best_predictions.items())
    predicted_links = sum(map(len, best_predictions.values()))
    singleton_ids = {entity_id for entity_id, actual in valid_truth.items() if not actual}
    source_recall = {}
    for prefix in ("S2-", "S3-"):
        denominator = sum(sum(candidate.startswith(prefix) for candidate in actual)
                          for actual in valid_truth.values())
        numerator = sum(sum(candidate.startswith(prefix) for candidate in
                            actual & {row[0] for row in candidates})
                        for _, actual, candidates, _ in scored_validation)
        source_recall[prefix[:2]] = numerator / denominator if denominator else None
    metrics = {
        "sample_divisor": sample_divisor, "per_key": per_key, "final_limit": final_limit,
        "requested_device": device, "selected_device": selected_device,
        "train_s1": recall_counts["train_s1"],
        "calibration_s1": recall_counts["calibration_s1"],
        "validation_s1": recall_counts["validation_s1"],
        "training_pairs": int(len(y)), "training_positives": int(y.sum()),
        "train_candidate_recall": recall_counts["train_recalled"] / max(1, recall_counts["train_links"]),
        "validation_candidate_recall": recall_counts["validation_recalled"] / max(1, recall_counts["validation_links"]),
        "validation_true_links": recall_counts["validation_links"],
        "validation_average_candidates": recall_counts["validation_candidates"] / max(1, recall_counts["validation_s1"]),
        "calibration_macro_f05": threshold_scores[best_threshold],
        "validation_macro_f05": macro_f05(valid_truth, best_predictions),
        "threshold": best_threshold, "calibration_threshold_scores": threshold_scores,
        "validation_pair_precision": true_positive_links / predicted_links if predicted_links else 0.0,
        "validation_pair_recall": true_positive_links / max(1, recall_counts["validation_links"]),
        "validation_singletons": len(singleton_ids),
        "validation_singleton_accuracy": (sum(not best_predictions[entity_id] for entity_id in singleton_ids)
                                          / len(singleton_ids) if singleton_ids else None),
        "validation_source_candidate_recall": source_recall,
        "index_runtime_seconds": index_seconds,
        "runtime_seconds": time.perf_counter() - run_started,
        "peak_memory_bytes": memory.stop(),
    }
    joblib.dump(model, output / "baseline_model.joblib")
    (output / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOG.info("Validation: candidate recall %.4f, macro F0.5 %.4f at threshold %.2f",
             metrics["validation_candidate_recall"], metrics["validation_macro_f05"], best_threshold)
    return metrics


def _init_prediction_worker(index_path: Path, model_path: Path, threshold: float,
                            per_key: int, final_limit: int) -> None:
    model = joblib.load(model_path)
    model.set_params(n_jobs=1)
    _WORKER.update(conn=open_index(index_path), model=model, threshold=threshold,
                   per_key=per_key, final_limit=final_limit)


def _predict_batch(anchors):
    conn = _WORKER["conn"]
    model = _WORKER["model"]
    batch = []
    matrices = []
    for anchor in anchors:
        candidates = generate_candidates(conn, anchor, per_key=_WORKER["per_key"],
                                         final_limit=_WORKER["final_limit"])
        batch.append((anchor[0], candidates))
        if candidates:
            matrices.append(candidate_features(anchor, candidates))
    scores = (model.predict_proba(pd.DataFrame(np.vstack(matrices), columns=FEATURE_NAMES))[:, 1]
              if matrices else np.array([]))
    offset = 0
    result = []
    for entity_id, candidates in batch:
        count = len(candidates)
        probabilities = scores[offset:offset + count]
        offset += count
        candidate_ids = [row[0] for row in candidates]
        matched_ids = [row[0] for row, probability in zip(candidates, probabilities)
                       if probability >= _WORKER["threshold"]]
        result.append((entity_id, candidate_ids, matched_ids))
    return result


def _anchor_batches(dataset: Path, size: int = 50, skip: int = 0):
    records = iter(iter_records(source_path(dataset, "test", 1)))
    if skip:
        for _ in islice(records, skip):
            pass
    while batch := list(islice(records, size)):
        yield batch


def predict(dataset: Path, output: Path, *, per_key: int = 30, final_limit: int = 120,
            workers: int = 4, submission_dir: Path | None = None) -> dict:
    run_started = time.perf_counter()
    memory = PeakMemorySampler()
    memory.start()
    metrics = json.loads((output / "baseline_metrics.json").read_text())
    if (metrics["per_key"], metrics["final_limit"]) != (per_key, final_limit):
        raise ValueError("Prediction candidate settings must match training settings")
    index_path = output / "test_candidates.sqlite"
    build_index(dataset, "test", index_path)
    threshold = metrics["threshold"]
    output_dir = submission_dir or output / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    if workers < 1:
        raise ValueError("workers must be at least 1")
    from contextlib import nullcontext
    pool_context = (get_context("spawn").Pool(workers, initializer=_init_prediction_worker,
                    initargs=(index_path, output / "baseline_model.joblib", threshold,
                              per_key, final_limit)) if workers > 1 else nullcontext())
    if workers == 1:
        _init_prediction_worker(index_path, output / "baseline_model.joblib", threshold,
                                per_key, final_limit)
    # Resume a prior interrupted prediction safely: result batches are consumed
    # in input order, and both output files are flushed after each batch.
    completed = 0
    candidate_counts = []
    total_matches = 0
    if matching_path.exists() and candidate_path.exists() and matching_path.stat().st_size and candidate_path.stat().st_size:
        with matching_path.open(newline="", encoding="utf-8") as match_file, \
             candidate_path.open(newline="", encoding="utf-8") as candidate_file:
            match_reader = csv.reader(match_file, delimiter="\t")
            candidate_reader = csv.reader(candidate_file, delimiter="\t")
            if next(match_reader, None) != ["source1_entity_id", "matched_entity_ids"] or \
               next(candidate_reader, None) != ["source1_entity_id", "candidate_entity_ids"]:
                raise ValueError("Existing output headers do not match; move the old outputs before predicting")
            for match_row, candidate_row in zip(match_reader, candidate_reader):
                if len(match_row) != 2 or len(candidate_row) != 2 or match_row[0] != candidate_row[0]:
                    raise ValueError("Existing outputs are malformed or out of sync; cannot resume")
                completed += 1
                candidate_counts.append(len([v for v in candidate_row[1].split(",") if v]))
                total_matches += len([v for v in match_row[1].split(",") if v])
            if next(match_reader, None) is not None or next(candidate_reader, None) is not None:
                raise ValueError("Existing output files contain different numbers of rows; cannot resume")
    LOG.info("Resuming after %s existing test S1 predictions", f"{completed:,}")
    file_mode = "a" if completed else "w"
    try:
      with pool_context as pool:
        with matching_path.open(file_mode, newline="", encoding="utf-8") as matching_file, \
             candidate_path.open(file_mode, newline="", encoding="utf-8") as candidate_file:
            matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            if not completed:
                matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
                candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            batches = _anchor_batches(dataset, skip=completed)
            results = pool.imap(_predict_batch, batches, chunksize=1) if pool else map(_predict_batch, batches)
            number = 0
            for result in results:
                for entity_id, candidate_ids, matched_ids in result:
                    candidate_writer.writerow([entity_id, ",".join(candidate_ids)])
                    matching_writer.writerow([entity_id, ",".join(matched_ids)])
                    candidate_counts.append(len(candidate_ids))
                    total_matches += len(matched_ids)
                # Flush each small batch so long runs have visible, recoverable
                # progress and the OS can stream the files to disk.
                matching_file.flush()
                candidate_file.flush()
                number += len(result)
                if (completed + number) % 100_000 == 0:
                    LOG.info("Wrote %s test S1 predictions", f"{completed + number:,}")
    finally:
        if workers == 1:
            _WORKER["conn"].close()
    LOG.info("Wrote %s and %s", matching_path, candidate_path)
    count_conn = open_index(index_path)
    try:
        candidate_corpus = int(count_conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    finally:
        count_conn.close()
    counts = np.asarray(candidate_counts, dtype=np.int64)
    candidate_total = int(counts.sum())
    submission_metrics = {
        "s1_entities": int(len(counts)),
        "candidate_corpus_records": candidate_corpus,
        "total_candidates": candidate_total,
        "average_candidates_per_s1": float(counts.mean()) if len(counts) else 0.0,
        "p95_candidates_per_s1": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "p99_candidates_per_s1": float(np.percentile(counts, 99)) if len(counts) else 0.0,
        "zero_candidate_s1": int((counts == 0).sum()),
        "over_100_candidates_s1": int((counts > 100).sum()),
        "over_1000_candidates_s1": int((counts > 1000).sum()),
        "candidate_reduction_ratio": (1 - candidate_total / (len(counts) * candidate_corpus)
                                      if len(counts) and candidate_corpus else 1.0),
        "predicted_matches": total_matches,
        "runtime_seconds": time.perf_counter() - run_started,
        "peak_memory_bytes": memory.stop(),
    }
    (output_dir / "submission_metrics.json").write_text(
        json.dumps(submission_metrics, indent=2) + "\n", encoding="utf-8")
    LOG.info("Test blocking: avg %.2f, p95 %.0f, p99 %.0f, reduction %.6f",
             submission_metrics["average_candidates_per_s1"],
             submission_metrics["p95_candidates_per_s1"],
             submission_metrics["p99_candidates_per_s1"],
             submission_metrics["candidate_reduction_ratio"])
    return submission_metrics
