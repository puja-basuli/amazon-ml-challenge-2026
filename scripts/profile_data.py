#!/usr/bin/env python3
"""Chunked, reproducible profile of the challenge TSV files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.text import normalize


class HyperLogLog:
    """Small-memory cardinality estimate from pandas' stable 64-bit hashes."""

    def __init__(self, precision: int = 12):
        self.precision = precision
        self.size = 1 << precision
        self.registers = np.zeros(self.size, dtype=np.uint8)

    def update(self, values: pd.Series) -> None:
        hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
        mask = np.uint64(self.size - 1)
        for hashed in hashes:
            value = int(hashed)
            bucket = value & int(mask)
            remainder = value >> self.precision
            rank = 64 - self.precision + 1 if remainder == 0 else 64 - self.precision - remainder.bit_length() + 1
            if rank > int(self.registers[bucket]):
                self.registers[bucket] = rank

    def estimate(self) -> int:
        m = self.size
        alpha = 0.7213 / (1 + 1.079 / m)
        estimate = alpha * m * m / np.power(2.0, -self.registers.astype(np.float64)).sum()
        zeros = int((self.registers == 0).sum())
        if zeros and estimate <= 2.5 * m:
            estimate = m * np.log(m / zeros)
        return round(float(estimate))


def source_profile(path: Path, *, chunk_size: int, sample_mod: int) -> dict:
    result = {
        "path": str(path), "rows": 0,
        "missing": Counter(), "countries": Counter(),
        "unique_ids_hll": HyperLogLog(), "unique_names_hll": HyperLogLog(),
        "unique_addresses_hll": HyperLogLog(), "postal_like": Counter(),
        "address_comma_segments": Counter(), "sample_names": Counter(), "sample_addresses": Counter(),
    }
    columns = ["entity_id", "business_name", "business_address", "country"]
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunk_size, dtype="string", keep_default_na=False):
        if list(chunk.columns) != columns:
            raise ValueError(f"Unexpected schema in {path}: {list(chunk.columns)}")
        result["rows"] += len(chunk)
        for column in columns:
            result["missing"][column] += int(chunk[column].str.strip().eq("").sum())
        result["countries"].update(chunk["country"].str.strip().replace("", "<missing>").value_counts().to_dict())
        result["unique_ids_hll"].update(chunk["entity_id"])
        normalized_names = chunk["business_name"].map(lambda v: normalize(v, name=True))
        normalized_addresses = chunk["business_address"].map(normalize)
        result["unique_names_hll"].update(normalized_names)
        result["unique_addresses_hll"].update(normalized_addresses)
        sample = (pd.util.hash_pandas_object(chunk["entity_id"], index=False).to_numpy(dtype="uint64") % sample_mod) == 0
        if sample.any():
            result["sample_names"].update(normalized_names[sample].tolist())
            result["sample_addresses"].update(normalized_addresses[sample].tolist())
        address = chunk["business_address"].astype(str)
        result["postal_like"]["5_digit"] += int(address.str.contains(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)", regex=True).sum())
        result["postal_like"]["6_digit"] += int(address.str.contains(r"(?<!\d)[1-9]\d{5}(?!\d)", regex=True).sum())
        result["postal_like"]["alphanumeric_postal_shape"] += int(address.str.contains(r"(?i)\b[A-Z0-9][A-Z0-9 -]{2,9}\b", regex=True).sum())
        result["address_comma_segments"]["2_or_more"] += int(address.str.count(",").ge(1).sum())
        result["address_comma_segments"]["3_or_more"] += int(address.str.count(",").ge(2).sum())
        if result["rows"] % 1_000_000 < len(chunk):
            print(f"{path.name}: scanned {result['rows']:,} rows", flush=True)

    names = result.pop("sample_names")
    addresses = result.pop("sample_addresses")
    for field in ("unique_ids_hll", "unique_names_hll", "unique_addresses_hll"):
        result[field] = result[field].estimate()
    result["sample_name_frequency_top20"] = names.most_common(20)
    result["sample_address_frequency_top20"] = addresses.most_common(20)
    result["sample_name_duplicate_rate"] = (sum(n for n in names.values() if n > 1) / max(1, sum(names.values())))
    result["sample_address_duplicate_rate"] = (sum(n for n in addresses.values() if n > 1) / max(1, sum(addresses.values())))
    return result


def truth_profile(path: Path, *, chunk_size: int) -> dict:
    result = Counter()
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunk_size, dtype="string", keep_default_na=False):
        for raw in chunk["matched_entity_ids"]:
            links = [part.strip() for part in str(raw).split(",") if part.strip()]
            result["s1_rows"] += 1
            result[f"matches_{min(len(links), 4)}{'+' if len(links) >= 4 else ''}"] += 1
            result["true_links"] += len(links)
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/student_resource/dataset"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--sample-mod", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/data_profile.json"))
    args = parser.parse_args()
    profile = {"sources": {}, "ground_truth": {}}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = args.dataset / split / f"{split}_source{source}.tsv"
            profile["sources"][f"{split}_source{source}"] = source_profile(path, chunk_size=args.chunk_size, sample_mod=args.sample_mod)
    profile["ground_truth"] = truth_profile(args.dataset / "train" / "train_ground_truth.tsv", chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(profile["ground_truth"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
