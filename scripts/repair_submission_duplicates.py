#!/usr/bin/env python3
"""Trim aligned submission TSVs at their first repeated S1 ID.

The prediction pipeline resumes by row count, so retaining a later suffix after
a duplicate could skip missing S1 IDs. Trimming both files at the first repeat
leaves a valid contiguous prefix that can be safely regenerated.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def repair(matching: Path, candidate: Path) -> tuple[int, str | None]:
    seen: set[bytes] = set()
    with matching.open("r+b") as match_file, candidate.open("r+b") as candidate_file:
        match_header = match_file.readline()
        candidate_header = candidate_file.readline()
        if match_header.rstrip(b"\r\n") != b"source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected matching TSV header")
        if candidate_header.rstrip(b"\r\n") != b"source1_entity_id\tcandidate_entity_ids":
            raise ValueError("Unexpected candidate TSV header")

        unique_rows = 0
        duplicate_id = None
        while True:
            match_offset = match_file.tell()
            candidate_offset = candidate_file.tell()
            match_row = match_file.readline()
            candidate_row = candidate_file.readline()
            if not match_row and not candidate_row:
                break
            if not match_row or not candidate_row:
                raise ValueError("Submission files have different row counts")
            match_id = match_row.split(b"\t", 1)[0]
            candidate_id = candidate_row.split(b"\t", 1)[0]
            if match_id != candidate_id:
                raise ValueError(
                    f"Files diverge at data row {unique_rows + 1}: "
                    f"{match_id!r} != {candidate_id!r}"
                )
            if match_id in seen:
                duplicate_id = match_id.decode("utf-8", errors="replace")
                match_file.truncate(match_offset)
                candidate_file.truncate(candidate_offset)
                break
            seen.add(match_id)
            unique_rows += 1
    return unique_rows, duplicate_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    rows, duplicate = repair(args.matching, args.candidate)
    if duplicate:
        print(f"Trimmed both files to {rows} aligned unique data rows; first duplicate was {duplicate}")
    else:
        print(f"No duplicate found; verified {rows} aligned unique data rows")


if __name__ == "__main__":
    main()
