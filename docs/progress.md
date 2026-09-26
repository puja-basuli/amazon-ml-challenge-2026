# Progress

## Current state

- The repository contains reusable ML code, notebooks, scripts, local data, and outputs.
- The challenge statement and raw train/test TSVs are available.
- `notebooks/01_eda.ipynb` contains an unrun EDA workflow.
- A disk-backed multi-pass geographic blocker, LightGBM matcher, and submission workflow is implemented under `src/` and `scripts/`.
- Shared dependencies are defined in the root `requirements.txt` for all experiments.
- Model runs use the config-driven `scripts/run_experiment.py` entry point. Blocking comparisons are explored in `notebooks/02_blocking_experiments.ipynb`.
- Reusable blocking code supports chunked character TF-IDF retrieval, a K sweep, persisted Parquet candidates, and missed-link reports. Its starting settings are in `configs/blocking_char_tfidf.json`.
- The per-K comparison output now reports only K, candidate recall, full-hit rate, average candidates, and missed true links; detailed calculations remain available through an opt-in formatter.
- Notebook blocking runs save timestamped artifacts under `outputs/experiments/<experiment_name>/`, including the exact config and compact metrics CSV.
- `docs/decisions.md` is the lightweight log template for future evidence-based experiment decisions.

## Decisions

- Keep challenge requirements in `docs/problem.md`.
- Keep reusable logic under `src/` and experiment-specific settings under `configs/`.
- Keep local datasets and generated outputs out of Git.
- Measure candidate recall on the complete training S2/S3 pool and tune the match threshold on held-out S1 entities using macro F0.5.

## Multi-pass pipeline run

- `scripts/profile_data.py` profiled the complete train/test TSVs. Train sources contain 2,206,821 S1, 5,034,616 S2, and 5,285,603 S3 rows. Test sources contain 1,732,544 S1, 4,887,273 S2, and 5,082,316 S3 rows. Ground truth has 7,638,365 links.
- `src/data/index.py` builds resumable SQLite indexes in bounded batches for country/state/PIN/city, name/address tokens, numeric components, and approximate character grams.
- `src/pipelines/baseline.py` unions nine routes, deduplicates and caps candidates, mines hard negatives, trains LightGBM, calibrates the threshold on held-out S1 entities, and streams test outputs. Interrupted predictions resume from the flushed row prefix.
- A sampled blocking sweep over 1,138 S1 anchors found 60.21% raw link recall and 58.69% retained recall at `per_key=80`, `final_limit=400`; average candidates were 383.31/S1 with P95/P99 of 400. Candidate reduction was 99.9963% against the 10,320,219-record S2/S3 training pool. The sweep took 833 seconds and peaked at about 298 MiB.
- Training used 4,517 sampled S1 entities, creating 50,294 candidate pairs (7,118 positives). Held-out validation candidate recall was 54.97%, macro F0.5 was 0.6552, and the selected threshold was 0.85. Training took 1,874 seconds and peaked at about 475 MiB.
- Test indexing completed for 9,969,589 S2/S3 records. Test prediction is running into `output/candidate_pairs.tsv` and `output/matching_results.tsv`; it resumes automatically from existing complete rows.

## Next steps

1. Let test prediction finish, then run the official validator.
2. Record final test candidate volume, P95/P99, reduction, runtime, and peak memory in this file.
3. Review missed-link examples from the benchmark and consider adding multiple rare tokens per entity if higher recall is needed.
