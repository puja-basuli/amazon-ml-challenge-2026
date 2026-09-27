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

## Recall improvement experiment

- Added a disk-backed inverted postings table for up to four informative name tokens and four address tokens per record. Two added retrieval routes query these postings by country and source, then union/deduplicate with the existing geographic, token, numeric, and approximate routes.
- Optimized benchmark compression to rank each candidate union once when comparing multiple final caps; checked parity against the prior compressor on several caps.
- On a deterministic 1,138-S1 training sample against the full 10,320,219-record S2/S3 pool, `per_key=400` and `final_limit=2400` produced 90.66% raw-union link recall and 85.50% retained candidate recall. Average candidates were 2,386/S1; P95/P99 were 2,400; reduction was 99.9769%. The sweep took 1,216 seconds and peaked near 312 MiB. This sample benchmark crosses the 85% blocking target, but needs a larger/complete validation run before treating it as stable.
- The first `baseline_recall85` training run used 400 postings per key and a 2,400-candidate cap. It retained 85.76% of training-fold links but 82.83% of validation-fold links. Validation macro F1 was 0.326 (macro F0.5 0.281) at the threshold selected on calibration data; this is much worse than the previous lower-cap model and nowhere near 0.999.
- A validation-fold-only cap benchmark (231 S1 anchors, 821 links) with 500 postings per key found raw union recall 89.65%. Retained recall was 84.04% at 2,400 candidates, 84.41% at 3,200, and 85.75% at 4,000. At 4,000, average candidates were 3,896, P95/P99 were 4,000, and candidate reduction was 99.9622%. The run took 323 seconds and peaked at about 312 MiB. This is a sampled-fold result, not full-dataset recall.
- Rescoring the saved 2,400-cap model on these wider candidate sets (thresholds through 0.99) gave validation-tuned macro F1 of 0.463 at cap 2,400 and 0.420 at cap 4,000; this threshold sweep uses the validation fold and is therefore diagnostic, not an unbiased final metric. Expanded threshold rescoring is in progress. The 4,000-cap config is now in `configs/baseline_recall85.json`; its saved model was trained with the lower 2,400 cap and must not be treated as that config's final model.
- The challenge's official metric is macro F0.5. A 0.999 F1/F0.5 has not been demonstrated. Current evidence shows a large gap to that target, so do not claim it has been achieved.

## Next steps

1. Finish expanded threshold rescoring of the saved model on the 231 validation anchors.
2. Improve candidate ranking/matching and retrain with the 500-per-key, 4,000-cap config; compare macro F0.5 and macro F1 on the deterministic calibration/validation folds.
3. Re-measure blocker recall on a larger validation sample before full test inference.
4. Let the existing test prediction finish, then run the official validator and record its result.
