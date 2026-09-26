# ML challenge

This repository is organized so experiments can share data preparation, feature engineering, model, and training code as the project grows.

Start with [the problem statement](docs/problem.md). [Progress](docs/progress.md) records the current state, and [AGENTS.md](AGENTS.md) gives coding agents the repository's working conventions.

```text
configs/         Shared configuration
docs/            Problem statement, progress, and experiment record
src/
  data/          Data loading, validation, and splitting
  features/      Reusable feature transformations
  models/        Model implementations
  training/      Training and evaluation logic
  pipelines/     Reusable workflows that connect the components
notebooks/       Exploration, EDA, and blocking experiments
scripts/         Config-driven model experiment entry point
data/            Local datasets (ignored by Git)
outputs/         Generated models, metrics, and predictions (ignored by Git)
```

Add reusable logic under `src/` and keep experiment-specific choices under `configs/`.

## Running experiments

All experiments use the shared environment in `requirements.txt`. Create it once:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_experiment.py --config configs/baseline.json
```

Model runs use `scripts/run_experiment.py`; the JSON config selects the model pipeline and its settings. Blocking exploration happens in notebooks and uses reusable implementations from `src/`.

Set `"device": "auto"` to try GPU LightGBM and fall back to CPU when GPU support is unavailable. Use `"cpu"` to skip the GPU attempt or `"gpu"` to require GPU support. Indexing, blocking, and feature generation remain CPU-based.

The model experiment runner supports `train`, `predict`, and `full` modes. The default config trains and predicts; generated indexes, model, and metrics go under the configured `outputs/` directory, and the submission TSVs go under `output/`.

## Full multi-pass run

The baseline uses Country → State → PIN as a primary block and unions geography/name, PIN/name, city/name, rare-name-token, address-token, numeric-component, and approximate name/address routes. It uses SQLite indexes rather than all-pairs comparisons.

```sh
python scripts/run_experiment.py --config configs/baseline.json --mode full
```

The checked-in config samples training S1 records at 1:500, uses up to 80 postings per route and retains at most 400 candidates per S1, and writes:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

If prediction is interrupted, rerun with `--mode predict`; it resumes after the complete rows already written. Validate from the repository root with:

```sh
python data/raw/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/raw/student_resource/dataset/test
```

Run `python scripts/benchmark_multipass.py --help` to change the sampled blocking sweep, and `python scripts/profile_data.py --help` to profile the input TSVs.

## Blocking experiments

Compare blocking approaches in [the blocking experiments notebook](notebooks/02_blocking_experiments.ipynb). It uses reusable code under `src/` and defaults to the settings in `configs/blocking_char_tfidf.json`. The blocker searches the complete training S2/S3 corpus in chunks. Its sample divisor controls how many Source 1 training records are evaluated; setting it to 1 evaluates them all and is substantially more expensive.

Results are saved under `outputs/experiments/<experiment_name>/<run_id>/`: `candidates.parquet` contains ranked candidate pairs for the largest K, `comparison.tsv` and `metrics.json` contain the K sweep, and `missed_pairs.tsv` lists missed true links at each K. The directory also contains the config snapshot and a manifest with data, code, and library signatures. Each run gets a timestamped directory so previous artifacts remain available.
