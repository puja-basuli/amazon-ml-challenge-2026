# Repository guidance

## Project context

- Read [docs/problem.md](docs/problem.md) before making challenge-specific choices.
- Check [docs/progress.md](docs/progress.md) before starting substantial work.
- The raw challenge data is under `data/raw/student_resource/dataset/`.
- Source files are tab-separated and use `entity_id`, `business_name`, `business_address`, and `country`.
- Source 1 is the reference source; Source 2 and Source 3 contain possible matches.
- Country labels are open-ended. Do not hard-code the analysis or pipeline to India and US; the test set includes France.
- External business lookup, geocoding, internet enrichment, and external entity-resolution services are prohibited.

## Code and data organization

- Keep reusable data, feature, model, training, and pipeline code under `src/`.
- Keep experiment-specific choices in JSON/config files under `configs/` or `experiments/`.
- Use `scripts/run_experiment.py` as the config-driven experiment entry point. Do not create a new runner script for every parameter variation.
- Set the experiment `device` to `auto` for GPU-when-available with CPU fallback; use `cpu` or `gpu` to force a choice. 
- Use the shared root `requirements.txt`; do not create one dependency file per experiment unless an experiment truly requires an isolated environment.
- Put local datasets under `data/` and generated artifacts under `outputs/`. Do not modify raw datasets.
- Keep notebooks for exploration and reproducible analysis; move reusable logic discovered there into `src/` when appropriate.

## Execution and validation

* Do not run training, blocking, large indexing jobs, full inference, full dataset processing, or other computationally expensive experiment workloads unless the user explicitly asks for execution.
* The default responsibility is to implement the requested change correctly and verify that the code is structurally runnable, not to execute the actual experiment.
* After modifying code, perform lightweight validation whenever practical. This may include:
  * syntax/compilation checks
  * import checks
  * config parsing and validation
  * unit tests or small targeted tests
  * checking interfaces, shapes, schemas, and expected inputs/outputs
  * running on tiny synthetic or explicitly small sample data when useful
* Fix errors discovered during lightweight validation before considering the implementation complete.
* Do not start a real training, blocking, indexing, inference, or large-data experiment merely to verify that the implementation works.
* Leave actual experiment execution to the user by default. At completion, provide the exact command the user should run.
* Do not claim that an experiment, training run, blocking run, or full pipeline succeeded unless it was actually executed and there is evidence of successful completion.
* When the user explicitly requests execution, report the command, configuration, runtime-relevant assumptions, and measured result.
* Submission files must be tab-separated and contain every test Source 1 entity exactly once. Matches may reference only existing Source 2 or Source 3 IDs, with no duplicates.
* Validate generated submissions with `data/raw/student_resource/utils/validate_submission.py`.

## Documentation and reproducibility

- After a meaningful code or configuration change, update `docs/progress.md` with what changed and the next step.
- The user maintains experiment results; do not invent, overwrite, or claim experiment metrics without a recorded run.
- Use fixed random seeds for sampled analyses or experiments when randomness is involved.
- Do not invent challenge requirements or conclusions unsupported by the available data.


## Experiment design

* Prefer config-driven experimentation. Parameters that are expected to vary between experiments should live in config files rather than being hard-coded in scripts or pipeline code.
* Treat `scripts/run_experiment.py` as a generic entry point. It should primarily load configuration, invoke the appropriate pipeline, and persist/report results. Keep reusable implementation logic under `src/`.
* A parameter-only experiment should normally require changing or creating a config file, not modifying Python source code.
* Do not duplicate implementations to test parameter variations. Reuse the same component and vary its configuration.
* When a genuinely new algorithm, feature, preprocessing method, or model is introduced, implement it as a reusable component under `src/` and expose the relevant experimental choices through configuration.
* Keep components sufficiently independent that one stage can be changed without requiring unrelated changes elsewhere in the pipeline.
* When comparing experiments or performing ablations, change only the intended variable whenever practical. Do not silently change preprocessing, data selection, evaluation definitions, or other pipeline behavior at the same time.
* Keep evaluation definitions consistent across comparable experiments. Additional diagnostic metrics may be added when useful, but existing metric semantics should not change silently.
* Every recorded experiment should be reproducible from its saved configuration. Preserve the exact configuration used alongside the resulting metrics or artifacts.
* Do not silently overwrite previous experiment results. Use experiment names, run identifiers, timestamps, or another simple mechanism to preserve distinct runs.
* Prefer the smallest abstraction that supports the current experiments. Do not introduce experiment-management frameworks, registries, factories, plugin systems, or other infrastructure until multiple implementations or a concrete workflow need justifies them.
* Before making a substantial architectural change, inspect the existing data flow and implementation first. Prefer extending the current design over rewriting working components.
* Keep experiment history and conclusions out of this file. Record current project state in `docs/progress.md`, experiment-specific parameters in configs, and measured results in experiment outputs.
