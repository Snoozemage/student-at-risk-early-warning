# Developer Runbook

## Environment

- Python 3.12 (see `.python-version`).
- `uv` manages the environment and lock file.
- Raw OULAD CSVs belong in `data/raw/oulad/`; expected filenames and source details are in [`data_sources.md`](data_sources.md). Do not commit raw data.

From the repository root:

```powershell
uv sync
```

## Rebuild the complete pipeline

```powershell
uv run python -m student_retention.pipeline.run_pipeline
```

This runs nine stages in order: source inventory; clean/normalize and quality validation; Policy C Day-35 cohort and target; EDA; temporal feature matrix; training/CV/test evaluation; evaluation figures; calibration, subgroup audits, and the held-out demonstration queue; then the four generated deliverables. The command can take several minutes because `studentVle.csv` contains over 10 million rows. It overwrites reproducible local artifacts in the data, models, outputs, and deliverables directories.

The stage runner accepts an optional `--stages` list when invoking its module, for example:

```powershell
uv run python -m student_retention.pipeline.run_pipeline --list-stages
uv run python -m student_retention.pipeline.run_pipeline --stages generate_deliverables
```

Run upstream stages before a selected downstream stage; the runner does not infer or build omitted prerequisites.

## Dashboard

Start the dashboard after generating its required artifacts:

```powershell
uv run streamlit run app/streamlit_app.py
```

Streamlit prints the local URL. The dashboard reads the generated queue, model metrics, and fairness/report tables from `outputs/`. Missing files in a clean checkout are handled by the application as unavailable data; generate the pipeline first for populated views.

## Tests and static checks

```powershell
uv run pytest -v
uv run ruff check .
uv lock --check
git diff --check
```

The integration suite includes checks against real OULAD artifacts when present and synthetic fixtures for isolated behaviors. Report skipped tests and warnings with their reason; a fixture test is not evidence of a real-data run.

## Output map

| Path | Contents |
|---|---|
| `outputs/quality_reports/` | Source inventory, source quality audit, cohort and feature validation JSON |
| `data/processed/cohort/` | Cohort datasets and target labels |
| `data/processed/features/` | Day-35 feature matrices |
| `outputs/tables/` | EDA, model evaluation, calibration, subgroup, and queue audit tables |
| `outputs/figures/` | EDA and model diagnostic charts |
| `outputs/predictions/` | Held-out demonstration support queue |
| `outputs/reports/` | EDA, model interpretation, decision-support, and fairness reports |
| `deliverables/` | Generated executive summary, technical report, recommendations, and presentation outline |
| `models/baseline/` | Locally trained model artifacts (ignored by Git) |

Generated reports, tables, figures, and Markdown deliverables are reproducible and intentionally version-controlled. Raw/interim/processed data, model binaries, logs, and local prediction queues are excluded by `.gitignore`.

## Troubleshooting and interpretation

- If a source file is missing, verify its exact filename under `data/raw/oulad/` before running the pipeline.
- If the dashboard reports unavailable artifacts, run the full pipeline and check the paths above.
- The full pipeline starts with source scanning and can use substantial memory and disk space for `studentVle.csv` and Parquet intermediates.
- The registered K=500 queue is a course-presentation enrollment demonstration. Unique-student capacity remains unresolved until a cross-course aggregation policy is defined.
- Retrospective prediction/ranking metrics do not establish intervention effectiveness, optimal intervention timing, calibrated operational cutoffs, or service-level performance. Human review and prospective evaluation are required before any operational use.
- The data-quality report may contain documented source anomalies or warnings even when its validation status is `PASSED`; inspect `all_warnings` in `outputs/quality_reports/data_quality_report.json`.
