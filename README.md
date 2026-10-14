# Smart Student Retention & Early Warning System

A reproducible portfolio prototype for retrospective, Day-35 withdrawal-risk analysis on the Open University Learning Analytics Dataset (OULAD). It builds a human-reviewed support queue; it does not automate student decisions or establish that an intervention improves retention.

## Quick start

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Obtain OULAD from the [official dataset portal](https://analyse.kmi.open.ac.uk/open_dataset), preserve its attribution, and place the seven source CSV files under `data/raw/oulad/` using the filenames documented in [`docs/data_sources.md`](docs/data_sources.md). Raw data is intentionally excluded from Git.

```powershell
uv sync
uv run python -m student_retention.pipeline.run_pipeline
uv run streamlit run app/streamlit_app.py
```

The full pipeline audits and normalizes source files, constructs the Policy C Day-35 cohort, runs EDA, creates the feature matrix, trains/evaluates models, runs calibration and subgroup audits, builds a held-out demonstration queue, and regenerates the four Markdown deliverables. It reads inputs from `data/raw/oulad/` and writes intermediate data under `data/interim/`, cohort/features under `data/processed/`, reports/tables/figures under `outputs/`, model artifacts under `models/`, and public summaries under `deliverables/`. Generated reports, tables, figures, and deliverables are version-controlled; raw/interim/processed data, model binaries, and local prediction queues are excluded.

For individual commands and troubleshooting, see [`docs/runbook.md`](docs/runbook.md). For data definitions, evaluation scope, and responsible-use limitations, see the documentation under [`docs/`](docs/).

## Validation

```powershell
uv run pytest -v
uv run ruff check .
uv lock --check
```

## Interpretation and limits

Model metrics are retrospective estimates from the documented split designs. Risk scores support prioritization for human review; they are not causal effects or guaranteed individual outcomes. Score bands, response windows, and capacity figures are demonstration settings, not institutionally validated service policies. Prospective evaluation is required to assess intervention effectiveness. See [`docs/limitations_and_risks.md`](docs/limitations_and_risks.md) and [`docs/intervention_workflow.md`](docs/intervention_workflow.md).
