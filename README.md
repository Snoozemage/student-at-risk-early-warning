# Smart Student Retention & Early Warning System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Environment: uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://docs.astral.sh/uv/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test framework: pytest](https://img.shields.io/badge/test%20framework-pytest-blue.svg)](https://docs.pytest.org/)

A reproducible learning-analytics prototype for identifying **course-withdrawal risk** from course-context, registration, VLE-engagement, and assessment signals in the Open University Learning Analytics Dataset (OULAD). It combines a temporal data pipeline, retrospective model evaluation, subgroup audits, and a Streamlit dashboard for human-reviewed support prioritization.

> **Decision support, not automated action.** The project predicts withdrawal from a specific module-presentation; it does not predict permanent university dropout, establish causation, or show that an intervention improves retention.

---

## 1. Project Overview

Student-support teams may have limited time to review every course enrollment. This prototype explores how historical registration, assessment, and virtual learning environment (VLE) data can be turned into an auditable early-warning workflow:

1. Validate source data and document its quality.
2. Define which student-course enrollments are eligible at the prediction cutoff.
3. Build features using information available by that cutoff.
4. Train and retrospectively evaluate risk models.
5. Rank cases for a **human** to review, with evidence and limitations visible.

The prediction unit is one student in one module-presentation: `(code_module, code_presentation, id_student)`. The primary snapshot is at the end of **relative Day 35**. Policy C eligibility rules exclude late registrations, pre-cutoff withdrawals, and records with missing or contradictory timing. Time-varying VLE and submission features use information available through Day 35; outcome and withdrawal fields define eligibility or labels and are not predictors.

### Intended users

- Academic advisors and student-support teams reviewing a prioritized caseload.
- Course tutors and module leads comparing engagement patterns.
- Analysts and administrators reviewing evaluation results, fairness audits, and data limitations.

### What this prototype does not do

- Make or automate academic decisions, penalties, or student-facing risk disclosures.
- Infer that a student will leave the university based on withdrawal from one module.
- Establish that a particular behavior causes withdrawal.
- Validate intervention effectiveness, an optimal outreach time, or institutional service-level targets.

---

## 2. Evidence Snapshot

The values below are from the documented Phase 9 run on the available OULAD source files. They are retrospective results for this dataset and evaluation design, not expected performance guarantees for another institution or cohort.

| Measure | Observed result | Interpretation |
|---|---:|---|
| Eligible Policy C cohort | 27,128 enrollments | After applying documented Day-35 eligibility and data-quality rules. |
| Post-cutoff withdrawals | 4,718 | Positive labels; withdrawal is from the specific module-presentation. |
| Eligible non-withdrawal outcomes | 22,410 | Negative labels (`Pass`, `Fail`, or `Distinction`). |
| Primary input features (before encoding) | 22 | Features available under the documented Day-35 feature rules. |
| Random Forest cross-validation PR-AUC | 0.3312 mean | Mean average precision across five GroupKFold validation folds on the training partition. |
| Random Forest held-out random-split PR-AUC | 0.3123 | Uncalibrated Random Forest ranking result on a 5,414-enrollment, 20% student-disjoint random holdout. |
| Demonstration queue | 2,000 enrollment rows; 1,827 unique students | Includes 173 enrollment rows beyond the unique-student count, reflecting concurrent enrollments. It is not a validated unique-student caseload capacity. |

The provisional held-out PR-AUC benchmark of 0.40 was not met. The held-out score is a retrospective ranking metric, not a deployment guarantee. The demonstration queue uses a separately calibrated model; its thresholds and capacity have not been prospectively validated. The project also reports chronological per-term evaluation, calibration diagnostics, error analysis, and subgroup audits. Consult the generated reports for their exact evaluation scope; pooled random-split results and chronological results answer different questions.

---

## 3. Core Capabilities

- **Source audit and data contracts** — inventory OULAD tables, validate schemas and identifiers, check relationships and temporal rules, and report source warnings.
- **Explicit cohort and target construction** — use Policy C for the primary cohort, with a documented Policy B sensitivity analysis; keep enrollment eligibility, target labels, and predictors distinct.
- **Temporal feature engineering** — construct engagement and assessment features from information available through Day 35; test that future events cannot change the feature snapshot.
- **Modeling and evaluation** — compare baselines and candidate classifiers using student-aware splits, cross-validation, held-out metrics, ranking analysis, and error audits.
- **Calibration and subgroup review** — record the selected calibration method and inspect available subgroup metrics with sample-size and stability context.
- **Human-reviewed decision support** — produce a demonstration queue and explanatory summaries. Queue thresholds, response windows, and capacity are proposed settings, not empirically validated operating policies.
- **Interactive dashboard and generated reports** — inspect cohort summaries, engagement patterns, a filtered support queue, model performance, and fairness limitations.

---

## 4. Architecture and Repository Map

```text
.
├── app/                         # Streamlit dashboard and reusable UI components
├── config/                      # Source, cohort, feature, model, and evaluation settings
├── data/                        # Local data; raw/interim/processed files are Git-ignored
├── deliverables/                # Executive, technical, recommendation, and presentation reports
├── docs/                        # Data, target, feature, project, and runbook documentation
├── models/                      # Local model artifacts; binaries are Git-ignored
├── notebooks/                   # Exploratory and analytical notebooks
├── outputs/                     # Reproducible reports, tables, figures, and local queue data
├── scripts/                     # Stage-specific pipeline entry points
├── src/student_retention/       # Ingestion, validation, EDA, labels, features, modeling, fairness, reporting
├── tests/                       # Unit and integration tests
├── pyproject.toml               # Python package and tool configuration
└── uv.lock                      # Locked dependency resolution
```

The canonical orchestrator is `src/student_retention/pipeline/run_pipeline.py`. The dashboard reads generated artifacts; it does not duplicate the training or cohort-building logic. Reproducible reports, tables, figures, and Markdown deliverables are tracked. Raw/interim/processed datasets, trained model binaries, local prediction queues, logs, and private planning files are excluded from Git.

---

## 5. Data Source and Setup

### Dataset

This project uses the [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset), covering seven modules across four presentations. It contains student-course outcomes, registrations, assessments, and daily VLE interactions. The `studentVle.csv` source has more than 10 million rows, so the complete pipeline can take several minutes and requires adequate disk space for intermediate files.

Download OULAD from its official source and place the seven CSV files here:

```text
data/raw/oulad/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

Raw data is intentionally excluded from this repository. Preserve the dataset's [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/) and see [`docs/data_sources.md`](docs/data_sources.md) for provenance, schema notes, and known source peculiarities.

### Requirements

- Python **3.12** (see `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) for environment and lockfile management.
- The seven OULAD CSV files above for a populated pipeline and dashboard.

From the repository root in PowerShell:

```powershell
uv sync
```

---

## 6. Run the Project

### Rebuild the full pipeline

```powershell
uv run python -m student_retention.pipeline.run_pipeline
```

The orchestrator runs nine stages: source inventory; cleaning and quality validation; cohort and target construction; exploratory analysis; feature generation; model training; model evaluation figures; calibration, fairness audits, and the demonstration queue; and generated deliverables. Outputs are written under `data/`, `models/`, `outputs/`, and `deliverables/` as described in [`docs/runbook.md`](docs/runbook.md).

To inspect or run selected stages, use:

```powershell
uv run python -m student_retention.pipeline.run_pipeline --list-stages
uv run python -m student_retention.pipeline.run_pipeline --stages generate_deliverables
```

Run required upstream stages before selecting a downstream stage; the orchestrator does not infer omitted prerequisites.

### Launch the dashboard

After generating the artifacts:

```powershell
uv run streamlit run app/streamlit_app.py
```

The app provides five pages:

1. **Operational Overview** — cohort and priority-tier summaries.
2. **Engagement Patterns** — VLE, registration, and assessment patterns.
3. **Student Support Queue** — filter and inspect course-enrollment cases; status updates are simulated in-session.
4. **Model Performance** — held-out curves, calibration, and ranking diagnostics.
5. **Fairness and Limitations** — subgroup measures, warnings, and responsible-use context.

After environment setup, the dashboard can handle unavailable or missing generated artifacts. Populated views require the local OULAD data and pipeline artifacts.

---

## 7. Validation

Run the project's tests and static checks with the locked environment:

```powershell
uv run pytest -v
uv run ruff check .
uv lock --check
git diff --check
```

Integration tests use synthetic fixtures for isolated behavior and exercise real-data checks when the local OULAD inputs are available. A passing test suite alone does not mean the full data pipeline or a live browser session was run; report those results separately.

---

## 8. Methodological Notes and Limitations

1. **Course withdrawal is not institutional dropout.** The outcome belongs to one module-presentation.
2. **Day 35 is a relative course day.** The primary feature snapshot includes eligible information through that day; it is not a calendar date.
3. **Retrospective prediction is not intervention evaluation.** Ranking or discrimination metrics do not prove that outreach changes outcomes.
4. **Observational associations are not causal explanations.** The project does not claim that a feature causes withdrawal.
5. **Thresholds and service targets remain proposals.** Demonstration score bands, response windows, and a per-presentation queue cap have not been prospectively validated.
6. **Queue rows represent enrollments.** Concurrent module enrollments mean a count of enrollment rows does not directly establish unique-student staff capacity.
7. **Source warnings remain visible.** A quality report can pass its configured checks while disclosing documented anomalies or missing values.
8. **Data-dependent reproduction requires OULAD.** The source dataset and derived local data are not bundled in Git.

See [`docs/target_definition.md`](docs/target_definition.md), [`docs/feature_catalog.md`](docs/feature_catalog.md), [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/business_requirements.md`](docs/business_requirements.md), and [`docs/runbook.md`](docs/runbook.md) for detail. Generated stakeholder-facing results are in [`deliverables/`](deliverables/).

---

## 9. Data Citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. **Scientific Data**, 4, 170171. [https://doi.org/10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171).

The dataset is provided under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Follow the dataset documentation and preserve attribution when reusing or redistributing derived work.
