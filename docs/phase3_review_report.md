# Independent Phase 3 Correctness Audit Report
## Cohort Construction and Withdrawal Target Verification

**Project:** Smart Student Retention & Early Warning System
**Audit Phase:** Phase 3 — Cohort and Target Construction
**Audit Date:** 2026-10-13
**Status:** **VERIFIED (Solid Junior-Level Engineering Standard Achieved)**
**Auditor:** Independent Automated Correctness Audit

---

## 1. Executive Summary

An independent correctness and reproducibility audit was performed on **Phase 3 — Cohort and Target Construction** of the Smart Student Retention & Early Warning System. The audit verified the cohort-building logic, target derivation, edge-case handling, and artifact persistence against the specifications defined in `docs/PROJECT_PLAN.md`, `docs/target_definition.md`, and the empirical Open University Learning Analytics Dataset (OULAD).

### Key Audit Conclusions:
1. **Canonical Grain & Context Preservation:** The prediction unit is strictly preserved at `(code_module, code_presentation, id_student)`. There is zero row multiplication, zero duplicate canonical keys, and student multi-offering enrollments (3,538 students across 7,346 records) are correctly isolated without cross-presentation leakage.
2. **Policy C Cohort & Boundary Semantics:** The primary clean-data cohort (Policy C at Day-35 cutoff) is 100% mathematically reconciled against raw inputs: 32,593 raw records $\to$ 27,128 eligible records (83.23%) and 5,465 excluded records (16.77%). Boundary conditions (including 6 students unregistering exactly on Day 35) are classified deterministically by the 5-step precedence hierarchy.
3. **Target Derivation & Anti-Leakage:** The target variable `withdrawn_after_cutoff` is binary, integer-typed (`Int64`), and strictly non-leaking. All 4,718 positive cases ($y=1$, 17.3916% base rate) withdrew strictly after Day 35 (`date_unregistration` $\ge 36$). All 22,410 negative cases ($y=0$) represent valid course completers (12,355 Pass, 3,024 Distinction, 7,031 Fail). Zero administrative or outcome columns leak into predictor candidates.
4. **Confirmed Defect Identified & Remediated:** A test side-effect defect in `CohortBuilder.run()` was identified: when a caller specified a custom `output_dir` (e.g., `tmp_path` in synthetic integration tests), the audit report path remained hardcoded to `QUALITY_REPORTS_DIR`, overwriting production quality reports with synthetic 2-record fixtures. This was fixed by routing the audit report to `output_dir` whenever a custom output directory is provided, and a regression test was added.
5. **Test & Pipeline Reproducibility:** All 77 Phase 3 tests and all 171 tests across the entire project repository pass cleanly. Linter checks (`ruff check .`) and Git formatting checks (`git diff --check`) pass with zero errors.

---

## 2. Audit Scope & Verification Methodology

The audit systematically evaluated the Phase 3 implementation across eight criteria:

| Audit Dimension | Target Verification Standard | Evidence & Method |
| :--- | :--- | :--- |
| **1. Prediction Unit & Grain** | Exactly 1 row per `(code_module, code_presentation, id_student)`. | Verified on raw data, intermediate joins, and final parquet/csv artifacts via `scratch/audit_phase3_deep.py`. |
| **2. Multi-Course Enrollments** | Students taking multiple courses are tracked independently per presentation. | Empirical audit identified 3,538 multi-course students; zero cross-course interference found. |
| **3. Registration Timing** | Registration $\le$ Day 35; late registrations ($> 35$) excluded. | Audited boundary cases; verified 15 late registrations excluded at Precedence Step 2. |
| **4. Pre-Cutoff Departures** | Unregistrations $\le$ Day 35 excluded; boundary Day 35 inclusive. | Audited 6 records with `date_unregistration == 35`; all 6 excluded at Step 1 (`EXCL_PRE_CUTOFF_UNREG`). |
| **5. Post-Cutoff Withdrawals** | Withdrawals after Day 35 retained as positive class ($y=1$). | Audited all 4,718 positive records; minimum unregistration day is 36, maximum is 444. |
| **6. Target Categorization** | Completed outcomes (Pass, Distinction, Fail) are negative class ($y=0$). | Reconciled negative breakdown: 12,355 Pass, 7,031 Fail, 3,024 Distinction; 0 unassigned. |
| **7. Contradictory Records** | Contradictory outcome semantics identified and excluded in Policy C. | Record `BBB/2013J/554243` (Fail with unregistration day 166) excluded at Precedence Step 5. |
| **8. Anti-Leakage Invariants** | Zero future or outcome columns leak into feature candidates. | Validated via `assert_no_target_leakage()` and runtime assertion `_assert_cohort_invariants()`. |

---

## 3. Cohort and Target Invariant Matrix

The implementation in `src/student_retention/transformation/build_cohort.py` and `src/student_retention/labels/withdrawal_target.py` enforces the following invariants:

| Invariant Rule | Enforcement Mechanism | Failure Action | Audit Verification |
| :--- | :--- | :--- | :--- |
| **Course Reference Integrity** | `validate_courses_reference_table()` and `validate_relationship()` against `courses` | Raises `CohortInvariantError` | Verified: invalid module ('XXX'), presentation ('2099X'), or length (<200 or >300) in `courses` strictly rejected. |
| **Canonical Key Validity** | `validate_canonical_keys()` | Raises `CohortInvariantError` | Verified: rejects null, blank, float (`123.0`), boolean, or malformed `id_student`. |
| **Key-Set Symmetry** | Symmetric difference check between `student_info` and `student_registration` | Raises `CohortInvariantError` | Verified: zero missing or extra keys between source tables. |
| **One-to-One Joins** | `pd.merge(..., validate="one_to_one")` | Raises `CohortInvariantError` / `MergeError` | Verified: row counts before and after join equal 32,593. |
| **Missing-Marker Integrity** | `parse_and_validate_temporal_date()` | Raises `CohortInvariantError` | Verified: '?' parses to `pd.NA`; forbidden markers (`""`, `"NA"`, `"null"`, `"None"`) raise errors. |
| **Mutually Exclusive Exclusions** | Precedence hierarchy mask evaluation with `unassigned` guard | Deterministic step tagging | Verified: zero double-counting; sum of steps equals total excluded (5,465). |
| **Strict Integer Target** | `validate_withdrawal_target()` | Raises `ValueError` | Verified: target dtype is `Int64`; rejects float `[0.0, 1.0]`, booleans, and non-integers. |
| **Runtime Anti-Leakage** | `_assert_cohort_invariants()` | Raises `CohortInvariantError` | Verified: 0 eligible records with unregistration $\le 35$; 0 early departures receiving $y=1$. |

---

## 4. Empirical Data Findings & Reconciled Counts

All counts were computed and verified directly on the empirical OULAD datasets (`data/raw/oulad/` and `data/interim/normalized/`):

### 4.1. Raw Source Inventory
- **`studentInfo` Records:** 32,593 rows, 12 columns.
- **`studentRegistration` Records:** 32,593 rows, 5 columns.
- **`courses` Records:** 22 course-presentation offerings.
- **Unique Student Identifiers:** 28,785 distinct students.
- **Multi-Course Enrollments:** 3,538 students took multiple course-presentations (accounting for 7,346 records).

### 4.2. Primary Policy C Reconciled Counts (Day-35 Cutoff)
- **Total Source Records:** 32,593 (100.0%)
- **Total Excluded Records:** 5,465 (16.7674%)
- **Eligible Cohort Records:** **27,128 (83.2326%)**
- **Reconciliation Invariant:** $27,128 + 5,465 = 32,593$ (Exact match; `is_reconciled = True`).

#### Target Distribution (`withdrawn_after_cutoff`):
- **Positive Cases ($y=1$):** **4,718 records** (17.3916% base rate)
  - All 4,718 have `final_result == 'Withdrawn'` and `date_unregistration > 35`.
  - Distribution of `date_unregistration`: Min = Day 36, Median = Day 106, Max = Day 444. Null count = 0.
- **Negative Cases ($y=0$):** **22,410 records** (82.6084%)
  - Pass: 12,355 records
  - Fail: 7,031 records
  - Distinction: 3,024 records
  - Distribution of `date_unregistration`: 22,410 nulls, 0 non-nulls.

#### Mutually Exclusive Exclusion Precedence Hierarchy:
| Step | Exclusion Code | Reason / Description | Record Count | % of Raw Data |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `EXCL_PRE_CUTOFF_UNREG` | Unregistered on or before Day 35 cutoff | 5,349 | 16.4115% |
| **2** | `EXCL_LATE_REGISTRATION` | Registered after Day 35 cutoff | 15 | 0.0460% |
| **3** | `EXCL_MISSING_REGISTRATION_DATE` | Unrecorded registration date | 7 | 0.0215% |
| **4** | `EXCL_WITHDRAWN_MISSING_UNREG_DATE` | Recorded as Withdrawn but unregistration date missing | 93 | 0.2853% |
| **5** | `EXCL_CONTRADICTORY_RECORD` | Fail outcome with late unregistration (record `554243`) | 1 | 0.0031% |
| **Total** | — | — | **5,465** | **16.7674%** |

### 4.3. Boundary Value Analysis (Cutoff = Day 35)
- **Registrations on Day 35:** 0 records in raw data.
- **Registrations on Day 36:** 0 records in raw data (first post-cutoff registration is Day 38).
- **Unregistrations on Day 35:** Exactly 6 records in raw data.
  - All 6 records are assigned to Step 1 (`EXCL_PRE_CUTOFF_UNREG`) and excluded. Zero leak into eligible cohort.
- **Unregistrations on Day 36:** Exactly 2 records in raw data.
  - Both records are retained in the eligible cohort and correctly assigned positive target $y=1$.

### 4.4. Sensitivity Benchmark: Policy B Comparison
- **Policy B Eligible Cohort:** 27,136 records (83.2571%)
- **Policy B Positive Cases ($y=1$):** 4,719 records (17.3902% base rate)
- **Policy B Negative Cases ($y=0$):** 22,417 records (82.6098%)
- **Delta (Policy B − Policy C):** Exactly **+8 records** (+1 positive, +7 negative)
  - Policy B retains the 7 records with missing registration dates (Step 3 bypassed; 1 post-cutoff withdrawal + 6 completers) and the 1 contradictory record `BBB/2013J/554243` (Step 5 bypassed; classified as negative `Fail`).
  - Base rate difference is minimal (17.3916% vs 17.3902%, $\Delta = -0.0014\%$).

---

## 5. Confirmed Defects and Remediation

### Defect: Test Pollution of Production Quality Report
- **Description:** In `CohortBuilder.run()`, the output destination for the audit report JSON was hardcoded as:
  ```python
  report_path = QUALITY_REPORTS_DIR / f"{prefix}_audit_report.json"
  ```
  When integration tests ran `CohortBuilder.run(save_outputs=True, output_dir=tmp_path)` using minimal synthetic fixtures, the Parquet and CSV files were saved to `tmp_path`, but the JSON audit report was written directly to the project's production `outputs/quality_reports/cohort_policy_c_audit_report.json`. As a result, running `pytest` overwrote the production report with a 2-record synthetic fixture (`eligible_cohort_records: 2`).
- **Remediation:**
  1. Updated `CohortBuilder.run()` to accept an optional `report_dir: Path | None = None`.
  2. If `output_dir` is passed and differs from `PROCESSED_COHORT_DIR`, `rep_dir` defaults to `output_dir`.
  3. Added an explicit regression assertion in `tests/integration/test_cohort_pipeline.py::test_cohort_builder_run_persists_artifacts` verifying that `Path(saved_paths["audit_report_json"]).parent == output_dir`.
  4. Regenerated the canonical production quality reports via `student-retention build-cohort --policy policy_c` and `--policy policy_b`.
  5. Verified that subsequent `pytest` executions leave `cohort_policy_c_audit_report.json` and `cohort_policy_b_audit_report.json` unaltered.

---

## 6. Verification and Test Evidence

### 6.1. Unit and Integration Tests
Execution of Phase 3 test modules:
```bash
uv run pytest tests/unit/test_cohort.py tests/unit/test_target.py tests/integration/test_cohort_pipeline.py -v
```
- **Outcome:** **77 passed in 13.64s** (100% pass rate).

Execution of full project test suite:
```bash
uv run pytest
```
- **Outcome:** **171 passed, 6 warnings in 26.90s** (100% pass rate across all phases).

### 6.2. Code Quality & Formatting
```bash
uv run ruff check .
git diff --check
```
- **Outcome:** All checks passed with 0 errors and 0 whitespace issues.

### 6.3. CLI Pipeline Execution
```bash
uv run student-retention build-cohort --policy policy_c --cutoff 35
uv run student-retention build-cohort --policy policy_b --cutoff 35
```
- **Outcome:** Exited with code 0; persisted all datasets and validated reports.

---

## 7. Remaining Risks and Limitations

1. **Policy B Selection Sensitivity:** While the base rate difference between Policy C and Policy B is negligible ($-0.0014\%$), prevalence stability does not prove the complete absence of sample selection bias. In Phase 5, modeling sensitivity benchmarks should assess ranking consistency and subgroup calibration across both definitions.
2. **Phase Boundary Scope:** This audit confirmed the correctness of Phase 3 cohort and target artifacts. Downstream Phase 4 EDA and Phase 5 feature engineering depend on these artifacts and must continue to enforce the Day-35 temporal cutoff to avoid data leakage.

---

## 8. Audit Sign-Off

- **Audit Status:** **VERIFIED**
- **Readiness for Phase 5:** **APPROVED** (Phase 3 cohort and target definitions are reproducible, mathematically reconciled, and protected by comprehensive regression tests).
