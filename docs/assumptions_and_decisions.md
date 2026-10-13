# Technical Assumptions & Architecture Decisions Log (ADR)

**Project:** Smart Student Retention & Early Warning System
**Phase:** Phase 1 — Business Specification and Data Contract
**Date:** 2026-10-13
**Status:** Proposed Specification (Pending Review)

---

## ADR-001: Selection of Primary Cohort Policy C (Strict Clean-Data)

### Context & Problem Statement
To evaluate early withdrawal risk at Day 35, the modeling cohort must represent learners who were actively enrolled and eligible for intervention at the prediction cutoff. In the raw OULAD dataset, several edge cases present ambiguous or contradictory registration and unregistration data.

Two candidate policies were evaluated in Phase 0:
* **Policy C (Strict Clean-Data):** Excludes records with missing registration dates, unverified withdrawal timing, or contradictory outcome records.
* **Policy B (Permissive Sensitivity):** Retains records with missing registration dates if no early withdrawal is recorded, and retains flagged contradictory outcome records.

### Decision
**Adopt Policy C as the primary cohort policy** for model training, evaluation, and operational decision support. Retain **Policy B as a sensitivity analysis benchmark**.

### Rationale
1. **Defensibility over Sample Size:** An educational early-warning model must be built upon verifiable administrative facts. Including records where registration timing cannot be confirmed introduces unquantifiable noise into feature calculation (e.g., registration lead time).
2. **Deterministic Precedence:** Policy C applies an unambiguous 5-step precedence hierarchy that accounts for all 32,593 raw source records without overlap or double-counting.
3. **Purity of Target Labels:** Requiring that withdrawn students have a verified `date_unregistration > 35` ensures zero label ambiguity.

### Consequences & Impact
* Total excluded source records: **5,465** (16.77% of raw data).
* Primary eligible cohort: **27,128 records**.
* Eventual withdrawals (positive class): **4,718 records**.
* Non-withdrawing completers (negative class): **22,410 records**.
* Baseline positive prevalence: **17.3916%**.

---

## ADR-002: Role of Policy B in Sensitivity Analysis

### Context
Stakeholders may ask whether excluding 8 edge-case records (7 with missing registration dates and 1 contradictory Fail record) artificially alters model metrics or biases retention estimates.

### Decision
Maintain Policy B in the data contract and evaluation protocol as a formal sensitivity analysis check.

### Rationale
* Policy B yields **27,136 eligible records** (4,719 positive cases; 17.3902% base rate).
* The delta between Policy B and Policy C is exactly **8 records** (+1 positive, +7 negative).
* The base rate difference is negligible (0.0014 percentage points).
* Sensitivity analysis in Phase 4/5 will confirm that model discrimination (PR-AUC, ROC-AUC) and ranking behavior are entirely robust to the choice between Policy C and Policy B.

---

## ADR-003: Resolution of Observed Source-Data Edge Cases & Record `BBB / 2013J / 554243`

### 1. The 93 Withdrawn Records with Missing Unregistration Date
* **Finding:** 93 records in `studentInfo` have `final_result = Withdrawn`, but `date_unregistration = '?'` in `studentRegistration`.
* **Decision:** Exclude from both Policy C and Policy B (`EXCL_WITHDRAWN_MISSING_UNREG_DATE`).
* **Rationale:** Without a recorded date of unregistration, it is impossible to determine whether the withdrawal occurred on or before Day 35 (which would make the student ineligible) or after Day 35 (which would make them a positive case). Imputing an arbitrary date would create arbitrary ground truth.

### 2. The 9 Fail Records with Recorded Unregistration Dates & Verification of Record `('BBB', '2013J', '554243')`
* **Finding:** 9 records have `final_result = Fail` yet record an unregistration date.
  * 8 records have `date_unregistration <= 0` (withdrew prior to course start).
  * 1 record (`'BBB', '2013J', '554243'`) has `date_unregistration = 166`.
* **Empirical Row Verification:**
  - `studentInfo.csv`: `{'code_module': 'BBB', 'code_presentation': '2013J', 'id_student': 554243, 'gender': 'F', 'region': 'South West Region', 'highest_education': 'Lower Than A Level', 'imd_band': '60-70%', 'age_band': '0-35', 'num_of_prev_attempts': 0, 'studied_credits': 60, 'disability': 'N', 'final_result': 'Fail'}`
  - `studentRegistration.csv`: `{'code_module': 'BBB', 'code_presentation': '2013J', 'id_student': 554243, 'date_registration': -35.0, 'date_unregistration': 166.0}`
* **Official Source Semantics:**
  The official OULAD data documentation explicitly defines:
  *"date_unregistration – date of the student unregistration to the module presentation measured as the number of days from the start of the module-presentation. Students who completed the course have this field empty. Students who unregistered have Final Result as Withdrawn."*
* **Decision:**
  - The 8 pre-course cases are excluded by Precedence Step 1 (`EXCL_PRE_CUTOFF_UNREG`, `date_unregistration <= 35`) in both policies.
  - Record `('BBB', '2013J', '554243')` is **excluded in Policy C** under Precedence Step 5 (`EXCL_CONTRADICTORY_RECORD`) because having both `Fail` and an unregistration date violates the official source semantics. In **Policy B**, it is retained as a negative case to test sensitivity to source contradictions.
  - Neither source field is silently modified.

### 3. The 45 Missing Registration Dates
* **Finding:** 45 records have `date_registration = '?'`.
  * 38 records have `date_unregistration <= 35` (departed before cutoff).
  * 1 record has `date_unregistration = 146` (`Withdrawn`).
  * 6 records have `date_unregistration = '?'` (5 Fail, 1 Pass).
* **Decision:**
  * The 38 pre-cutoff departures are excluded by Precedence Step 1 in both policies.
  * The remaining 7 records are excluded in Policy C (`EXCL_MISSING_REGISTRATION_DATE`) and retained in Policy B.
* **Rationale:** For the 7 records, whether the student was actually enrolled by Day 35 cannot be verified from the data.

---

## ADR-004: Exclusion of Assessment Scores from Day-35 Baseline

### Context
`studentAssessment.score` records marks awarded for submitted assignments. However, early assignments (e.g., TMA 1 due around Day 28–32) require human tutor marking.

### Decision
1. Strictly exclude assessment score attributes from the primary Day-35 feature set. Permit only submission occurrence, submission count, and submission timeliness features.
2. Restrict all five assessment progress features strictly to the common due-assessment universe ($A_{\text{due}}$: continuous non-exam coursework with verified scheduled deadline $\text{date} \le 35$). Genuine submissions for assessments due after Day 35 submitted early (observed in 2,350 learners) are deliberately excluded from due-progress counts to prevent mathematical anomalies (such as on-time submissions exceeding due assessments, which would occur in 2,186 records under an unconstrained definition) and to ensure that $\text{submissions\_count\_lte\_cutoff} = \text{submitted\_on\_time\_count} + \text{late\_submissions\_count\_lte\_cutoff} \le \text{assessments\_due\_count}$. Early learner preparation is captured through VLE interaction signals.
3. Treat banked assessment records (`is_banked = 1`) as administrative credit transfers from prior presentations (timestamped with sentinel `date_submitted = -1`). Banked records are excluded from genuine current-presentation submission counts, but are credited toward required coursework so that students with banked credit on due assessments are not falsely penalized as missing due submissions.

### Rationale
* The OULAD schema records `date_submitted` but records no grade publication timestamp.
* Assuming that an assignment submitted on Day 30 has been marked and returned to the student and advisor by Day 35 is an unverified operational assumption that risks retrospective leakage.
* Anchoring all assessment features to the same due-assessment universe ($A_{\text{due}}$) ensures total mathematical consistency across counts, timeliness, and missingness signals without confusing advisors.
* Submission events are logged instantaneously by learning management systems and are 100% verified to be available at Day 35.
* Banked records represent prior academic attainment, not current learning effort; treating them as new current-term submissions would distort behavioral engagement tracking.

---

## ADR-005: Student Group-Aware Evaluation vs. Chronological Temporal Validation

### Context
In OULAD, 3,538 students enroll in multiple module-presentations. When evaluating predictive models, two distinct evaluation objectives exist: measuring generalization to unseen learners versus measuring forward temporal generalization across academic years.

### Decision
Define and maintain two separate validation split protocols:
1. **Primary Unseen-Student Evaluation (`GroupShuffleSplit` on `id_student`):**
   - Mandates strictly **zero student overlap** between training and test sets.
   - Verifies the model's ability to generalize to new learners.
2. **Chronological Temporal Validation (Train on 2013, Test on 2014):**
   - Evaluates forward-chaining across academic presentation cycles.
   - **Natural Operational Design:** Acknowledges that 1,309 students from 2013 enroll in 2014 courses in the Policy C cohort. Expected student overlap is exactly 1,309 students (9.16% of 2014 cohort).
   - **Disjoint Purged Design:** In a supplementary test, the 1,309 returning students are purged from the test set to evaluate joint temporal and unseen-student generalization.

---

## ADR-006: Subgroup Fairness Audit & Non-Punitive Deployment Policy

### Context
OULAD contains demographic attributes (gender, age band, disability, deprivation decile, region, highest education).

### Decision
1. Deploy fairness evaluation in an **audit-only** mode.
2. Exclude demographic attributes from the primary 22-feature allowlist to prevent direct disparate treatment.
3. Explicitly acknowledge that **demographic exclusion does not by itself eliminate proxy effects or guarantee fairness**, as behavioral engagement variables can correlate with demographic background. Continuous fairness auditing across protected attributes remains essential and mandatory.
4. Allow an optional demographic-feature sensitivity experiment to measure performance deltas and monitor bias shifts.
5. Mandate that model outputs are used solely for **supportive, proactive advisor outreach**, never for punitive actions, automated filtering, or resource withholding.
6. Require minimum subgroup sample size of $N \ge 50$ to avoid misleading small-sample statistics.

---

## ADR-007: Canonical Workspace Data Directory Path

### Context
Earlier Phase 0 audit reports contained references to `data/raw/oalad/` (typographical discrepancy) alongside the intended path `data/raw/oulad/`.

### Decision
Standardize strictly on `data/raw/oulad/` as the canonical raw data path across all code, configuration files, and documentation.

---

## ADR-008: Sign Conventions & Inactivity Sentinel Values

### Context
Engineering features from relative dates requires clear sign conventions and defensible missingness representation.

### Decision
1. **Registration Timing Features & Policy B Missing Handling:**
   - $\text{registration\_lead\_time\_days} = -\text{date\_registration}$. Under this convention, advance registrations before Day 0 are positive (e.g., $-(-30) = +30$ days), registrations on Day 0 are 0, and eligible late registrations after Day 0 are negative (e.g., $-(+10) = -10$ days).
   - $\text{is\_late\_registration} = 1 \text{ if } 0 < \text{date\_registration} \le 35, 0 \text{ if } \text{date\_registration} \le 0, \text{null if } \text{date\_registration is null}$.
   - Under primary Policy C, all 27,128 records have confirmed registration dates (zero nulls). Under Policy B sensitivity analysis, the 7 records with unrecorded registration dates evaluate to null rather than being falsely classified as confirmed non-late registrants ($0$), ensuring consistent numerical median/indicator imputation across registration features.
2. **VLE Inactivity Sentinel Value:** The maximum possible observed date gap between the earliest interaction (Day -25) and the cutoff (Day 35) is 60 days. For students with zero interactions on or before Day 35, the recency date is unobserved. During feature extraction (prior to pipeline imputation), `days_since_last_vle_interaction` is filled with the upper-bound sentinel value **65** (strictly greater than the 60-day maximum observed span), paired with the explicit binary indicator `has_no_vle_activity_by_cutoff = 1`. Consequently, `days_since_last_vle_interaction` contains no missing values at pipeline input and is not modified by the general numerical median imputer. The general imputer (`SimpleImputer(strategy="median", add_indicator=True)`) operates exclusively on other numerical features with genuine missingness (such as `registration_lead_time_days`).

---

## ADR-009: Stable Presentation-Season Feature for Out-of-Time Generalization

### Context
In the chronological temporal evaluation design, models are trained on academic year 2013 (`2013B`, `2013J`) and evaluated on academic year 2014 (`2014B`, `2014J`). If the raw course presentation string `code_presentation` is passed to `OneHotEncoder(handle_unknown='ignore')`, the categories `2014B` and `2014J` are unseen during fitting on 2013 data. Consequently, all one-hot encoded columns for presentation evaluate to zero vectors on the 2014 test set, collapsing presentation distinction entirely.

### Decision
Replace `code_presentation` in the primary 22-feature baseline allowlist with `presentation_season`:
- Extracted as the terminating seasonal character: `'B'` (February start / Spring intake) or `'J'` (October start / Autumn intake).
- The parent key `(code_module, code_presentation, id_student)` is strictly retained on all data tables for record identification, grain integrity, and entity relationships, while `presentation_season` serves as the generalized predictive predictor.

### Rationale
1. **Out-of-Time Generalization:** Both `'B'` and `'J'` seasons are observed in 2013 training data and recur in 2014 testing data, ensuring valid category alignment without zero-vector collapse.
2. **Pedagogical Meaning:** Seasonal start dates reflect distinct academic pacing, institutional calendars, and intake demographics that generalize stably across calendar years.
