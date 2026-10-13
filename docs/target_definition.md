# Target Definition & Cohort Specification

**Project:** Smart Student Retention & Early Warning System
**Phase:** Phase 1 — Business Specification and Data Contract
**Date:** 2026-10-13
**Status:** Proposed Specification (Pending Review)

---

## 1. Prediction Unit & Observation Grain

The fundamental prediction unit of the system is:
$$\text{Observation Grain} = (\text{code\_module}, \text{code\_presentation}, \text{id\_student})$$

* **Entity Semantics:** An observation represents an individual learner enrolled in a specific course module presentation.
* **Non-Uniqueness of Student ID:** `id_student` is **not** unique across the dataset. In OULAD, 3,538 students enroll in multiple module-presentations. An observation identifier must always be composite to avoid cross-presentation collisions.
* **Prediction Task:** Binary classification estimating the probability of eventual withdrawal from the specific module presentation after the prediction cutoff.

---

## 2. Prediction Cutoff: Relative Day 35

* **Definition:** The end of relative Day 35 after the official course start date (inclusive of all activity through Day 35).
* **Calendar Context:** Relative Day `0` denotes the official start date of the presentation. Day 35 corresponds approximately to the end of Study Week 5.
* **Relative Duration Placement:** Day 35 represents $\approx 14.5\%$ of a February (`B`) presentation (234–241 days) and $\approx 13.0\%$ of an October (`J`) presentation (261–269 days).
* **Configurability:** Specified in `config/project.yaml` and `config/feature_windows.yaml`.

---

## 3. Primary Cohort Policy: Policy C (Strict Clean-Data)

### 3.1. Core Eligibility Principles
Policy C prioritizes methodological rigor and data cleanliness over dataset size. An enrollment record is eligible if and only if:
1. The student registered on or before Day 35 (`date_registration <= 35` and not null).
2. The student did not unregister on or before Day 35 (`date_unregistration > 35` or unregistration date is null).
3. The student's withdrawal timing is verifiable (records with `final_result = Withdrawn` must have a valid non-null `date_unregistration`).
4. The student's outcome record does not contain contradictory administrative data (specifically excluding `('BBB', '2013J', '554243')` which records `final_result = Fail` yet has `date_unregistration = 166`).

### 3.2. Deterministic Exclusion Precedence Order
Because multiple exclusion conditions can overlap for a single record (e.g., 38 records with missing registration date also unregistered on or before Day 35), exclusions must be applied in a deterministic, mutually exclusive sequence.

The precedence hierarchy is defined as follows:

| Precedence Step | Exclusion Code | Rule & Condition | Reason & Rationale | Records Excluded |
|:---:|---|---|---|:---:|
| **1** | `EXCL_PRE_CUTOFF_UNREG` | `date_unregistration <= 35` (regardless of `final_result`) | Learner departed on or before the cutoff; early warning is moot. | **5,349** |
| **2** | `EXCL_LATE_REGISTRATION` | `date_registration > 35` | Learner enrolled after the Day 35 prediction cutoff. | **15** |
| **3** | `EXCL_MISSING_REGISTRATION_DATE` | `date_registration is null` (and passed Steps 1–2) | Registration timing cannot be verified at cutoff. | **7** |
| **4** | `EXCL_WITHDRAWN_MISSING_UNREG_DATE` | `final_result == 'Withdrawn'` and `date_unregistration is null` | Timing of withdrawal is completely unrecorded. | **93** |
| **5** | `EXCL_CONTRADICTORY_RECORD` | `('BBB', '2013J', '554243')` (`final_result == 'Fail'` & `date_unregistration == 166`) | Contradictory outcome record violating official source semantics. | **1** |
| **Total** | **All Exclusions** | **Mutually Exclusive Sum** | **Total Excluded Source Records** | **5,465** |

### 3.3. Reconciliation of Primary Policy C Cohort

| Cohort Metric | Source Count | % of Raw Data | Breakdown / Detail |
|---|:---:|:---:|---|
| **Total Raw Source Records** | **32,593** | 100.00% | `studentInfo` $\cap$ `studentRegistration` |
| **Total Excluded Records** | **5,465** | 16.77% | Accounted for by mutually exclusive precedence rules |
| **Eligible Primary Cohort (Policy C)** | **27,128** | **83.23%** | **Evaluation & Modeling Cohort** |
| **Positive Class (`Withdrawn`)** | **4,718** | **17.3916%** | Withdrew strictly after Day 35 (`date_unregistration > 35`) |
| **Negative Class (`Completers`)** | **22,410** | **82.6084%** | `Pass`: 12,355; `Distinction`: 3,024; `Fail`: 7,031 |

$$\text{Positive Class Prevalence (Base Rate)} = \frac{4,718}{27,128} \approx 17.3916\%$$

---

## 4. Alternative Cohort Policy: Policy B (Sensitivity Analysis)

Policy B is formalized for sensitivity analysis to evaluate whether excluding edge cases biases model performance or changes operational conclusions.

### 4.1. Policy B Rules
- Exclude records with pre-cutoff unregistration (`date_unregistration <= 35`): 5,349 records.
- Exclude late registrations (`date_registration > 35`): 15 records.
- Exclude Withdrawn records with missing unregistration date: 93 records.
- **Retain** the 7 records with missing registration date that have no record of pre-cutoff unregistration.
- **Retain** the contradictory record `('BBB', '2013J', '554243')` as a negative case (`final_result = Fail`).

### 4.2. Policy B Reconciliation & Comparison with Policy C

| Metric | Primary Policy C | Sensitivity Policy B | Delta (B − C) | Explanation of Delta |
|---|:---:|:---:|:---:|---|
| **Eligible Cohort Size** | **27,128** | **27,136** | **+8** | 7 missing reg date + 1 contradictory record |
| **Positive Cases (`Withdrawn`)** | **4,718** | **4,719** | **+1** | 1 record with missing reg date who withdrew Day 146 |
| **Negative Cases (`Completers`)** | **22,410** | **22,417** | **+7** | 5 Fail, 1 Pass (missing reg date) + 1 Fail (`BBB_2013J_554243`) |
| **Positive Base Rate** | **17.3916%** | **17.3902%** | **-0.0014%** | Difference is negligible ($< 0.002\%$) |

---

## 5. Detailed Source-Data Edge Cases Resolution

### 5.1. The 93 Withdrawn Records with Missing Unregistration Date
* **Source Observation:** 93 records in `studentInfo` have `final_result = Withdrawn`, yet in `studentRegistration` their `date_unregistration` is `'?'` (missing).
* **Analysis:** Without an unregistration date, it is impossible to determine whether the student withdrew prior to Day 35 (making them ineligible) or after Day 35 (making them a positive case).
* **Policy Decision:** **Excluded** under both Policy C and Policy B (`EXCL_WITHDRAWN_MISSING_UNREG_DATE`). Imputing an arbitrary unregistration date would introduce label noise and compromise evaluation validity.

### 5.2. The 9 Fail Records with Recorded Unregistration Dates
* **Source Observation:** 9 records have `final_result = Fail`, but also have a non-null `date_unregistration`.
  * 8 records have `date_unregistration <= 0` (unregistered before course start).
  * 1 record (`'BBB', '2013J', '554243'`) has `date_unregistration = 166`.
* **Resolution for the 8 Pre-Course Cases:** Successfully excluded under Precedence Step 1 (`EXCL_PRE_CUTOFF_UNREG`) in both Policy C and Policy B. Because `date_unregistration <= 35`, they departed before the cutoff.
* **Resolution for Record `('BBB', '2013J', '554243')`:**
  * **Empirical Verification:** The exact raw row in `studentInfo.csv` is `{'code_module': 'BBB', 'code_presentation': '2013J', 'id_student': 554243, 'final_result': 'Fail', ...}` and in `studentRegistration.csv` is `{'date_registration': -35.0, 'date_unregistration': 166.0}`. Neither field is altered.
  * **Official Documentation Semantics:** Official OULAD documentation specifies: *"date_unregistration – date of the student unregistration to the module presentation measured as the number of days from the start of the module-presentation. Students who completed the course have this field empty. Students who unregistered have Final Result as Withdrawn."*
  * **Violation of Official Semantics:** Record `554243` simultaneously asserts that the student completed the course without passing (`Fail`) and unregistered on Day 166 (`Withdrawn`). This contradictory state violates official source semantics.
  * **Policy C Decision:** Formally excluded under Precedence Step 5 (`EXCL_CONTRADICTORY_RECORD`) to preserve training data integrity.
  * **Policy B Decision:** Retained as a negative case to evaluate model sensitivity to unresolved administrative anomalies.

### 5.3. The 45 Missing Registration Dates
* **Source Observation:** 45 records have `date_registration = '?'`.
  * 38 records have `date_unregistration <= 35` (departed before cutoff).
  * 1 record has `date_unregistration = 146` (`final_result = Withdrawn`).
  * 6 records have `date_unregistration = '?'` (5 Fail, 1 Pass).
* **Resolution:**
  * The 38 pre-cutoff unregistrations are excluded by Precedence Step 1 in both policies.
  * The remaining 7 records:
    * In Policy C: Excluded by Precedence Step 3 (`EXCL_MISSING_REGISTRATION_DATE`) because course entry timing cannot be confirmed.
    * In Policy B: Retained because they were not unregistered before Day 35.

---

## 6. Target Construction & Evaluation Splitting Designs

### 6.1. Target Variable Definition
The binary target variable `withdrawn_after_cutoff` is defined on eligible records as:
$$y_i = \begin{cases} 1 & \text{if } \text{final\_result} = \text{'Withdrawn'} \land \text{date\_unregistration} > 35 \\ 0 & \text{if } \text{final\_result} \in \{\text{'Pass'}, \text{'Distinction'}, \text{'Fail'}\} \end{cases}$$

### 6.2. Two Distinct Evaluation Split Protocols
The evaluation protocol recognizes two distinct objectives requiring different split designs:
1. **Primary Unseen-Student Group Split (`GroupShuffleSplit` on `id_student`):**
   - **Objective:** Evaluate generalization to learners the model has never encountered before.
   - **Student Overlap Invariant:** Enforces strictly **0 student overlap** between training and test sets ($\text{Train} \cap \text{Test} = \emptyset$).
2. **Chronological Temporal Split (Train on 2013, Test on 2014):**
   - **Objective:** Evaluate forward-chaining across academic presentation years.
   - **Natural Operational Design:** Reflects real-world university operations where 1,309 learners from 2013 re-enroll in 2014 courses. Expected student overlap is **1,309 students** (9.16% of 2014 cohort).
   - **Disjoint Purged Design:** If evaluated for joint temporal and unseen-student generalization, the 1,309 returning students are purged from the 2014 test set to enforce 0 student overlap.

### 6.3. Strict Outcome Leakage Prevention Rules
Under no circumstances may target-related information enter the feature pipeline:
1. **`final_result` Forbidden:** `final_result` is an outcome label and must NEVER be used as a predictor.
2. **`date_unregistration` Forbidden:** `date_unregistration` encodes the exact occurrence and timing of withdrawal. It is strictly used for cohort filtering and target labeling; it must NEVER be included as a predictor.
3. **Future Events Forbidden:** No VLE interaction, assessment submission, or administrative event occurring after Day 35 (`date > 35` or `date_submitted > 35`) may enter feature extraction.
4. **Independent Target Construction:** The target generation module (`labels/withdrawal_target.py`) must execute independently from the feature extraction module (`transformation/build_feature_table.py`).
