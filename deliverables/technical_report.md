# Technical Report: End-to-End Retention Modeling & Decision Support Architecture

**Version:** 1.0 (Phase 8 Production Review)<br>
**Dataset:** Open University Learning Analytics Dataset (OULAD)<br>
**Author:** AI Engineering & Learning Analytics Team<br>

---

## 1. System Overview & Problem Formulation

The early-warning retention system predicts the probability that an enrolled student will withdraw from a course module strictly after relative **Day 35** of the academic presentation:

$$\hat{y} = P(\text{withdrawn\_after\_cutoff} = 1 \mid \mathbf{x}_{\le 35})$$

- **Observation Unit & Grain:** `(code_module, code_presentation, id_student)` across 27,128 eligible student registrations.
- **Temporal Cutoff:** Day 35 (end of week 5, inclusive).
- **Prohibited Leakage:** Zero post-cutoff VLE interactions, zero post-cutoff coursework submissions, zero assessment grades, and zero demographic features in the primary model.

---

## 2. Ingestion, Cleaning & Exclusion Precedence

Starting from raw enrollment records, the pipeline enforces a strictly deterministic 5-step exclusion precedence hierarchy:

```
[Raw Source Records: 32,593]
         |
         +--> Step 1: Student unregistered on or before prediction cutoff (Day 35). Already left course. ------> Excluded: 5,349 records
         |
         +--> Step 2: Student registered after prediction cutoff (Day 35). Not yet enrolled at cutoff. ------> Excluded: 15 records
         |
         +--> Step 3: Registration date is unrecorded. Enrollment presence on Day 35 cannot be confirmed. ------> Excluded: 7 records
         |
         +--> Step 4: Recorded as Withdrawn but withdrawal timing is unrecorded. Cannot verify post-cutoff status. ------> Excluded: 93 records
         |
         +--> Step 5: Final result recorded as Fail but date_unregistration is Day 166. Contradictory outcome semantics. ------> Excluded: 1 records
         |
         v
[Eligible Cohort Policy C: 27,128 records (83.23%)]
  - Positive Cases (Withdrawn > Day 35): 4,718 records (17.3916%)
  - Negative Cases (Pass / Distinction / Fail): 22,410 records (82.6084%)
```

---

## 3. Feature Engineering & Strict Temporal Safeguards

The primary feature store comprises **22 baseline features** across four functional categories:

1. **Course Context (4 features):** `code_module` (7 categories), `presentation_season` (B vs J), `studied_credits`, `num_of_prev_attempts`.
2. **Registration Dynamics (2 features):** `registration_lead_time_days` (clamped to $[-365, 35]$), `is_late_registration` (binary flag for registration after Day 0).
3. **Digital VLE Engagement (11 features):**
   - Cumulative volumes: `vle_total_clicks_window`, `vle_active_days_window`, `vle_unique_sites_accessed`.
   - Temporal windows: `vle_pre_course_clicks` ($[-25, -1]$), `vle_recent_clicks_14d` ($[22, 35]$).
   - Resource diversity: `vle_clicks_forumng`, `vle_clicks_oucontent`, `vle_clicks_quiz`.
   - Behavioral disengagement: `has_no_vle_activity_by_cutoff`, `days_since_last_vle_interaction` (imputed with deterministic sentinel value **65.0** for inactive students).
   - Pacing momentum: `vle_velocity_ratio` = $\frac{\text{recent clicks} / 14}{\text{total clicks} / 61}$.
4. **Assessment Progress Milestones (5 features):**
   - Restricted strictly to scheduled coursework due $\le$ Day 35 ($A_{\text{due}}$): `assessments_due_count`, `submissions_count_lte_cutoff`, `submitted_on_time_count`, `late_submissions_count_lte_cutoff`, `missing_due_submissions_count`.
   - Explicit exclusion of assignment scores and feedback content to prevent target leakage.

---

## 4. Modeling, Evaluation & Probability Calibration

### Evaluation Splitting Protocol
- Group-aware 80/20 train/test split strictly on `id_student` via `GroupShuffleSplit` (random seed 42).
- Zero student overlap between partitions: Train $N=21,714$ (5,414 test records).

### Benchmark Comparison (Held-Out Test Cohort)

| Model | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | Decision Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dummy (Class Prior) | 0.1681 | 1.0000 | 0.2878 | 0.1681 | 0.5000 | 0.13988 | 0.01 |
| Logistic Regression (Unweighted) | 0.3152 | 0.2923 | 0.3033 | 0.2966 | 0.6703 | 0.13221 | 0.26 |
| Logistic Regression (Balanced) | 0.3156 | 0.3055 | 0.3104 | 0.2934 | 0.6695 | 0.22433 | 0.62 |
| Random Forest (Unweighted) | 0.3242 | 0.3253 | 0.3247 | 0.3123 | 0.6813 | 0.13118 | 0.25 |
| Random Forest (Balanced) | 0.3176 | 0.3110 | 0.3143 | 0.3077 | 0.6762 | 0.21816 | 0.59 |

### Probability Calibration Diagnostics

| Model Configuration | Calibration Method | Brier Score | ECE | MCE | Min Prob | Max Prob | Mean Prob |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest (Raw Probabilities) | none | 0.13118 | 0.01239 | 0.38937 | 0.0280 | 0.5624 | 0.1741 |
| Random Forest (Isotonic Calibrated) | isotonic_group_kfold_cv | 0.13107 | 0.00649 | 0.18942 | 0.0016 | 0.9425 | 0.1732 |

- Evaluated candidate calibration procedures strictly on the training partition via 5-fold `GroupKFold` out-of-fold cross-validation.
- Isotonic calibration was selected based on training-only selection metadata.
- On untouched test data, Isotonic calibration improved empirical probability alignment, achieving a **47.6% error reduction** in Expected Calibration Error ($0.01239 \to 0.00649$).

---

## 5. Subgroup Fairness & Disparity Analysis

Conducted dual subgroup audits across protected demographic characteristics (evaluating classifier sensitivity at a fixed diagnostic threshold and held-out demonstration queue inclusion):

```
+-----------------------------------------------------------------------------------------+
|                                SUBGROUP PARITY FINDINGS                                 |
+-----------------------------------------------------------------------------------------+
|  1. Classifier Diagnostic (Fixed Threshold theta = 0.25):                               |
|     - Disparate Impact Warnings: 12 stable subgroups outside [0.80, 1.25]               |
|     - Elevated FPR Warnings: 7 stable subgroups exceeding FPR ratio 1.50                |
|                                                                                         |
|  2. Operational Support Queue Policy (Top-500 Calibrated Allocation):                   |
|     - Disparate Impact Warnings: 5 stable subgroups                                     |
|       * gender F: DIR = 0.6560 (Selection Rate 28.82%)                                  |
|       * age_band 35-55: DIR = 0.7686 (Selection Rate 30.62%)                            |
|       * region London Region: DIR = 1.2587 (Selection Rate 40.68%)                      |
|       * region North Western Region: DIR = 1.2763 (Selection Rate 41.25%)               |
|       * region Scotland: DIR = 1.2939 (Selection Rate 41.82%)                           |
|     - Elevated FPR Warnings: 0 stable subgroups exceeding FPR ratio 1.50                |
+-----------------------------------------------------------------------------------------+
```

---

## 6. Decision Support Caseload & Operational Grain Audit

The demonstration support queue deliverable (`outputs/predictions/student_support_queue.csv`) was generated strictly on the held-out test cohort ($N=5,414$) with capacity capped at $K=500$ enrollment slots per presentation:

- `2013B`: 500 slots, 500 unique students, 0 duplicate slots.
- `2013J`: 500 slots, 500 unique students, 0 duplicate slots.
- `2014B`: 500 slots, 483 unique students, 17 duplicate slots.
- `2014J`: 500 slots, 477 unique students, 23 duplicate slots.
- **Total:** 2,000 slots, 1,827 unique students, 173 duplicate slots across terms.

**Technical vs Governance Boundary:** The modeling grain is the course enrollment. Transforming enrollment slots into an unduplicated 500-unique-student operational caseload requires institutional policy approval regarding multi-course risk aggregation.
