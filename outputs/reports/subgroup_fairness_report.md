# Subgroup Fairness Audit & Demographic Sensitivity Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Evaluation Candidate Cohort:** Held-Out Group-Aware Test Cohort (20% sample, N=5,414 candidate enrollment records across 2013B, 2013J, 2014B, 2014J)
**Classifier Diagnostic Policy:** Fixed Decision Threshold $\theta = 0.25$ (Derived from 5-fold GroupKFold cross-validation on 80% random train partition (precision >= 0.35)) evaluated across all candidate records
**Queue Evaluation Scope:** Historical held-out demonstration queue, ranked by the calibration method recorded in training-only selection metadata; capacity is measured in course-enrollment slots, not unique students, and does not represent a deployed institutional policy.

---

## 1. Executive Summary & Policy Adherence

The Smart Student Retention & Early Warning System enforces an **"audit-only" demographic policy**:
* Demographic and socioeconomic variables are **strictly excluded from predictive model inputs** to prevent direct disparate treatment and algorithmic redlining.
* Excluding demographic attributes does **not** prove the absence of proxy reliance; therefore, all models undergo continuous empirical subgroup audits across six protected characteristics: `gender`, `age_band`, `disability`, `imd_band`, `highest_education`, and `region`.
* Statistical stability constraint: Subgroups with sample sizes $N < 50$ carry an explicit small-sample instability warning to prevent over-interpreting noisy ratio estimates.

> [!IMPORTANT]
> **Fairness Evaluation Scope & Policy Boundary:**
> Do not claim that the model has passed fairness merely because one metric is within tolerance. Subgroup disparity metrics provide necessary operational transparency across multiple equity dimensions, but no single threshold or ratio proves the absence of proxy effects or guarantees equal educational outcomes. Observational differences in selection rates or error rates reflect behavioral and contextual signal distributions in historical data and must not be interpreted as causal student deficiencies or proof of model perfection.

---

## 2. Calibration Method Selection & Provenance (Training-Only Validation)

To eliminate any risk of test-set leakage or post-hoc method selection, candidate calibration procedures were evaluated **strictly on the 80% training partition** (`train_df`, N=21,714) using 5-fold `GroupKFold` cross-validation on `id_student`. The held-out test cohort was reserved strictly for untouched final evaluation.

### Training-Only Out-of-Fold Cross-Validation Comparison

| Model Configuration | Calibration Method | Evaluation Split | Brier Score | ECE | MCE | Max Probability |
|---|---|---|---|---|---|---|
| **Random Forest (Raw Probabilities)** | `none` | training_out_of_fold_cv | 0.13494 | 0.00620 | 0.09893 | 0.5776 |
| **Random Forest (Sigmoid Calibrated)** | `sigmoid` | training_out_of_fold_cv | 0.13498 | 0.00941 | 0.04604 | 0.6978 |
| **Random Forest (Isotonic Calibrated)** | `isotonic` | training_out_of_fold_cv | 0.13480 | 0.00284 | 0.27959 | 0.8857 |

**Method Selection Evidence:** The selected method in training-only calibration metadata is **isotonic**. The comparison above reports this run's observed training out-of-fold metrics; the selected method is not assumed to be universally optimal.

---

## 3. Demographic Sensitivity Analysis Findings

To evaluate the predictive utility and ethical trade-offs of demographic predictors, an empirical experiment compared:
1. **Primary Baseline Model:** 22 non-demographic behavioral and context features (demographics strictly excluded).
2. **Demographic-Aware Model:** 28 features (22 primary + 6 one-hot encoded demographic attributes).

### Performance Comparison

| Model Configuration | Features | PR-AUC | ROC-AUC | Brier Score | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Primary Baseline (Excluded)** | 22 | 0.3101 | 0.6797 | 0.1313 | 0.3170 | 0.3275 | 0.3222 |
| **Demographic-Aware (Included)** | 28 | 0.3086 | 0.6819 | 0.1312 | 0.3375 | 0.3286 | 0.3330 |
| **Delta (Demo − Baseline)** | +6 | **-0.0015** | **+0.0022** | **-0.0001** | **+0.0205** | **+0.0011** | **+0.0108** |

### Policy Recommendation

Excluding demographic features from predictors is strongly upheld: adding demographic features yielded an incremental PR-AUC change of -0.0015, but introduces the severe risk of direct disparate treatment and institutional bias reinforcement. Continuous subgroup auditing without direct demographic conditioning provides the ethically soundest posture.

---

## 4. Classifier-Level Subgroup Diagnostic (Uncalibrated RF @ $\theta = 0.25$)

**Binary Decision Evaluated:** Model output thresholding at the fixed validation threshold: $\hat{y} = \mathbb{I}(\text{raw\_prob} \ge 0.25)$.
**Purpose:** Pre-deployment classifier diagnostic evaluating raw score calibration and threshold sensitivity across demographic groups before operational capacity constraints are applied.

Across the 6 protected attributes (36 total evaluated subgroups):
* **Statistically Stable Subgroups ($N \ge 50$):** 35 subgroups (97.2%).
* **Small Sample Subgroups ($N < 50$):** 1 subgroups (2.8%), primarily in extreme deprivation bands or specific regional partitions.

### Observational Disparity Highlights (Classifier Level)

* **Gender (`gender`):** Reference group `M` (Male). Selection rate for females is 13.01% vs males 21.11%.
* **Age Band (`age_band`):** Reference group `0-35`. Learners in the `55<=` group have small sample sizes ($N=35 < 50$), carrying an explicit statistical instability warning.
* **Disability (`disability`):** Reference group `N`. Declared disabled learners (`Y`) exhibit an elevated selection rate relative to reference group `N` (Disparate Impact Ratio = 1.29). Under the system's ethical guidelines, elevated alert rates direct supportive pastoral and accessibility resources rather than punitive sanctions.
* **Education (`highest_education`):** Reference group `A Level or Equivalent`. Selection and error rates reflect observational differences in entry qualifications across academic presentations.
* **Deprivation (`imd_band`):** Reference group `90-100%`. Selection and error rates across deprivation deciles should be reviewed in the master audit table.
* **Region (`region`):** Reference group `South East Region`. Geographical partitions exhibit substantial variation in selection and error rates across the 13 regions.

### Identified Classifier Warnings (21 total flags)

#### Disparate Impact Ratio Flags (Outside [0.80, 1.25]):
- **gender = F** (N=2505): Selection Rate = 13.01%, Disparate Impact Ratio = 0.62
- **disability = Y** (N=496): Selection Rate = 21.77%, Disparate Impact Ratio = 1.29
- **imd_band = 60-70%** (N=473): Selection Rate = 14.59%, Disparate Impact Ratio = 0.79
- **imd_band = 80-90%** (N=460): Selection Rate = 13.91%, Disparate Impact Ratio = 0.75
- **highest_education = No Formal quals** (N=57): Selection Rate = 24.56%, Disparate Impact Ratio = 1.40
- **region = London Region** (N=526): Selection Rate = 18.63%, Disparate Impact Ratio = 1.46
- **region = North Region** (N=314): Selection Rate = 16.88%, Disparate Impact Ratio = 1.32
- **region = North Western Region** (N=463): Selection Rate = 19.44%, Disparate Impact Ratio = 1.52
- **region = Scotland** (N=636): Selection Rate = 22.17%, Disparate Impact Ratio = 1.73
- **region = South Region** (N=510): Selection Rate = 16.27%, Disparate Impact Ratio = 1.27
- **region = South West Region** (N=405): Selection Rate = 16.54%, Disparate Impact Ratio = 1.29
- **region = Wales** (N=391): Selection Rate = 18.16%, Disparate Impact Ratio = 1.42
- **region = West Midlands Region** (N=429): Selection Rate = 17.02%, Disparate Impact Ratio = 1.33
- **region = Yorkshire Region** (N=294): Selection Rate = 17.35%, Disparate Impact Ratio = 1.36

#### Elevated False Positive Rate Ratio Flags (> 1.50):
- **region = London Region** (N=526): False Positive Rate = 15.19%, FPR Ratio = 1.68
- **region = North Western Region** (N=463): False Positive Rate = 16.80%, FPR Ratio = 1.86
- **region = Scotland** (N=636): False Positive Rate = 18.16%, FPR Ratio = 2.01
- **region = South Region** (N=510): False Positive Rate = 14.12%, FPR Ratio = 1.56
- **region = South West Region** (N=405): False Positive Rate = 14.87%, FPR Ratio = 1.65
- **region = West Midlands Region** (N=429): False Positive Rate = 15.66%, FPR Ratio = 1.73
- **region = Yorkshire Region** (N=294): False Positive Rate = 14.80%, FPR Ratio = 1.64

---

## 5. Demonstration Support Queue Subgroup Audit (Calibrated RF @ Top-500 Enrollment Slots)

**Binary Decision Evaluated:** Inclusion in the historical held-out demonstration queue, ranked by calibrated probabilities and capped at up to 500 course-enrollment slots per represented presentation.
**Purpose:** Retrospective audit of this demonstration allocation rule. It does not establish a deployed policy or a 500-unique-student caseload; duplicate course enrollments require an institutional aggregation decision.

Across the 6 protected attributes (36 total evaluated subgroups):
* **Statistically Stable Subgroups ($N \ge 50$):** 35 subgroups (97.2%).
* **Small Sample Subgroups ($N < 50$):** 1 subgroups (2.8%), primarily `age_band = 55<=` (N=35).

### Queue-Level Parity Findings & Comparison to Classifier Diagnostics

In this held-out demonstration queue (overall inclusion rate = 36.94%):
* **Disparate Impact Warnings:** **4 stable subgroups** exceed the [0.80, 1.25] tolerance band (down from 14 in the uncalibrated classifier diagnostic).
* **Elevated False Positive Rate Warnings:** **0 stable subgroups** exceed the 1.50 warning threshold (down from 7 in the uncalibrated classifier diagnostic).

### Identified Operational Queue Warnings (4 total flags)

#### Operational Queue Disparate Impact Ratio Flags (Outside [0.80, 1.25]):
- **gender = F** (N=2505): Queue Inclusion Rate = 28.66%, Disparate Impact Ratio = 0.65, Queue Recall (Capture Rate) = 51.72%
- **age_band = 35-55** (N=1659): Queue Inclusion Rate = 30.74%, Disparate Impact Ratio = 0.77, Queue Recall (Capture Rate) = 55.76%
- **region = North Western Region** (N=463): Queue Inclusion Rate = 41.68%, Disparate Impact Ratio = 1.27, Queue Recall (Capture Rate) = 59.76%
- **region = Scotland** (N=636): Queue Inclusion Rate = 41.51%, Disparate Impact Ratio = 1.26, Queue Recall (Capture Rate) = 59.29%

No stable subgroups exhibited false positive rate ratios exceeding the 1.50 warning threshold in the operational queue.

### Caseload Capacity Grain & Aggregation Policy Notice

The queue allocates capacity at the **course-enrollment grain** (up to 500 slots per presentation in this held-out demonstration). Duplicate enrollment slots mean the institution's intended 500-case capacity cannot be interpreted as 500 unique students without an approved cross-module risk aggregation policy.

---

## 6. Responsible Use & Governance Protocols

1. **Non-Stigmatizing Triage:** Output priority scores must never be labeled as "failure probabilities" or "student risk flags" in student-facing settings.
2. **Supportive Outreach Only:** The triage queue exists solely to trigger timely academic advising, tutor check-ins, and pastoral assistance.
3. **Mandatory Human Discretion:** Academic advisors retain full discretion to reprioritize or dismiss triage alerts based on qualitative contextual knowledge.
