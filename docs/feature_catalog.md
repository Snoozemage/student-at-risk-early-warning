# Feature Catalog & Temporal Availability Specification

**Project:** Smart Student Retention & Early Warning System
**Phase:** Phase 1 — Business Specification and Data Contract
**Date:** 2026-10-13
**Status:** Proposed Specification (Pending Review)

---

## 1. Feature Engineering Principles & Anti-Leakage Constraints

1. **Temporal Horizon Invariant:** Every predictive feature must be derived strictly from activity observed on or before **relative Day 35** ($t \le 35$).
2. **Strict Outcome Exclusion:** Neither `final_result`, `date_unregistration`, nor any derived indicator of eventual withdrawal may be used as a predictive feature.
3. **Assessment Score Availability Constraint:** `studentAssessment.score` records the grade awarded by a tutor or computer. However, OULAD does **not** record the date when grades were returned to students or advisors. Because grading turnaround latency cannot be proven at Day 35, **assessment scores are excluded from the primary Day-35 baseline models** to prevent retrospective temporal leakage. Only submission occurrence, count, and timeliness are permitted.
4. **Primary Baseline Feature Allowlist:** Primary models are restricted strictly to **22 non-demographic predictors** across four functional groups: Course Context (4), Registration (2), VLE Digital Engagement (11), and Assessment Progress (5).
5. **Demographic Variables Policy:** Demographic attributes are excluded from the primary baseline feature allowlist. However, excluding demographic variables does **not** by itself eliminate algorithmic bias or proxy effects, as behavioral engagement patterns may correlate with learner background. Systematic subgroup fairness auditing across protected attributes remains essential and mandatory.
6. **Appropriate Grain Aggregation:** Child logs (`studentVle`, `studentAssessment`) must be aggregated to the observation grain `(code_module, code_presentation, id_student)` before merging onto the cohort table.

---

## 2. Reconciled Temporal Observation Windows

All temporal intervals $[d_{\text{start}}, d_{\text{end}}]$ are defined as **inclusive closed integer intervals** of relative days:
$$\text{Duration (days)} = d_{\text{end}} - d_{\text{start}} + 1$$

* **Total Observation Window ($[-25, 35]$):** 61 inclusive days. Covers the earliest recorded pre-course click in OULAD (Day -25) through the Day 35 cutoff.
* **Pre-Course Window ($[-25, -1]$):** 25 inclusive days. Strictly prior to course start date (Day 0); captures early orientation and syllabus exploration. Disjoint from active course.
* **Active Course Window ($[0, 35]$):** 36 inclusive days. Begins on Day 0 (official start date) through Day 35 (end of Study Week 5).
* **Recent Window ($[22, 35]$):** Exactly 14 inclusive days ($35 - 22 + 1 = 14$). Captures engagement vitality during the two weeks immediately preceding the cutoff.
* **Velocity Window ($[29, 35]$):** Exactly 7 inclusive days ($35 - 29 + 1 = 7$). Captures engagement acceleration/deceleration during the week immediately preceding the cutoff.
* **Prior Baseline Active Window ($[0, 28]$):** Exactly 29 inclusive days ($28 - 0 + 1 = 29$). Serves as the active instruction baseline period against which the 7-day velocity window is compared.

### Velocity Ratio Definition
$$\text{vle\_velocity\_ratio} = \frac{\sum \text{clicks}_{[29, 35]} / 7}{\left(\sum \text{clicks}_{[0, 28]} / 29\right) + 1.0}$$
* **Numerator:** Average daily clicks during the 7-day velocity window $[29, 35]$.
* **Denominator:** Average daily clicks during the 29-day baseline window $[0, 28]$, with a $+1.0$ Laplace smoothing term to prevent division by zero for inactive students.

### Representation of Unobserved VLE Inactivity
For a student with zero recorded VLE interactions on or before Day 35:
* In feature transformation (prior to numeric pipeline imputation), `days_since_last_vle_interaction` is populated with the sentinel value **65**.
* The maximum possible observed date gap between the earliest interaction (Day -25) and Day 35 is $35 - (-25) = 60$ days. The value 65 is an **upper-bound sentinel value strictly greater than the maximum possible observed gap (60 days)**, ensuring that decision trees can cleanly isolate non-active students from learners who engaged on Day -25.
* A companion binary indicator `has_no_vle_activity_by_cutoff` is explicitly assigned `1` (and `0` for students with $\ge 1$ interaction).
* Because the sentinel 65 is filled during feature extraction, `days_since_last_vle_interaction` contains no missing values and is **not** modified by the general numerical median imputer. The general imputer (`SimpleImputer(strategy="median", add_indicator=True)`) operates on other numerical columns with genuine missingness (such as `registration_lead_time_days`).

---

## 3. Primary Baseline Feature Allowlist (Exact Count: 22 Features)

Primary predictive models are strictly restricted to the following 22 non-demographic features across 4 groups:

### 3.1. Course Context Features (Count: 4)
| Feature Name | Source Table | Source Field | Definition / Derivation Rule |
|---|---|---|---|
| `code_module` | `studentInfo` | `code_module` | Nominal module code (`AAA`–`GGG`). One-Hot encoded. |
| `presentation_season` | `studentInfo` | Derived from `code_presentation` | Recurring presentation season: `B` (February start / Spring) or `J` (October start / Autumn). Replaces raw year-specific strings `2013B`/`2014B` to ensure stable generalization out-of-time under `OneHotEncoder(handle_unknown="ignore")`. |
| `studied_credits` | `studentInfo` | `studied_credits` | Total concurrent credits studied (Median: 60). |
| `num_of_prev_attempts` | `studentInfo` | `num_of_prev_attempts` | Number of previous attempts on this module (0 to 6). |

### 3.2. Registration Features (Count: 2)
| Feature Name | Source Table | Source Field | Definition / Derivation Rule |
|---|---|---|---|
| `registration_lead_time_days` | `studentRegistration` | `date_registration` | Days relative to Day 0: $-\text{date\_registration}$. **Sign Convention:** Positive for advance registration before Day 0 (e.g., $-(-30) = +30$ days); zero for Day 0; negative for eligible late registration after Day 0 (e.g., $-(+10) = -10$ days). Under Policy B, evaluates to null if registration date is unrecorded. |
| `is_late_registration` | `studentRegistration` | `date_registration` | Binary indicator: $1$ if $0 < \text{date\_registration} \le 35$; $0$ if $\text{date\_registration} \le 0$; $\text{null}$ (missing) if $\text{date\_registration}$ is unrecorded. Under primary Policy C, all 27,128 records have confirmed registration dates (0 missing). Under Policy B sensitivity analysis, the 7 records with missing registration dates evaluate to null (missing) rather than 0, allowing the numerical median/indicator imputer to handle them consistently with `registration_lead_time_days`. |

### 3.3. VLE Digital Engagement Features (Count: 11; Filtered to $\text{date} \le 35$)
| Feature Name | Source Table | Window | Definition / Derivation Rule |
|---|---|---|---|
| `vle_total_clicks_window` | `studentVle` | $[-25, 35]$ | Total clicks across entire observation window. |
| `vle_active_days_window` | `studentVle` | $[-25, 35]$ | Count of distinct active calendar days with $\ge 1$ click. |
| `vle_unique_sites_accessed` | `studentVle` | $[-25, 35]$ | Count of distinct VLE materials/pages visited. |
| `vle_pre_course_clicks` | `studentVle` | $[-25, -1]$ | Total clicks during pre-course orientation prior to Day 0. |
| `vle_recent_clicks_14d` | `studentVle` | $[22, 35]$ | Total clicks during the 14 days immediately preceding cutoff. |
| `vle_velocity_ratio` | `studentVle` | $[0, 35]$ | Ratio of average daily clicks $[29, 35]$ to $[0, 28]$ (formula in Section 2). |
| `days_since_last_vle_interaction` | `studentVle` | Cutoff | $35 - \max(\text{date} \le 35)$; filled with sentinel value 65 during feature extraction if 0 clicks. |
| `has_no_vle_activity_by_cutoff` | `studentVle` | $[-25, 35]$ | Binary indicator: $1$ if 0 clicks recorded by Day 35, else $0$. |
| `vle_clicks_forumng` | `studentVle` $\bowtie$ `vle` | $[-25, 35]$ | Clicks on collaborative discussion forums (`activity_type == 'forumng'`). |
| `vle_clicks_oucontent` | `studentVle` $\bowtie$ `vle` | $[-25, 35]$ | Clicks on core interactive syllabus content (`activity_type == 'oucontent'`). |
| `vle_clicks_quiz` | `studentVle` $\bowtie$ `vle` | $[-25, 35]$ | Clicks strictly on formative quizzes (`activity_type == 'quiz'`; excludes `subpage` or other types). |

### 3.4. Assessment Progress Features (Count: 5)
All assessment features are constructed by joining `studentAssessment` $\bowtie$ `assessments` on `id_assessment` to recover `code_module` and `code_presentation`.

#### Assessment Universe Specification:
- **Scope ($A_{\text{due}}$):** Restricted strictly to **continuous non-exam assessments** (`assessment_type != 'Exam'` with non-null scheduled deadline $\text{date} \le 35$). All five assessment progress features are strictly anchored to this common due-assessment universe.
- **Handling of Early Submissions for Future Assessments:** Empirical inspection identifies 3,196 genuine submissions across 2,350 students submitted on or before Day 35 for assignments whose scheduled deadlines occur strictly after Day 35 (e.g., TMA 2 due Day 47 submitted on Day 31). In the primary baseline contract, these are **excluded from due-progress counts** to preserve strict mathematical consistency ($\text{submissions\_count\_lte\_cutoff} = \text{submitted\_on\_time\_count} + \text{late\_submissions\_count\_lte\_cutoff} \le \text{assessments\_due\_count}$). If future assignments were counted, 2,186 students would present an operational anomaly where on-time submissions exceed total due assessments. Student engagement on future materials is captured through VLE interaction signals.
- **Distribution Check of `assessments_due_count` across Policy C Cohort (N = 27,128):**
  - $0$ due assessments: **2,426 records** (8.94%) — Module GGG.
  - $1$ due assessment: **20,208 records** (74.49%) — Modules AAA, BBB, DDD (2013J/2014B/2014J), EEE, FFF.
  - $2$ due assessments: **4,494 records** (16.57%) — Modules CCC (2014B, 2014J) and DDD (2013B).
  *Conclusion:* `assessments_due_count` is **not constant** across the cohort and captures meaningful pacing variations.
- **Exam Exclusion:** Final examinations (and offline components without electronic submission logs) are excluded from the due assessment universe. An unrecorded assessment result is never equated with a missed submission without verified deadline evidence.
- **Handling of Banked Records (`is_banked`):** The OULAD documentation specifies `is_banked` as an assessment result transferred from a prior course attempt (all 1,909 banked records are timestamped with administrative sentinel `date_submitted = -1`). Banked records reflect prior-year attainment, not current study effort. Therefore, banked records are excluded from genuine current-presentation submission counts (`is_banked == 0`). However, because banked credits officially satisfy course requirements, learners with banked records on due assessments are **not** penalized as having missed due submissions.

| Feature Name | Source Table(s) | Definition / Derivation Rule |
|---|---|---|
| `assessments_due_count` | `assessments` | Count of scheduled non-exam assessments (`TMA`/`CMA`) in the due universe ($\text{date} \le 35$). |
| `submissions_count_lte_cutoff` | `studentAssessment` $\bowtie$ `assessments` | Count of genuine current-presentation submissions for assessments in the due universe with $\text{date\_submitted} \le 35 \land \text{is\_banked} == 0 \land \text{date} \le 35$. |
| `submitted_on_time_count` | `studentAssessment` $\bowtie$ `assessments` | Count of genuine current-presentation submissions for assessments in the due universe with $\text{date\_submitted} \le \text{date} \land \text{date\_submitted} \le 35 \land \text{is\_banked} == 0 \land \text{date} \le 35$. |
| `late_submissions_count_lte_cutoff` | `studentAssessment` $\bowtie$ `assessments` | Count of genuine current-presentation submissions for assessments in the due universe with $\text{date} < \text{date\_submitted} \le 35 \land \text{is\_banked} == 0 \land \text{date} \le 35$. |
| `missing_due_submissions_count` | `assessments` $\bowtie$ `studentAssessment` | Count of scheduled non-exam assessments in the due universe ($\text{date} \le 35$) where learner has **neither** a genuine submission by cutoff ($\text{date\_submitted} \le 35 \land \text{is\_banked} == 0$) **nor** a banked credit ($\text{is\_banked} == 1$). |

---

## 4. Demographic Attributes Policy: Audit-Only vs. Sensitivity Experiment

To support responsible predictive analytics, student demographic and background attributes are handled according to an explicit two-tier policy:

1. **Primary Baseline Model (Excluded):**
   `gender`, `age_band`, `disability`, `imd_band`, `highest_education`, and `region` are excluded from the primary 22-feature allowlist as a safeguard against direct disparate treatment.
   *Caveat:* Excluding demographic variables does not by itself eliminate proxy effects or guarantee fairness, because behavioral features (e.g. course choice, study credits, activity schedules) can correlate with learner background. Continuous subgroup fairness auditing remains essential and mandatory.
2. **Fairness Subgroup Disparity Audit (Included):**
   Demographic fields are retained in the evaluation dataset to systematically audit whether the model produces disparate impact, elevated false positive rates, or unequal recall across demographic groups.
3. **Optional Demographic-Feature Sensitivity Experiment (Secondary):**
   A secondary experimental pipeline will train candidate models with demographic features included alongside behavioral features to:
   - Measure incremental predictive performance when demographic features are added.
   - Examine changes in subgroup fairness metrics to evaluate potential indirect proxy effects.
   - *Note:* This experiment measures performance and metric deltas; it does not claim to prove that the primary baseline has zero proxy reliance.
