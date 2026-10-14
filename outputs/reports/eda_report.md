# PHASE 4 — EXPLORATORY DATA ANALYSIS & LEARNING-PATTERN REPORT

**Date:** 2026-10-13
**Analysis Target:** Withdrawal strictly after relative Day 35 cutoff (`withdrawn_after_cutoff`)
**Eligible Population (Policy C):** 27,128 student-course observations (4,718 withdrawn, 22,410 retained)
**Positive Class Base Rate:** 17.39%

---

## 1. Executive Summary & Framing

This exploratory data analysis investigates early behavioral, academic, and temporal signals associated with course withdrawal after relative Day 35 in the Open University Learning Analytics Dataset (OULAD). Operating strictly within the information boundary available by the Day 35 cutoff, this analysis isolates verifiable early indicators of student attrition to guide **Phase 5 (Temporal Feature Engineering)**.

> [!IMPORTANT]
> **Association vs. Causation Warning:** All relationships documented herein represent observational statistical associations. They must not be interpreted as causal determinants. For example, lower VLE click volume reflects disengagement rather than a direct biological or cognitive cause of failure; artificially requiring students to click links without instructional support cannot be assumed to improve retention.

---

## 2. Evidence-Backed Analytical Observations

### Observation 1: Severe Cross-Presentation and Domain Heterogeneity
- **Empirical Finding:** Post-cutoff withdrawal rates vary substantially across the 22 module-presentations, ranging from a minimum of **3.7%** to a maximum of **28.3%** (mean across offerings: 16.4%).
- **Module and Discipline Differences:** In the OULAD dataset specification (Kuzilek et al., 2017), modules are categorized into STEM (`CCC`, `DDD`, `EEE`, `FFF`) and Social Sciences (`AAA`, `BBB`, `GGG`). While STEM offerings show a higher overall mean attrition rate (19.7%) than Social Sciences (11.6%), substantial intra-domain variation exists across specific modules (e.g., Module `EEE` has 11.9%–13.3% attrition while Module `CCC` has 26.0%–28.3%).
- **Seasonality Patterns:** February (`B`) presentations exhibit higher attrition on average (18.3%) than October (`J`) presentations (15.1%) across offerings (pooled student-level rates: 18.9% vs 16.4%). However, seasonal disparities are not uniform across all offerings: while `CCC`, `DDD`, and `EEE` experience higher attrition in February cohorts, `BBB 2014B` (11.9%), `FFF 2013B` (16.6%), and `GGG 2014B` (7.7%) had lower withdrawal rates than their respective October cohorts (`BBB 2014J`: 12.8%; `FFF 2013J`: 18.2%; `GGG 2014J`: 12.5%).
- **Architectural Consequence for Phase 5:** Features must incorporate course-context normalizations (e.g. `code_module` and `presentation_season`), and model evaluation must employ group-aware or chronological blocking to prevent memorization of course-level base rates.

### Observation 2: Early Intensity and Diversity of Virtual Learning Engagement (VLE)
- **Empirical Finding:** Retained students engage in significantly higher interaction volumes prior to Day 35 compared to withdrawn students (Observation window: Days -25 to 35; N = 27,128).
  - Retained students logged an average of **430.2 clicks** (median: 275.0) across **17.7 active days** (median: 16.0).
  - Withdrawn students logged an average of **334.3 clicks** (median: 217.0) across **15.3 active days** (median: 13.0).
- **Statistical Significance & Effect Size:** Mann-Whitney U test confirms a statistically significant difference ($p = 1.36e-33$) with Cliff's delta = -0.1117 and Cohen's d = -0.1948.
- **Breadth of Exploration:** Retained students interacted with a median of 26 distinct resource sites compared to 24 for withdrawn students ($p = 2.47e-11$, Cliff's delta = -0.0617).

### Observation 3: Longitudinal Engagement Decay & Trajectory Divergence
- **Empirical Finding:** Analysis of weekly click trajectories from Week -3 through Week 5 reveals that interaction volumes between retained and withdrawn students are comparable during pre-course orientation (Weeks -3 to -1).
- **Divergence Point:** Engagement diverges immediately upon official course commencement (Week 0) and continues to widen across Weeks 2–5.
- **Recency of Inactivity:** By Day 35, students who eventually withdraw have a longer period of inactivity: a median of 3.0 days elapsed since last click versus 2.0 days for retained students ($p = 8.22e-36$, Cliff's delta = 0.1140).

### Observation 4: Early Coursework Milestone Failure as a Critical Warning Signal
- **Empirical Finding:** Across the 18 module-presentations with continuous assessments due on or before Day 35 under the Reasonable Opportunity Policy (Eligible cohort: N = 24,700 learners; 4 offerings have 0 assessments due by Day 35):
  - Retained students achieved a mean submission rate of **90.7%** (with 69.8% submitted on time relative to scheduled coursework).
  - Withdrawn students achieved a mean submission rate of only **78.4%** (49.0% on time).
  - Eventual withdrawals missed at least one scheduled assessment in **26.3%** of cases, compared to **11.3%** for retained students.
- **Milestone Failure Odds Ratio:** Students with due coursework who failed to submit at least one due assessment by Day 35 had an **Odds Ratio of 2.80** (95% CI: [2.58, 3.03], Fisher's exact $p = 1.72e-132$).
- **Reasonable Opportunity & Non-Penalty Invariants:**
  - For cohort learners in offerings with zero assessments due by Day 35 (`DDD 2013B`, `GGG 2013J`, `GGG 2014B`, `GGG 2014J`), or whose deadlines strictly preceded registration without satisfaction, `submission_rate` is represented as undefined (NaN) and `has_missed_due_assessment` is 0.
  - Across the cohort, exactly 5 records had deadlines prior to registration without prior banked submission:
    - 2 students had all due assessments scheduled prior to registration and were excluded from the assessment-due denominator ($N: 24,702 \to 24,700$; `BBB 2013J 1170618` [retained, leaves Cell b] and `FFF 2014B 628323` [withdrawn, leaves Cell a]).
    - 2 students submitted their subsequent post-registration assessment on time and were reclassified from missed to non-missed (`CCC 2014J 350267` [withdrawn, Cell a $\to$ Cell c] and `CCC 2014J 245600` [retained, Cell b $\to$ Cell d]).
    - 1 student remained classified as missed because their subsequent post-registration assessment was also unsubmitted (`CCC 2014J 589463` [retained, stays in Cell b]).
  - Net transitions: $\Delta a = -2, \Delta b = -2, \Delta c = +1, \Delta d = +1$, reducing total missed cases from 3,477 to 3,473 and increasing non-missed cases from 21,225 to 21,227.
- **Phase 5 Guidance:** Milestone adherence constitutes a major early behavioral signal, but must be paired with indicator variables (`has_assessments_due`) to accommodate curriculum pacing differences.

### Observation 5: Registration Timing (Exploratory)
- **Group Comparison:** Median `date_registration` is **-52.0 days** for retained and **-58.0 days** for withdrawn learners relative to Day 0 (Mann-Whitney U $p = 1.72e-09$, Cliff's delta = -0.0557). Negative values mean registration before Day 0; more negative values mean earlier registration. The `registration_lead_time_days` feature is `-date_registration`, so positive/larger values also mean earlier registration.
- **Interpretation Limit:** The withdrawn group registered earlier at the median. The test's p-value alone gives no direction, and this retrospective comparison does not justify risk-directed targeting or establish a causal relationship or cutoff.

### Observation 6: Missingness and Contextual Demographic Indicators
- **Missingness Stability:** `imd_band` missingness occurs in 3.90% of retained students vs 3.09% of withdrawn students. Fisher's exact test indicates an unadjusted statistically significant difference ($p = 0.0079$). However, the absolute delta is small (-0.81 percentage points; 3.90% in retained vs 3.09% in withdrawn), indicating a negligible practical effect size rather than substantial selective attrition bias.
- **Academic Preparedness:** Eventual withdrawals exhibit higher numbers of previous course attempts (`num_of_prev_attempts`) compared to retained students (retained mean: 0.15, withdrawn mean: 0.20; Mann-Whitney U $p = 7.79e-12$), confirming prior academic struggle history as a relevant background indicator.

---

## 3. Summary Tables

### Table 1: Statistical Comparisons of Candidate Continuous Predictors
| Feature | N (Retained) | N (Withdrawn) | Retained Mean | Retained Median | Withdrawn Mean | Withdrawn Median | Mann-Whitney U | p-value | Cliff's Delta | Cohen's d |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vle_total_clicks` | 22,410 | 4,718 | 430.23 | 275.0 | 334.28 | 217.0 | 46959524.0 | 1.36e-33 | -0.1117 | -0.1948 |
| `vle_active_days` | 22,410 | 4,718 | 17.71 | 16.0 | 15.35 | 13.0 | 47301134.0 | 4.97e-30 | -0.1052 | -0.1969 |
| `vle_distinct_sites` | 22,410 | 4,718 | 32.16 | 26.0 | 29.40 | 24.0 | 49602285.5 | 2.47e-11 | -0.0617 | -0.1180 |
| `vle_precourse_clicks` | 22,410 | 4,718 | 78.75 | 29.0 | 58.28 | 20.0 | 48500947.0 | 3.21e-19 | -0.0826 | -0.1498 |
| `vle_recent_clicks` | 22,410 | 4,718 | 118.84 | 64.0 | 88.13 | 44.0 | 46460904.0 | 2.85e-39 | -0.1211 | -0.1905 |
| `days_since_last_vle` | 22,410 | 4,718 | 6.07 | 2.0 | 7.22 | 3.0 | 58892584.0 | 8.22e-36 | 0.1140 | 0.0944 |
| `vle_click_velocity_ratio` | 22,410 | 4,718 | 1.12 | 1.0 | 1.07 | 0.9 | 50124149.5 | 2.01e-08 | -0.0518 | -0.0531 |
| `vle_quiz_clicks` | 22,410 | 4,718 | 56.02 | 1.0 | 46.83 | 1.0 | 50759222.0 | 5.28e-06 | -0.0398 | -0.0794 |
| `vle_forum_clicks` | 22,410 | 4,718 | 114.53 | 47.0 | 80.37 | 30.0 | 46659916.5 | 4.52e-37 | -0.1174 | -0.1610 |
| `studied_credits` | 22,410 | 4,718 | 74.44 | 60.0 | 84.23 | 60.0 | 59727775.0 | 4.75e-53 | 0.1298 | 0.2582 |
| `num_of_prev_attempts` | 22,410 | 4,718 | 0.15 | 0.0 | 0.20 | 0.0 | 54792543.0 | 7.79e-12 | 0.0365 | 0.1010 |
| `date_registration` | 22,410 | 4,718 | -65.42 | -52.0 | -70.81 | -58.0 | 49920939.0 | 1.72e-09 | -0.0557 | -0.1136 |
| `submission_rate` | 20,167 | 4,533 | 0.91 | 1.0 | 0.78 | 1.0 | 38860319.5 | 9.49e-151 | -0.1498 | -0.4163 |
| `has_missed_due_assessment` | 22,410 | 4,718 | 0.10 | 0.0 | 0.25 | 0.0 | 60840671.0 | 8.06e-175 | 0.1509 | 0.4583 |

### Table 2: Assessment Milestone Failure Odds Ratio
- **Contingency Table (Students with Due Assessments <= Day 35):**
  - Eventual Withdrawals with Missed Due Assessment (a): 1,192
  - Retained Students with Missed Due Assessment (b): 2,281
  - Eventual Withdrawals with Zero Missed Assessments (c): 3,341
  - Retained Students with Zero Missed Assessments (d): 17,886
- **Odds Ratio:** **2.80** (95% CI: [2.58, 3.03], Fisher exact $p = 1.72e-132$)

### Table 3: Missingness Comparisons by Outcome Group
| Feature | Retained Missing (%) | Withdrawn Missing (%) | Delta (%) | Fisher p-value |
| :--- | :--- | :--- | :--- | :--- |
| `imd_band` | 3.9% | 3.09% | -0.81% | 0.0079 |
| `date_registration` | 0.0% | 0.0% | 0.0% | 1.0000 |
| `studied_credits` | 0.0% | 0.0% | 0.0% | 1.0000 |

---

## 4. Methodological Limitations & Analytical Caveats

1. **Observational Nature:** All derived signals are descriptive behavioral correlations under distance-learning conditions, not direct pedagogical causes. Mandating interaction volume without instructional support cannot be assumed to improve retention.
2. **Unequal Assessment Due Schedules:** Not all modules schedule continuous assessments prior to Day 35 (e.g., Module `GGG` has 0 assessments due <= Day 35). Assessment features must be handled with appropriate indicator variables (`has_assessments_due`) so modules without early assignments are not penalized.
3. **VLE Logging Density Variation:** Different modules utilize online resources with varying intensity (e.g. STEM modules have extensive online quizzing, whereas Humanities rely heavily on printed textbooks). Course-relative scaling or presentation fixed effects are critical.
4. **Repeated-Observation Clustering Limitation:** The 27,128 cohort observations comprise enrollments across 24,567 unique students (2,450 students appear in multiple offerings, accounting for 5,011 records). Unadjusted two-sample statistical tests assume independent observations; mild positive intra-student correlation may slightly understate standard errors and overstate asymptotic test precision. These statistics serve as exploratory heuristic ranking signals rather than formal confirmatory causal proofs. Phases 5 and 6 will enforce grouped cross-validation blocking on student identifiers to ensure independence between training and test sets.

---

## 5. Candidate Features Recommended for Phase 5

Based on the verified empirical associations, the following 12 core early signals are justified for inclusion in the Phase 5 temporal feature pipeline:
1. `vle_total_clicks` (Overall engagement volume in $[-25, 35]$)
2. `vle_active_days` (Pacing consistency)
3. `vle_distinct_sites` (Breadth of resource interaction)
4. `vle_recent_clicks` (Clicks in days $[22, 35]$)
5. `vle_click_velocity_ratio` (Ratio of recent to overall engagement rate)
6. `days_since_last_vle` (Inactivity recency with sentinel 65)
7. `assessments_due_count` (Curriculum pacing expectation with opportunity)
8. `submissions_count_lte_cutoff` (Submission volume)
9. `submitted_on_time_count` (Punctuality)
10. `has_missed_due_assessment` (Milestone failure indicator)
11. `date_registration` (Registration lead time before start)
12. `presentation_season` (February vs October seasonality indicator)
