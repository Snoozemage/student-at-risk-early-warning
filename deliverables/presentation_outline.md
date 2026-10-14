# Presentation Outline: Smart Student Retention & Early Warning System

**Format:** 12-Slide Executive & Academic Stakeholder Presentation<br>
**Target Audience:** University Leadership, Faculty Deans, Advising Directors, Academic Evaluators<br>
**Duration:** 20 minutes presentation + 10 minutes Q&A<br>

---

## Slide 1: Title & Executive Vision
- **Slide Title:** Smart Student Retention: Operational Early Warning & Advising Triage
- **Subtitle:** Machine Learning Decision Support for Timely Student Success at Relative Day 35
- **Presenter:** AI Engineering & Learning Analytics Team
- **Key Message:** Early behavioral signals at Day 35 can identify students at elevated risk of course departure; intervention effectiveness in preventing attrition requires prospective, randomized evaluation.

---

## Slide 2: The Core Educational Problem
- **Talking Points:**
  - Distance learning suffers from high, silent attrition (17.39% withdraw after Week 5).
  - Traditional advising discovers disengagement too late (after missed exams or formal dropout).
  - The model predicts post-cutoff risk from information available by Day 35; this study did not compare intervention timing or establish whether later outreach is unrecoverable.
- **Visual:** Flowchart comparing reactive advising vs proactive early warning.

---

## Slide 3: Analytical Formulation & Anti-Leakage Safeguards
- **Talking Points:**
  - **Cutoff Horizon:** Strictly bounded at relative Day 35 (first 5 weeks of study).
  - **Dataset:** Open University Learning Analytics Dataset (27,128 eligible enrollment records).
  - **Anti-Leakage Safeguards:** Zero post-cutoff clicks, zero post-cutoff submissions, zero assessment grades, and zero demographic attributes in the primary predictive feature store.
- **Visual:** Timeline diagram illustrating the closed Day 35 observation window and forward target horizon.

---

## Slide 4: Early Behavioral Signals of Student Attrition
- **Talking Points:**
  - **Coursework Pacing:** Missed due assessments and submission progress were examined as retrospective associations.
  - **VLE Engagement:** Activity and resource-diversity measures are candidate signals for further validation.
  - **Registration Horizon:** The continuous comparison does not validate a binary late-registration cutoff.
  - These observations are not causal and do not establish intervention effectiveness.
- **Visual:** Pair of charts: Weekly VLE activity curves + Assessment milestone progress.

---

## Slide 5: Rigorous Machine Learning Evaluation
- **Talking Points:**
  - Strict group-aware splitting on student ID (zero student overlap between train and test).
  - Heavy class imbalance (17.39% positive base rate) makes accuracy misleading; PR-AUC is primary.
  - **Random Forest Performance:** PR-AUC **0.3101** (+84.5% relative lift over dummy baseline 0.1681).
  - ROC-AUC: 0.6797.
- **Visual:** Precision-Recall Curves comparing Dummy, Logistic Regression, and Random Forest.

---

## Slide 6: Probability Calibration & Decision Quality
- **Talking Points:**
  - Raw tree ensemble probabilities are compressed and unsuited for direct risk thresholds.
  - **Methodology:** Training-only cross-validation selected Isotonic calibration based on persisted selection metadata.
  - **Out-of-Time Test Result:** Expected Calibration Error (ECE) dropped from 0.0115 to **0.0073 (36.2% error reduction)**.
  - The 0.70 Urgent and 0.40 Watch cut points are proposed demonstration settings; probability calibration does not validate decision thresholds or intervention benefit.
- **Visual:** Reliability diagram (Calibration curve) showing 10-bin calibration before and after.

---

## Slide 7: Raw Random Forest Top-K Diagnostic & Chronological Capacity Evaluation
- **Talking Points:**
  - Pooled random-test diagnostic using uncalibrated Random Forest scores: top-500 precision is **37.4%** versus 16.81% test-split baseline; this is not a per-term capacity estimate.
  - **Pooled-test enrichment:** **2.23x** relative to random ranking (187 positive enrollment records among the pooled top 500).
- **Chronological per-term K=500 results:** 2014B: Precision@500=0.394 (197 positive enrollment records captured; 479 expected unique students selected; 21 expected duplicate enrollment slots); 2014J: Precision@500=0.436 (218 positive enrollment records captured; 481 expected unique students selected; 19 expected duplicate enrollment slots). Student-level capacity remains unresolved pending an approved multi-course aggregation policy.
- **Visual:** Pooled-test ranking diagnostic and separate chronological per-term results.

---

## Slide 8: Student Support Queue & Triage Workflow
- **Talking Points:**
  - Demonstration queue deliverable: 2,000 cases across academic terms.
  - **Tier 1 (proposed cutoff $\ge 70\%$):** 2 cases; proposed 3-day human-review response target.
  - **Tier 2 (proposed 40-70%):** 167 cases; proposed 7-day supportive check-in target.
  - **Tier 3 (Standard, $< 40\%$):** 1,831 cases $\to$ Routine monitoring.
- **Visual:** Interactive Streamlit dashboard screenshot showing the Student Support Queue.

---

## Slide 9: Subgroup Fairness & Algorithmic Accountability
- **Talking Points:**
  - Audited 36 demographic subgroups across 6 protected characteristics (age_band, disability, gender, highest_education, imd_band, region).
  - Under the calibrated support queue, **zero stable subgroups exhibit elevated False Positive Ratios ($> 1.50$)**.
  - Observational disparities reflect historical platform usage, not causal deficiencies.
  - Regular fairness re-auditing monitors for emerging disparities across learner subgroups.
- **Visual:** Horizontal bar chart of Disparate Impact Ratios with 4/5ths reference bands.

---

## Slide 10: Ethical Boundaries & Harm Prevention
- **Talking Points:**
  - Exclusively supportive: algorithm CANNOT trigger disciplinary or financial sanctions.
  - Human-in-the-loop: academic advisors hold 100% discretion to override model flags.
  - Stigma-free messaging: students are contacted regarding support resources, never told they are "flagged."
- **Visual:** 3 Core Pillars of Ethical Early Warning diagram.

---

## Slide 11: Implementation Roadmap & Governance
- **Talking Points:**
  - **Phase A (Immediate):** University governance approves multi-course student caseload aggregation policy.
  - **Phase B (Pilot):** 1-semester randomized controlled trial across two STEM departments.
  - **Phase C (Consider broader deployment):** Proceed only if prospective evaluation supports benefit and institutional governance approves the policy and integration.
- **Visual:** 3-phase deployment timeline with clear milestones.

---

## Slide 12: Summary & Discussion
- **Summary Takeaways:**
  1. Pooled-test Top-500 diagnostic showed 2.23x enrichment over random ranking.
  2. Isotonic calibration improved empirical probability alignment (0.0073 ECE) on the held-out evaluation split.
  3. Strict human-in-the-loop guardrails uphold supportive, stigma-free advising protocols.
- **Open for Questions:** Discussion on advisor capacity, multi-course aggregation, and pilot deployment.
