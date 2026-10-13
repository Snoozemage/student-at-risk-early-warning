# Project Charter

**Project Name:** Smart Student Retention & Early Warning System
**Repository:** `smart-student-retention`
**Phase:** Phase 1 — Business Specification and Data Contract
**Date:** 2026-10-13
**Status:** Proposed Specification (Pending Review)

---

## 1. Executive Summary & Problem Statement

Higher education institutions face substantial learner attrition, particularly in distance, blended, and modular learning environments. In large-scale online course offerings, a significant fraction of enrolled students withdraw before completion. When course withdrawal occurs unobserved, institutional support teams lose the opportunity to provide targeted academic, personal, or administrative assistance while it is still timely and actionable.

The Open University Learning Analytics Dataset (OULAD) demonstrates that over 31% of course registrations result in withdrawal. However, many students withdraw very early in the semester, while others persist until later assessment milestones before discontinuing.

Traditional reactive interventions fail because support teams are only alerted after formal unregistration or exam absence. A proactive early-warning system provides educational advisors with an early, evidence-based prioritization queue to identify students exhibiting disengagement signals by **Day 35** of instruction, enabling supportive outreach before formal withdrawal occurs.

---

## 2. Business Objectives & Value Proposition

1. **Early Risk Identification:** Predict whether an eligible enrolled student will withdraw from a specific course module presentation after relative Day 35, using exclusively data available on or before Day 35.
2. **Support Prioritization under Capacity Constraints:** Rank students into actionable risk tiers (Urgent, Watch, Standard) so academic advisors can allocate limited intervention bandwidth (provisional planning capacity: 500 cases per presentation) to learners with the highest risk of withdrawal.
3. **Transparent Decision Support:** Provide interpretable behavioral indicators (e.g., steep drop in Virtual Learning Environment engagement, missing early formative submissions) alongside risk probabilities to guide advisor conversations.
4. **Non-Punitive, Equitable Delivery:** Ensure early-warning risk scores are deployed solely for supportive outreach, never for gatekeeping, course restriction, or automated penalization, backed by continuous fairness auditing.

---

## 3. Project Scope & Boundaries

### 3.1. In Scope
- **Domain:** Early course withdrawal prediction in higher education modular coursework.
- **Dataset:** The Open University Learning Analytics Dataset (OULAD) covering 7 undergraduate modules (`AAA`–`GGG`) across 4 academic presentations (`2013B`, `2013J`, `2014B`, `2014J`), comprising 22 distinct module presentations and 32,593 enrollment records.
- **Prediction Grain:** Individual student enrollment per course offering: `(code_module, code_presentation, id_student)`.
- **Prediction Cutoff:** Relative Day 35 after course presentation start date (inclusive).
- **Primary Cohort Policy:** **Policy C (Strict Clean-Data)**, establishing an eligible cohort of 27,128 records with 4,718 eventual withdrawals (~17.39% positive base rate).
- **Alternative Cohort Policy:** **Policy B (Sensitivity Analysis)**, establishing an alternative cohort of 27,136 records (~17.39% base rate) to verify model robustness against registration data missingness.
- **Deliverables:** Formal data contracts (`config/`), technical documentation (`docs/`), reproducible ingestion/validation pipeline, baseline and ML models, group-aware evaluation protocol, fairness audit, decision support queue (`student_support_queue.csv`), and Streamlit interactive dashboard.

### 3.2. Out of Scope
- **Institutional Dropout Prediction:** This system predicts withdrawal from a specific course module, **not** permanent withdrawal from the Open University.
- **Automated Actions:** The system does not automate academic decisions, cancellations, or administrative actions; human review is mandatory.
- **Causal Claims:** The model identifies associative risk patterns, not causal mechanisms. High predicted risk does not prove that a specific factor caused withdrawal.
- **Mid/Late-Term Prediction:** Prediction beyond Day 35 (e.g., Day 60, Day 100) is out of scope for the primary prototype.
- **Production LMS Integration:** Live API integration with production LMS (e.g., live Moodle/Canvas webhooks) is out of scope.

---

## 4. Key Stakeholders & Users

| Stakeholder Role | Primary Responsibilities & Interaction | Key Value Derived |
|---|---|---|
| **Academic Advisors & Student Support Officers** | Review daily/weekly support queues, conduct proactive student outreach, log intervention status. | Prioritized caseload, transparent reason summaries, actionable early window. |
| **Course Directors & Module Leaders** | Monitor presentation-level engagement trends, identify modules with anomalous attrition patterns. | Cohort-level visibility into early attrition dynamics. |
| **Education Administrators & Deans** | Review resource allocation, capacity constraints, institutional retention trends. | Data-driven advisor capacity planning and retention reporting. |
| **Learners (Beneficiaries)** | Receive timely academic advice, study skills coaching, or pastoral support. | Timely support before falling insurmountable distances behind. |

---

## 5. Success Criteria & Target Benchmarks

### 5.1. Data Engineering & Integrity Criteria
- **Zero Information Leakage:** 100% of feature extraction strictly restricted to relative days $\le 35$. Verified by automated temporal leakage tests.
- **Strict Group Splitting:** 100% disjoint student groups between training and test sets on `id_student`, guaranteeing zero cross-split student leakage.
- **Reproducible Pipeline:** Fully automated data ingestion, validation, and cohort assembly executing cleanly via `uv run`.

### 5.2. Machine Learning & Analytical Criteria (Provisional Planning Targets)
*Note: All performance targets are provisional planning benchmarks subject to baseline modeling validation and institutional review.*
- **Baseline Superiority:** Candidate predictive models (Logistic Regression, Random Forest) must demonstrably outperform the Dummy baseline across PR-AUC, ROC-AUC, and F1-score.
- **Discrimination on Imbalanced Target:** Model should target PR-AUC exceeding the ~17.4% positive class prior (provisional target: PR-AUC $\ge 0.40$).
- **Top-K Operational Utility:** Precision@500 targeted at $\ge 50\%$ (at least 1 in 2 flagged cases in the top 500 represents an actual eventual withdrawal, subject to caseload calibration).

### 5.3. Ethical & Governance Criteria
- **Subgroup Fairness Audit:** Systematic evaluation of disparate impact and false positive rate ratios across gender, age band, disability status, and deprivation deciles (`imd_band`).
- **Support-Oriented Governance:** Formal institutional policy mandating that risk scores remain private to advising teams and are never displayed punitively to students.
