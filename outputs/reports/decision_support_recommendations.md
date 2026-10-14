# Decision Support & Caseload Prioritization Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Output Deliverable:** [`outputs/predictions/student_support_queue.csv`](file:///c:/Users/dangnguyen/Repo/project_for_offer/student-at-risk-early-warning/outputs/predictions/student_support_queue.csv)
**Evaluation Candidate Cohort:** Held-Out Group-Aware Test Cohort (5,414 candidate enrollment records across four academic terms: 2013B, 2013J, 2014B, 2014J)
**Generated Support Queue:** Capped demonstration triage queue (Top-500 enrollment slots per term $\times$ 4 terms = 2,000 total slots across 1,827 unique students)
**Triage Capacity Limit:** K=500 course-enrollment records per academic term (cross-module advisor caseload)

---

## 1. Caseload Triage Breakdown

| Priority Tier | Operational Definition | Response SLA | Active Cases | Caseload % | Prescribed Action |
|---|---|---|---|---|---|
| **Tier 1 (Urgent)** | Calibrated Prob $\ge 0.70$ | **3 business days** | 2 | 0.1% | Direct personal outreach (phone/1-on-1 email) |
| **Tier 2 (Watch)** | $0.40 \le$ Calibrated Prob $< 0.70$ | **7 business days** | 167 | 8.3% | Automated check-in & tutor forum check |
| **Tier 3 (Standard)** | Calibrated Prob $< 0.40$ | Routine | 1,831 | 91.5% | Standard course delivery & monitoring |

---

## 2. Caseload Capacity & Unique-Student Representation Audit

| Academic Term | Caseload Slots (K=500 Limit) | Unique Students | Duplicate Slots | Multi-Course Students | Institutional Aggregation Status |
|---|---|---|---|---|---|
| 2013B | 500 | 500 | 0 | 0 | `unresolved_pending_institutional_policy` |
| 2013J | 500 | 500 | 0 | 0 | `unresolved_pending_institutional_policy` |
| 2014B | 500 | 485 | 15 | 15 | `unresolved_pending_institutional_policy` |
| 2014J | 500 | 477 | 23 | 23 | `unresolved_pending_institutional_policy` |
| **ALL_TERMS_TOTAL** | 2,000 | 1,827 | 173 | 167 | `unresolved_pending_institutional_policy` |

> **Important Operational Governance Notice:**
> Institutional business requirements specify academic advising capacity at approximately 500 cases per academic term across modules.
> In this historical demonstration queue, the triage ranking operates at the **course-enrollment record grain** (`code_module`, `code_presentation`, `student_key`) across four terms: 2013B, 2013J, 2014B, and 2014J.
> Learners concurrently enrolled in multiple course offerings during the same term may appear in multiple module-presentation slots (173 duplicate slots across all presentations).
> Because an institutional policy defining how multi-course learner risk should be aggregated (e.g., maximum module risk vs mean risk) has not yet been established, a true 500-unique-student operational queue remains **unresolved pending institutional policy definition**. In-sample training records are strictly excluded.

---

## 3. Probability Calibration & Reliability

Model risk probabilities are calibrated using **Isotonic Calibration** fitted strictly on the 80% training partition via 5-fold `GroupKFold` cross-validation:
* **Held-Out Test Brier Score:** 0.13135
* **Expected Calibration Error (ECE):** 0.00734 (substantial error reduction over uncalibrated RF)
* **Calibrated Probability Range:** [0.0000, 0.7924] (mean: 0.1731)
* **Calibration Artifact:** [`models/baseline/random_forest_calibrated.joblib`](file:///c:/Users/dangnguyen/Repo/project_for_offer/student-at-risk-early-warning/models/baseline/random_forest_calibrated.joblib)

---

## 4. Decision Support Queue Data Dictionary

The generated queue delivers structured, pseudonymized context for academic advising teams:

1. `student_key`: Pseudonymized student identifier (`id_student`).
2. `code_module`: Course module offering (AAA–GGG).
3. `code_presentation`: Academic term (2013B, 2013J, 2014B, 2014J).
4. `prediction_cutoff_day`: Always `35` (relative Day 35 cutoff).
5. `risk_probability`: Calibrated model probability of withdrawal after Day 35.
6. `priority_tier`: Urgent, Watch, or Standard.
7. `support_reason_summary`: Top 2–3 observed behavioral signals associated with risk (e.g. days inactive, missed scheduled assignments). Observational only; not a causal explanation.
8. `review_status`: Advisor workflow tracker (`Pending`, `Under Review`, `Contacted`, `Resolved`, `No Action Needed`).
9. `recommended_next_step`: Concrete intervention guidelines tailored to risk profile.

---

## 5. Human-in-the-Loop Operational Workflow

```
[Day 35 Data Cutoff]
        │
        ▼
[Calibrated Scoring Engine (Trained strictly on train partition)]
        │
        ▼
[Priority Triage Engine (Tier 1 / 2 / 3)]
        │
        ▼
[Caseload Allocation (Max K=500 per term on unseen cohort)]
        │
        ▼
[Advisor Context Review & Behavioral Verification]
        │
        ├─► [Pastoral Outreach: Personal/Technical Disruptions]
        ├─► [Academic Outreach: Assignment Pacing & Scaffolding]
        └─► [Advisor Override: Known Context / No Action Needed]
        │
        ▼
[Log Outcome in System: Contacted / Resolved]
```

### Ethical Guardrails:
* **No Automated Penalties:** Predictions never trigger automated drops or administrative holds.
* **Supportive Communications:** Stigmatizing language is prohibited in all student communications.
* **Professional Override:** Advisors have complete authority to adjust priority tiers based on unmodeled student circumstances.
