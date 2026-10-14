# Executive Summary: Smart Student Retention & Early Warning System

**Project:** Smart Student Retention & Early Warning System<br>
**Evaluation Scope:** Open University Learning Analytics Dataset (OULAD)<br>
**Prediction Horizon:** Early-warning cutoff at relative Day 35 (first 5 weeks of study)<br>
**Target Variable:** Post-cutoff course withdrawal (`withdrawn_after_cutoff`)<br>

---

## 1. The Strategic Retention Challenge

In distance higher education, disengagement may be less visible in routine classroom interactions. Retrospective withdrawal records describe course outcomes; they do not establish when an advisor could have noticed a student's difficulties or whether those difficulties were recoverable.

The **Smart Student Retention System** was evaluated at **relative Day 35**—the end of the fifth study week—to estimate and rank post-cutoff withdrawal risk. This retrospective evaluation does not show that outreach at Day 35 changes outcomes, that this is the most effective intervention time, or that departure is inevitable.

---

## 2. Key Empirical Findings

Operating under primary **Policy C (Strict Clean-Data)** on 27,128 eligible student enrollment records, retrospective evaluation showed ranking lift and pooled-test top-K enrichment:

| Evaluation Metric | Empirical Value | Context & Significance |
| :--- | :--- | :--- |
| **Eligible Student Cohort** | **27,128 records** | Complete courses across 22 module-presentations |
| **Withdrawal Base Rate** | **17.39%** | 4,718 post-cutoff withdrawals (class imbalance ~1:5) |
| **Evaluation Design** | **Zero-Leakage Group Split** | 80% train / 20% test ($N=5,414$) with 0 student overlap |
| **Model PR-AUC (Random Forest)** | **0.3101** | Primary ranking metric on held-out test cohort |
| **PR-AUC Relative Lift** | **+84.5%** | Measured relative lift over class-prior dummy baseline (0.1681) |
| **Pooled-Test Precision@500 Diagnostic (Raw Random Forest Ranking)** | **37.4%** | 187 positive enrollment records among the top 500 in the pooled random test split; not a per-term capacity estimate |
| **Pooled-Test Enrichment Diagnostic** | **2.23x** | More than double the test-split random-ranking baseline in this pooled diagnostic |
| **Probability Calibration (ECE)** | **0.0073 (36.2% error reduction)** | Isotonic calibration improved empirical probability alignment on this test split |

### Critical Early Warning Indicators

1. **Coursework Pacing:** Missed due assessments and submission progress were examined as retrospective associations with course withdrawal.
2. **VLE Engagement:** Activity level, recency and resource diversity were examined as candidate early signals.
3. **Registration Timing:** The continuous registration-date comparison does not establish a late-registration cutoff; no binary threshold is validated.

*Note: These analyses are observational and retrospective. They do not establish causation or show that an intervention based on a signal improves retention.*

---

## 3. Proposed Human-Reviewed Triage Workflow (Demonstration)

The demonstration queue contains held-out test enrollment records, capped at up to $K=500$ enrollment slots per presentation (2,000 slots across the represented terms). It is not an estimate of a future per-term unique-student caseload. The 0.70/0.40 tier cut points and 3/7-day response targets below are proposed workflow settings, not empirically validated decision thresholds or service levels.

Chronological out-of-time per-term K=500 evaluation using uncalibrated Random Forest ranking scores: 2014B: Precision@500=0.394 (197 positive enrollment records captured; 479 expected unique students selected; 21 expected duplicate enrollment slots); 2014J: Precision@500=0.436 (218 positive enrollment records captured; 481 expected unique students selected; 19 expected duplicate enrollment slots). Student-level capacity remains unresolved pending an approved multi-course aggregation policy.

```
+-----------------------------------------------------------------------------------------+
|                                ADVISOR TRIAGE PIPELINE                                  |
+-----------------------------------------------------------------------------------------+
|  Tier 1: Urgent (Risk >= 70%)     --> Proposed 3-day human-review response target       |
|  Tier 2: Watch (40% <= Risk < 70%) --> Proposed 7-day supportive check-in target         |
|  Tier 3: Standard (proposed Risk < 40%) --> Routine course communications & regular pacing |
+-----------------------------------------------------------------------------------------+
```

- **Urgent Cases (Tier 1):** 2 records (0.1% of demonstration queue) fall in the proposed highest-score review band.
- **Watch Cases (Tier 2):** 167 records (8.3% of demonstration queue) fall in the proposed intermediate-score review band.
- **Standard Cases (Tier 3):** 1,831 records (91.5% of demonstration queue) fall below the proposed review bands.

---

**Workflow status:** Human review and supportive contact are proposed uses of this retrospective ranking. Intervention effectiveness has not been evaluated; any thresholds and response targets require local governance and prospective evaluation before operational adoption.

## 4. Responsible AI & Ethical Boundaries

To prevent algorithmic harm and protect student agency, the system enforces four foundational ethical boundaries:

1. **Exclusively Supportive Interventions:** Algorithmic risk predictions must never trigger punitive actions, academic probation, or financial aid cancellation.
2. **Advisor Discretion:** Early-warning scores represent observational triage signals, not definitive academic destinies. Human advisors maintain absolute discretion to override or dismiss alerts.
3. **Non-Stigmatizing Communication:** Communications with students focus strictly on available resources and learning support; students are never told they were flagged by a "predictive algorithm."
4. **Transparent Fairness Audits:** Operational support queue fairness was evaluated across 6 protected demographic attributes (age_band, disability, gender, highest_education, imd_band, region; 36 subgroups total). Under the calibrated support queue, zero demographic subgroups exhibit elevated False Positive Ratios ($> 1.50$) among statistically stable subgroups (with 1 subgroups flagged as statistically unstable, N < 50). Ongoing monitoring is required as zero detected warnings does not guarantee the complete absence of disparate impact.

---

## 5. Next Steps for Production Implementation

1. **Institutional Aggregation Policy:** Formalize university governance on how multi-course students (173 duplicate slots across 1,827 unique students) are prioritized across faculties.
2. **Prospective Randomized Pilot:** Deploy in a single academic department to test whether proactive outreach measurably lifts retention over control cohorts.
3. **Advisor Feedback Integration:** Incorporate qualitative advisor outcomes back into the training feedback loop.
