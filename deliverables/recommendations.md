# Evidence-Based Recommendations & Decision Support Guidelines

**Project:** Smart Student Retention & Early Warning System<br>
**Framework:** Decision Support & Human-in-the-Loop Advising<br>

---

## 1. Structured Institutional Recommendations

Each recommendation connects observed empirical patterns to concrete institutional actions, detailing supporting evidence, target cohorts, uncertainties, and necessary verification.

### Recommendation 1: Pilot Early Milestone Review for Missed Coursework

- **Empirical Evidence:** In module-presentations with coursework due on or before Day 35, missing scheduled continuous assessments is strongly associated with eventual course departure ($p = 8.06e-175$, Mann-Whitney $U$ test).
- **Proposed Review Cohort:** Records in the proposed highest-score demonstration band with `missing_due_submissions_count >= 1`; the score band is a pilot configuration, not a validated risk threshold.
- **Proposed Pilot Action:** Test a rapid-response coursework extension protocol: when an assessment deadline passes without submission, offer a supportive 7-day penalty-free extension and academic tutoring. A 3-business-day advisor-review target may be evaluated as a pilot setting; it is not a validated service level, and the effectiveness of outreach or extensions remains unknown pending prospective evaluation.
- **Uncertainties & Alternative Explanations:** Missing an early assessment may reflect external personal crises (illness, bereavement, work pressures) rather than academic disinterest. An extension alone may not address underlying life disruptions.
- **Additional Data Needed:** Post-extension submission completion rates and advisor contact logs to determine if coursework flexibility translates into long-term course completion.

---

### Recommendation 2: Review of Extended VLE Inactivity

- **Empirical Evidence:** In this retrospective cohort, days since last VLE interaction differed between withdrawn and retained learners ($p = 8.22e-36$, Mann-Whitney $U$ test). This continuous comparison does not validate a 14-day threshold.
- **Proposed Review Cohort:** Students in Tier 2 (Watch) or Tier 1 (Urgent) with longer observed inactivity, using the measure as a continuous review signal. No inactivity threshold has been validated; evaluate any cutoff before operational use.
- **Proposed Pilot Action:** Evaluate supportive email or SMS check-ins about platform access barriers (e.g. login difficulties, textbook availability), with an optional link to book an advisor consultation. Their effect on retention is unknown until prospectively evaluated.
- **Uncertainties & Alternative Explanations:** Some students download offline learning materials, PDF study guides, or read physical course textbooks, appearing inactive on the web platform while remaining actively engaged offline.
- **Additional Data Needed:** Offline resource usage surveys and advisor check-in feedback to distinguish true disengagement from intentional offline study habits.

---

### Recommendation 3: Universal Access to Academic Onboarding Support

- **Empirical Evidence:** An exploratory Mann-Whitney U comparison of relative `date_registration` (negative values are days before course start, Day 0; N=22,410 retained and N=4,718 withdrawn) found a distributional difference ($p = 1.72e-09$). The withdrawn group registered earlier at the median (-58 vs -52 days relative to Day 0; equivalent lead times 58 vs 52 days). `registration_lead_time_days` is `-date_registration`, so larger positive values mean earlier registration. The p-value does not determine direction; this small observational difference does not justify targeting either group or establish a causal effect or registration cutoff.
- **Affected Target Cohort:** All students may access onboarding and academic scaffolding; offer additional help through student request or advisor review, not registration timing.
- **Recommended Action:** Provide an optional 2-week onboarding pathway with LMS orientation, study schedule templates, and peer-mentor pairing during Weeks 1-3.
- **Uncertainties & Alternative Explanations:** The retrospective registration comparison is small and observational. Its group medians do not establish that registration timing causes withdrawal, and they do not support prioritizing late registrants.
- **Additional Data Needed:** Prospectively evaluate onboarding uptake and outcomes before considering any risk-directed targeting.

---

### Recommendation 4: Faculty-Specific Caseload Allocation & Risk Calibration

- **Empirical Evidence:** Post-cutoff attrition rates vary substantially across academic disciplines: evaluated STEM modules average ~19.7% withdrawal across presentations, whereas Social Sciences and Arts modules average ~11.6%.
- **Affected Target Cohort:** Central Student Support Services and Faculty Advising Leads.
- **Recommended Action:** Disaggregate advisor caseload quotas. Instead of a flat cross-faculty allocation, assign advising FTE capacity proportionally based on course-level attrition history and calibrated risk volume.
- **Uncertainties & Alternative Explanations:** High attrition in STEM courses may be driven by steep early mathematical prerequisites rather than advising shortages.
- **Additional Data Needed:** Assessment-level prerequisite diagnostics and course-specific curriculum feedback.

---

## 2. Decision Support Workflow & Human Advisor Runbook

```
                         DAY-35 MODEL SCORING
                                   |
                                   v
             [ Calibrated Score & Proposed Priority Band ]
                                   |
          +------------------------+------------------------+
          |                                                 |
          v                                                 v
  Tier 1: Urgent (proposed >= 70%)                  Tier 2: Watch (proposed 40-70%)
          |                                                 |
          v                                                 v
  [ Proposed 3-Day Review Target ]                   [ Proposed 7-Day Check-In Target ]
          |                                                 |
          v                                                 v
  Example: Personal Advisor Review                 Example: Supportive Check-In Review
  - Phone / video call                              - Learning resource links
  - Coursework deadline flexibility                 - Tutor forum alert
  - Pastoral support referral                       - Advisor office hours invite
          |                                                 |
          +------------------------+------------------------+
                                   |
                                   v
                   ADVISOR QUALITATIVE DISCRETION
                     - Override / dismiss alert
                     - Record qualitative context
                     - Log student outcome
```

---

## 3. Strict Prohibitions & Harm Mitigation

1. **No Automated Sanctions:** Risk scores must NEVER be used to automate course deregistration, financial aid denial, or grade capping.
2. **No Algorithmic Labeling:** Advisors and automated templates must NEVER tell a student: *"You were flagged by an algorithm as high risk."*
3. **Mandatory Human Discretion:** Human judgment supersedes model recommendations in all circumstances.
4. **Observational Bounds:** Documented behavioral associations represent predictive triage cues rather than established causal drivers.
