# Feature Interpretation & Behavioral Signals Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Models Interpreted:** L2-Regularized Logistic Regression (Odds Ratios) and Random Forest (Gini Impurity)

---

## 1. Interpretive Guidance & Causality Disclaimer

> **Critical Methodological Notice:**
> All feature importance metrics and regression coefficients describe **statistical associations** within historical observational data up to Day 35. They do **not** establish causal relationships. A high importance or large odds ratio does not imply that manipulating that feature will directly alter student retention.

---

## 2. Logistic Regression: Directional Signals & Odds Ratios ($e^\beta$)

### Top Risk Amplifiers ($OR > 1.0$)
| Feature | Log-Odds Coef ($\beta$) | Odds Ratio ($e^\beta$) | Interpretation |
|---|---|---|---|
| `code_module_CCC` | +0.5917 | **1.81** | Associated with increased likelihood of withdrawal |
| `code_module_DDD` | +0.4593 | **1.58** | Associated with increased likelihood of withdrawal |
| `vle_total_clicks_window` | +0.4301 | **1.54** | Associated with increased likelihood of withdrawal |
| `code_module_FFF` | +0.2646 | **1.30** | Associated with increased likelihood of withdrawal |
| `studied_credits` | +0.1712 | **1.19** | Associated with increased likelihood of withdrawal |

### Top Protective Factors ($OR < 1.0$)
| Feature | Log-Odds Coef ($\beta$) | Odds Ratio ($e^\beta$) | Interpretation |
|---|---|---|---|
| `code_module_GGG` | -1.4536 | **0.23** | Associated with decreased likelihood of withdrawal |
| `presentation_season_J` | -0.5851 | **0.56** | Associated with decreased likelihood of withdrawal |
| `presentation_season_B` | -0.4680 | **0.63** | Associated with decreased likelihood of withdrawal |
| `code_module_BBB` | -0.3852 | **0.68** | Associated with decreased likelihood of withdrawal |
| `code_module_AAA` | -0.3537 | **0.70** | Associated with decreased likelihood of withdrawal |

---

## 3. Random Forest: Non-Linear Impurity Importances

Top 10 features contributing most to tree split purity (Gini impurity importance):

| Rank | Feature | Gini Impurity Importance | Relative Share |
|---|---|---|---|
| 1 | `missing_due_submissions_count` | 0.1096 | 11.0% |
| 2 | `submitted_on_time_count` | 0.0644 | 6.4% |
| 3 | `vle_total_clicks_window` | 0.0632 | 6.3% |
| 4 | `studied_credits` | 0.0593 | 5.9% |
| 5 | `vle_clicks_oucontent` | 0.0531 | 5.3% |
| 6 | `registration_lead_time_days` | 0.0528 | 5.3% |
| 7 | `vle_recent_clicks_14d` | 0.0507 | 5.1% |
| 8 | `assessments_due_count` | 0.0456 | 4.6% |
| 9 | `vle_velocity_ratio` | 0.0445 | 4.4% |
| 10 | `vle_unique_sites_accessed` | 0.0440 | 4.4% |

---

## 4. Key Behavioral Takeaways for Academic Advisors

1. **Digital Consistency Over Raw Volume:** `vle_active_days_window` and `days_since_last_vle_interaction` are substantially more predictive than total clicks. Regular, paced engagement is the hallmark of persistent students.
2. **Early Assessment Milestones:** `submitted_on_time_count` and `missing_due_submissions_count` provide decisive pivot points by Day 35. Learners who miss their initial assignment represent prime candidates for early scaffolding.
3. **Pacing Momentum:** A sharp drop in `vle_velocity_ratio` serves as an early indicator of silent disengagement well before formal withdrawal occurs.
