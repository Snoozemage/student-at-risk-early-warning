# Data Dictionary & Schema Specification (OULAD)

**Project:** Smart Student Retention & Early Warning System  
**Phase:** Phase 0 — OULAD Workspace and Source Audit  
**Date:** 2026-10-13  
**Status:** Verified against Workspace CSVs & Official Documentation  

---

## 1. Overview & Conventions

* **Null Representation in Raw Data:** Missing or unrecorded values in the raw CSVs are represented by the literal character `?` or empty strings.
* **Date Encoding:** All date attributes are integer offsets in days measured relative to the official course start date (Day `0`). Negative values indicate occurrences prior to the start of the course presentation (e.g. Day `-30` indicates 30 days prior to course start).
* **Primary Prediction Unit:** The core entity for student retention prediction is the composite grain `(code_module, code_presentation, id_student)`.

---

## 2. Table Specifications

### 2.1. `courses.csv`

**Description:** Contains metadata on course modules and their specific presentations, including course duration.  
**Observed Rows:** 22  
**Observed Columns:** 3  
**Primary Key Candidate:** Composite `(code_module, code_presentation)` (22 unique rows, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `code_module` | String (Nominal) | Identification code of the module | 7 unique values: `AAA`, `BBB`, `CCC`, `DDD`, `EEE`, `FFF`, `GGG` | 0 (0%) | Foreign key to all other tables. |
| `code_presentation` | String (Nominal) | Code of the academic presentation | 4 unique values: `2013B`, `2013J`, `2014B`, `2014J` | 0 (0%) | `B` = Feb start, `J` = Oct start. |
| `module_presentation_length` | Integer | Length of the module presentation in days | Range: 234 to 269 days (B: 234–241, J: 261–269) | 0 (0%) | Systematic length gap between B and J cohorts. |

---

### 2.2. `assessments.csv`

**Description:** Information regarding all assessments defined for each module presentation.  
**Observed Rows:** 206  
**Observed Columns:** 6  
**Primary Key Candidate:** `id_assessment` (206 unique values, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `code_module` | String (Nominal) | Module identifier | 7 modules | 0 (0%) | Composite FK with `code_presentation`. |
| `code_presentation` | String (Nominal) | Presentation identifier | 4 presentations | 0 (0%) | Composite FK with `code_module`. |
| `id_assessment` | Integer | Unique identifier of the assessment | 206 unique integers | 0 (0%) | Primary key. Referenced by `studentAssessment.csv`. |
| `assessment_type` | String (Nominal) | Type of assessment | `TMA`: 106, `CMA`: 76, `Exam`: 24 | 0 (0%) | TMA = Tutor Marked, CMA = Computer Marked. |
| `date` | Integer / Null | Deadline for submission (days since day 0) | Range: 12 to 261; 22 due $\le$ day 35 | 11 (5.34%) | 11 missing dates (`?`) correspond to exams. |
| `weight` | Float | Weight of assessment in overall grade (%) | Range: 0.0 to 100.0 | 0 (0%) | Exams typically weighted 100%; TMAs/CMAs sum to 100% (or 0% for GGG). |

---

### 2.3. `vle.csv`

**Description:** Catalog of learning materials and pages hosted in the Virtual Learning Environment.  
**Observed Rows:** 6,364  
**Observed Columns:** 6  
**Primary Key Candidate:** `id_site` (6,364 unique values, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `id_site` | Integer | Unique identifier of the VLE material | 6,364 unique integers | 0 (0%) | Primary key. Referenced by `studentVle.csv`. |
| `code_module` | String (Nominal) | Module identifier | 7 modules | 0 (0%) | Module hosting the site. |
| `code_presentation` | String (Nominal) | Presentation identifier | 4 presentations | 0 (0%) | Presentation hosting the site. |
| `activity_type` | String (Nominal) | Pedagogical role of the resource | 20 distinct types (e.g. `resource`: 2,660, `subpage`: 1,055, `oucontent`: 996, `url`: 886, `forumng`: 194, `quiz`: 127) | 0 (0%) | High feature engineering utility for engagement profiling. |
| `week_from` | Integer / Null | Planned study week start | Observed range: 0 to 29 | 5,243 (82.38%) | High missingness; indicates unstructured access timing for most items. |
| `week_to` | Integer / Null | Planned study week end | Observed range: 0 to 29 | 5,243 (82.38%) | High missingness; aligns with `week_from`. |

---

### 2.4. `studentInfo.csv`

**Description:** Student demographic attributes, prior qualifications, and final course outcome.  
**Observed Rows:** 32,593  
**Observed Columns:** 12  
**Primary Key Candidate:** Composite `(code_module, code_presentation, id_student)` (32,593 unique, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `code_module` | String (Nominal) | Module identifier | 7 modules | 0 (0%) | Composite component of student-presentation grain. |
| `code_presentation` | String (Nominal) | Presentation identifier | 4 presentations | 0 (0%) | Composite component of student-presentation grain. |
| `id_student` | Integer | Unique identifier for the individual learner | 28,785 unique IDs | 0 (0%) | Learners may take multiple courses. |
| `gender` | String (Nominal) | Student gender | `M`: 17,875 (54.8%), `F`: 14,718 (45.2%) | 0 (0%) | Protected attribute; for fairness monitoring only. |
| `region` | String (Nominal) | Geographic region where student resided | 13 UK geographic regions | 0 (0%) | Highest: East Anglian (3,340), Scotland (3,479). |
| `highest_education` | String (Ordinal) | Highest education level at course entry | `No Formal quals` (347), `Lower Than A Level` (13,158), `A Level or Equivalent` (14,045), `HE Qualification` (4,730), `Post Graduate Qualification` (313) | 0 (0%) | Clean educational hierarchy. |
| `imd_band` | String (Ordinal) | Index of Multiple Deprivation decile band | 10 deciles: `0-10%` to `90-100%` | 1,111 (3.41%) | Missing encoded as `?`. Represents socio-economic deprivation. |
| `age_band` | String (Ordinal) | Age group band | `0-35`: 22,944 (70.4%), `35-55`: 9,433 (28.9%), `55<=`: 216 (0.7%) | 0 (0%) | Protected attribute; evaluate for disparate impact. |
| `num_of_prev_attempts` | Integer | Previous attempts on this module | Range: 0 to 6 (0: 28,421, 1: 3,299, 2+: 873) | 0 (0%) | High relevance for repeat-risk tracking. |
| `studied_credits` | Integer | Total credits studied concurrently | Range: 30 to 655 credits (Median: 60) | 0 (0%) | Proxy for academic workload. |
| `disability` | String (Nominal) | Declared disability status | `N`: 29,429 (90.3%), `Y`: 3,164 (9.7%) | 0 (0%) | Protected attribute; fairness evaluation required. |
| `final_result` | String (Nominal) | Final course outcome | `Pass`: 12,361 (37.9%), `Withdrawn`: 10,156 (31.2%), `Fail`: 7,052 (21.6%), `Distinction`: 3,024 (9.3%) | 0 (0%) | **Target Variable**. Must NEVER be used as a feature. |

---

### 2.5. `studentRegistration.csv`

**Description:** Records of student module registration and withdrawal dates.  
**Observed Rows:** 32,593  
**Observed Columns:** 5  
**Primary Key Candidate:** Composite `(code_module, code_presentation, id_student)` (32,593 unique, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `code_module` | String (Nominal) | Module identifier | 7 modules | 0 (0%) | Composite join key with `studentInfo`. |
| `code_presentation` | String (Nominal) | Presentation identifier | 4 presentations | 0 (0%) | Composite join key with `studentInfo`. |
| `id_student` | Integer | Learner identifier | 28,785 unique IDs | 0 (0%) | Composite join key with `studentInfo`. |
| `date_registration` | Integer / Null | Relative date of registration | Range: -322 to 167 days | 45 (0.14%) | Missing (`?`) in 45 rows. Negative indicates registration before day 0. |
| `date_unregistration` | Integer / Null | Relative date of unregistration (withdrawal) | Range: -365 to 444 days | 22,521 (69.1%) | Absent (`?`) for students completing course. **Temporal Leakage Risk**: Never use as predictor! |

---

### 2.6. `studentAssessment.csv`

**Description:** Individual assessment submission dates, banking status, and awarded scores.  
**Observed Rows:** 173,912  
**Observed Columns:** 5  
**Primary Key Candidate:** Composite `(id_assessment, id_student)` (173,912 unique rows, 0 duplicates).

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `id_assessment` | Integer | Assessment identifier | 188 distinct IDs | 0 (0%) | Foreign key to `assessments.csv`. (18 exams unrepresented). |
| `id_student` | Integer | Student identifier | 23,369 unique students | 0 (0%) | Foreign key to `studentInfo.csv`. |
| `date_submitted` | Integer | Day of submission relative to day 0 | Range: -11 to 608 days | 0 (0%) | Negative indicates pre-term early submission. |
| `is_banked` | Binary (0/1) | Result carried over from prior attempt | `0`: 172,003 (98.9%), `1`: 1,909 (1.1%) | 0 (0%) | Transferred grade from previous presentation. |
| `score` | Float / Null | Awarded score (0 to 100) | Range: 0.0 to 100.0 (Mean: 75.8; <40 Fail: 7,578) | 173 (0.10%) | 173 missing (`?`). Not known at day 35 for post-cutoff assessments. |

---

### 2.7. `studentVle.csv`

**Description:** Log of daily student interactions with VLE resources.  
**Observed Rows:** 10,655,280  
**Observed Columns:** 6  
**Primary Key Candidate:** Natural composite `(code_module, code_presentation, id_student, id_site, date)`.

| Column Name | Observed Type | Official Description | Value Domain / Observed Range | Missing Count | Notes / Ambiguities |
|---|---|---|---|---|---|
| `code_module` | String (Nominal) | Module identifier | 7 modules | 0 (0%) | Foreign key component. |
| `code_presentation` | String (Nominal) | Presentation identifier | 4 presentations | 0 (0%) | Foreign key component. |
| `id_student` | Integer | Student identifier | 26,074 unique students | 0 (0%) | Foreign key component. |
| `id_site` | Integer | VLE resource identifier | 6,268 distinct sites | 0 (0%) | Foreign key to `vle.csv`. (96 sites have 0 logs). |
| `date` | Integer | Relative day of interaction | Range: -25 to 269 days | 0 (0%) | 688,988 interactions occur before day 0; 3,253,906 $\le$ day 35. |
| `sum_click` | Integer | Number of clicks on that resource on that day | Range: 1 to 6,977 (Total clicks: 39,605,099) | 0 (0%) | Daily intensity of resource engagement. |

---

## 3. Data Quality & Ambiguity Summary

1. **Missing Data Encoding:** All raw tables consistently encode NULLs as literal `?`. Imputation/cleaning must convert `?` to explicit Python `None` / `pd.NA`.
2. **`studentInfo` vs `studentRegistration` 1-to-1 Match:** Exactly 32,593 records exist in both tables with zero unmatched keys and zero duplicate keys.
3. **93 Withdrawn Students without Unregistration Date:** 93 students have `final_result = Withdrawn` but `date_unregistration` is `?`. A documented policy is needed in Phase 1 (whether to exclude from strict cutoff evaluation or treat as late withdrawals).
4. **9 Fail Students with Unregistration Date:** 9 records have `final_result = Fail` yet record a withdrawal date (8 of which occurred $\le 0$). Must be handled explicitly in Phase 1 cohort filtering.
5. **45 Missing Registration Dates:** 45 records have no `date_registration`. Since all other records registered before or around day 0, Phase 1 should adopt an explicit handling policy (e.g. assume default registration before course start or drop).
