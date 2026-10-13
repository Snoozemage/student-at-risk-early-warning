# Data Sources & Provenance Audit (OULAD)

**Project:** Smart Student Retention & Early Warning System  
**Phase:** Phase 0 — OULAD Workspace and Source Audit  
**Date:** 2026-10-13  
**Status:** Verified & Complete  

---

## 1. Overview & Official Provenance

The primary dataset used for this project is the **Open University Learning Analytics Dataset (OULAD)**, an internationally recognized open dataset capturing student demographic data, course registration history, assessment submissions, and clickstream interaction logs within a Virtual Learning Environment (VLE) at The Open University (UK).

### Official References
* **Official Dataset Portal:** [https://analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset)
* **Official Documentation & Schema:** [https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/)
* **Academic Publication:**  
  Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. **Scientific Data**, 4:170171. DOI: [10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171).
* **Dataset Owners & Donors:**
  * Original Owner: The Open University, Walton Hall, Milton Keynes, MK7 6AA, United Kingdom (Contact: Zdenek Zdrahal).
  * Donors: Jakub Kuzilek, Martin Hlosta, Zdenek Zdrahal (Knowledge Media Institute, Open University & CIIRC, Czech Technical University in Prague).
  * Date Received by Community: December 2015.

### Licensing & Attribution
* **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Permissions:** Free to share (copy and redistribute) and adapt (remix, transform, and build upon) for any purpose, including commercial, provided appropriate credit is given.
* **Preservation Requirement:** All downstream models, pipelines, and publications must maintain provenance attribution to the Open University authors.

---

## 2. Workspace File Inventory & Integrity

The dataset is located in the workspace under `data/raw/oalad/` (extracted from the raw archive). The directory contains 8 files: 7 core relational CSV tables and 1 metadata file (`OULAD.names`).

All files have been verified for integrity, format, and exact line counts using memory-aware streaming readers.

| Relative File Path | Format | Size (Bytes) | Size (MB) | Row Count | Column Count | SHA-256 Checksum |
|---|---|---:|---:|---:|---:|---|
| `data/raw/oalad/OULAD.names` | Text / Names | 12,278 | 0.01 MB | 247 lines | N/A | `e497bc5232f385ece30a7e964b0339bd9b74ba9b9083508b66628fed9dc23ebe` |
| `data/raw/oalad/courses.csv` | CSV | 526 | < 0.01 MB | 22 | 3 | `4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75` |
| `data/raw/oalad/assessments.csv` | CSV | 8,211 | 0.01 MB | 206 | 6 | `9ce92381e4a0ac457f8e251b2eb2c179ed51e023ab30d1d671a0268bd62316ba` |
| `data/raw/oalad/vle.csv` | CSV | 270,612 | 0.26 MB | 6,364 | 6 | `d7a146497edd0e47ef536144b58796c07b554002d3753e848b8e4cd8a56f4c38` |
| `data/raw/oalad/studentInfo.csv` | CSV | 3,462,763 | 3.30 MB | 32,593 | 12 | `815fbc3e2de29f79900bc63f0345a35db62c9146afaa1d86e7107b8b3beffc60` |
| `data/raw/oalad/studentRegistration.csv` | CSV | 1,132,550 | 1.08 MB | 32,593 | 5 | `bbc87a0de1fe2a9ec6decce1ca6c1a02cf17b54ad3bdcc673fc6413b41c61d47` |
| `data/raw/oalad/studentAssessment.csv` | CSV | 5,690,483 | 5.43 MB | 173,912 | 5 | `b1839e17618a5de36b6d189a242d3123f7939a56c8c9ccef0c17b9199c614a6b` |
| `data/raw/oalad/studentVle.csv` | CSV | 453,836,331 | 432.81 MB | 10,655,280 | 6 | `52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0` |

*Note on file paths:* In the workspace, the extracted directory is named `oalad`. `.gitignore` rule `data/raw/` ensures no raw files are tracked in Git.

---

## 3. High-Level Dataset Characteristics

1. **Modules & Presentations:**  
   Covers 7 distinct modules (`AAA` through `GGG`) across 4 academic presentations: `2013B`, `2013J`, `2014B`, and `2014J` (22 course-presentations in total).
2. **Student Population:**  
   32,593 module-presentation registrations representing 28,785 unique students. 3,808 students enrolled in more than one module-presentation.
3. **Assessment Activity:**  
   206 assessment items defined (TMA, CMA, Exam); 173,912 student submission records.
4. **VLE Interactions:**  
   6,364 virtual learning resources; 10,655,280 daily student-resource interaction logs totaling 39,605,099 clicks.

---

## 4. Known Dataset Limitations & Peculiarities

1. **Relative Date Representation:**  
   Dates are recorded as integer offsets relative to the official start of the module-presentation (day 0). Negative dates indicate events before the start (e.g. registration at -30 days or pre-course VLE access at -25 days).
2. **Presentation Length Variation:**  
   * February (`B`) presentations have lengths ranging from 234 to 241 days (median ~240 days).
   * October (`J`) presentations have lengths ranging from 261 to 269 days (median ~268 days).
   Cross-presentation models must account for these structural differences.
3. **Missing Exam Submissions:**  
   Of the 24 exams in `assessments.csv`, 18 have no student submission records in `studentAssessment.csv`. In OULAD, exams were typically evaluated outside the electronic VLE submission system, except for modules `CCC` and `DDD`.
4. **Assessment Score Latency:**  
   While `date_submitted` is recorded, the exact date when tutor grading was returned to the student is not recorded. Graded scores cannot be assumed available immediately upon submission.
5. **VLE Aggregation Level:**  
   Interactions in `studentVle.csv` are aggregated per student, per site, per day (`sum_click`). Precise timestamps, session durations, and dwell times are not available.
6. **Data Anomaly in Registration / Results:**  
   * 93 students with `final_result = Withdrawn` have missing `date_unregistration`.
   * 9 students with `final_result = Fail` have a recorded `date_unregistration` (8 with date $\le 0$).
   * 45 students have missing `date_registration`.
   * 1,111 student records have missing Index of Multiple Deprivation (`imd_band`).
   All anomalies must be addressed in Phase 1 data contract policies.
