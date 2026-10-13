"""Temporal properties and chronology validation for OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def validate_source_temporal_properties(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Validate temporal ranges, chronology, and date completeness across OULAD tables.

    Checks:
    - courses: module_presentation_length must be positive (> 0).
    - student_registration: missing registration dates (warning); unregistration before registration (error).
    - assessments: missing assessment deadlines (warning, typically final exams).
    - vle: week_to >= week_from when both are populated (error if inverted).
    - student_assessment: missing date_submitted (error).
    - student_vle: missing interaction dates (error); extreme anomalous interaction dates (warning).

    Args:
        tables: Dictionary of normalized or raw table DataFrames.

    Returns:
        Structured dictionary of temporal validation findings with severity, counts, and examples.
    """
    fatal_violations: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    # 1. Courses presentation length check
    if "courses" in tables:
        courses_df = tables["courses"]
        if "module_presentation_length" in courses_df.columns:
            lengths = pd.to_numeric(courses_df["module_presentation_length"], errors="coerce")
            invalid_len_mask = lengths.isna() | (lengths <= 0)
            invalid_len_count = int(invalid_len_mask.sum())

            invalid_len_examples = []
            if invalid_len_count > 0:
                for _, r in courses_df[invalid_len_mask].head(5).iterrows():
                    invalid_len_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "module_presentation_length": r.get("module_presentation_length"),
                    })
                fatal_violations.append(
                    f"courses contains {invalid_len_count} record(s) with non-positive or null module_presentation_length."
                )

            checks.append({
                "table": "courses",
                "check_name": "course_length_positive",
                "severity": "ERROR",
                "count": invalid_len_count,
                "is_valid": invalid_len_count == 0,
                "examples": invalid_len_examples,
                "rationale": "Module presentation length must be a strictly positive integer (> 0 days).",
            })

    # 2. Student registration dates check
    if "student_registration" in tables:
        reg_df = tables["student_registration"]

        # Missing registration dates
        if "date_registration" in reg_df.columns:
            missing_reg_mask = reg_df["date_registration"].isna()
            missing_reg_count = int(missing_reg_mask.sum())

            missing_reg_examples = []
            if missing_reg_count > 0:
                for _, r in reg_df[missing_reg_mask].head(5).iterrows():
                    missing_reg_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                    })
                warnings.append(
                    f"student_registration contains {missing_reg_count} record(s) with unrecorded date_registration."
                )

            checks.append({
                "table": "student_registration",
                "check_name": "missing_registration_dates",
                "severity": "WARNING",
                "count": missing_reg_count,
                "is_valid": True,  # Non-fatal source property
                "examples": missing_reg_examples,
                "rationale": "In OULAD, a small cohort of students (45 records) have unrecorded registration dates.",
            })

        # Unregistration before registration
        if "date_registration" in reg_df.columns and "date_unregistration" in reg_df.columns:
            both_mask = reg_df["date_registration"].notna() & reg_df["date_unregistration"].notna()
            reg_dates = pd.to_numeric(reg_df.loc[both_mask, "date_registration"], errors="coerce")
            unreg_dates = pd.to_numeric(reg_df.loc[both_mask, "date_unregistration"], errors="coerce")

            inverted_mask = unreg_dates < reg_dates
            inverted_count = int(inverted_mask.sum())

            inverted_examples = []
            if inverted_count > 0:
                bad_sub = reg_df.loc[both_mask][inverted_mask].head(5)
                for _, r in bad_sub.iterrows():
                    inverted_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                        "date_registration": r.get("date_registration"),
                        "date_unregistration": r.get("date_unregistration"),
                    })
                fatal_violations.append(
                    f"student_registration contains {inverted_count} record(s) where date_unregistration < date_registration."
                )

            checks.append({
                "table": "student_registration",
                "check_name": "registration_chronology_valid",
                "severity": "ERROR",
                "count": inverted_count,
                "is_valid": inverted_count == 0,
                "examples": inverted_examples,
                "rationale": "A student cannot unregister before registering.",
            })

    # 3. Assessments deadline date check
    if "assessments" in tables:
        ass_df = tables["assessments"]
        if "date" in ass_df.columns:
            missing_ass_date_mask = ass_df["date"].isna()
            missing_ass_date_count = int(missing_ass_date_mask.sum())

            missing_ass_date_examples = []
            if missing_ass_date_count > 0:
                for _, r in ass_df[missing_ass_date_mask].head(5).iterrows():
                    missing_ass_date_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "id_assessment": int(r["id_assessment"]) if pd.notna(r.get("id_assessment")) else None,
                        "assessment_type": str(r.get("assessment_type", "")),
                    })
                warnings.append(
                    f"assessments contains {missing_ass_date_count} assessment(s) with missing deadline dates (typically final exams)."
                )

            checks.append({
                "table": "assessments",
                "check_name": "missing_assessment_deadlines",
                "severity": "WARNING",
                "count": missing_ass_date_count,
                "is_valid": True,  # Non-fatal source property
                "examples": missing_ass_date_examples,
                "rationale": "Final examination assessments in OULAD often lack pre-scheduled relative submission deadlines.",
            })

    # 4. VLE resource week window check
    if "vle" in tables:
        vle_df = tables["vle"]
        if "week_from" in vle_df.columns and "week_to" in vle_df.columns:
            both_weeks = vle_df[vle_df["week_from"].notna() & vle_df["week_to"].notna()].copy()
            wf = pd.to_numeric(both_weeks["week_from"], errors="coerce")
            wt = pd.to_numeric(both_weeks["week_to"], errors="coerce")

            inverted_weeks_mask = wt < wf
            inverted_weeks_count = int(inverted_weeks_mask.sum())

            inverted_weeks_examples = []
            if inverted_weeks_count > 0:
                for _, r in both_weeks[inverted_weeks_mask].head(5).iterrows():
                    inverted_weeks_examples.append({
                        "id_site": int(r["id_site"]) if pd.notna(r.get("id_site")) else None,
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "week_from": r.get("week_from"),
                        "week_to": r.get("week_to"),
                    })
                fatal_violations.append(
                    f"vle contains {inverted_weeks_count} resource(s) where week_to < week_from (inverted window)."
                )

            checks.append({
                "table": "vle",
                "check_name": "vle_week_window_consistent",
                "severity": "ERROR",
                "count": inverted_weeks_count,
                "is_valid": inverted_weeks_count == 0,
                "examples": inverted_weeks_examples,
                "rationale": "Resource end availability week (week_to) must be greater than or equal to start week (week_from).",
            })

    # 5. Student assessment submission date check
    if "student_assessment" in tables:
        sa_df = tables["student_assessment"]
        if "date_submitted" in sa_df.columns:
            missing_sub_mask = sa_df["date_submitted"].isna()
            missing_sub_count = int(missing_sub_mask.sum())

            missing_sub_examples = []
            if missing_sub_count > 0:
                for _, r in sa_df[missing_sub_mask].head(5).iterrows():
                    missing_sub_examples.append({
                        "id_assessment": int(r.get("id_assessment")) if pd.notna(r.get("id_assessment")) else None,
                        "id_student": int(r.get("id_student")) if pd.notna(r.get("id_student")) else None,
                    })
                fatal_violations.append(
                    f"student_assessment contains {missing_sub_count} record(s) with missing date_submitted."
                )

            checks.append({
                "table": "student_assessment",
                "check_name": "submission_date_present",
                "severity": "ERROR",
                "count": missing_sub_count,
                "is_valid": missing_sub_count == 0,
                "examples": missing_sub_examples,
                "rationale": "Submitted assessment records must have a recorded submission timestamp.",
            })

    # 6. Student VLE interaction date checks
    if "student_vle" in tables:
        sv_df = tables["student_vle"]
        if "date" in sv_df.columns:
            # Missing / null dates (fatal error per non-nullable schema contract)
            null_date_mask = sv_df["date"].isna()
            null_date_count = int(null_date_mask.sum())

            null_date_examples = []
            if null_date_count > 0:
                for _, r in sv_df[null_date_mask].head(5).iterrows():
                    null_date_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                        "id_site": int(r["id_site"]) if pd.notna(r.get("id_site")) else None,
                        "date": r.get("date"),
                    })
                fatal_violations.append(
                    f"student_vle contains {null_date_count} record(s) with missing interaction date."
                )

            checks.append({
                "table": "student_vle",
                "check_name": "interaction_date_present",
                "severity": "ERROR",
                "count": null_date_count,
                "is_valid": null_date_count == 0,
                "examples": null_date_examples,
                "rationale": "Student VLE interaction date is non-nullable; valid timestamps are required for feature windowing.",
            })

            # Check for extreme interaction dates (documented pre-course window and presentation length bounds)
            # Pre-course VLE access window starts at Day -25 per OULAD documentation.
            # Presentation end is bounded by module_presentation_length from courses (or max 269 days).
            numeric_dates = pd.to_numeric(sv_df["date"], errors="coerce")
            valid_dates_mask = numeric_dates.notna()

            # Early bound: date < -25
            early_mask = valid_dates_mask & (numeric_dates < -25)
            early_count = int(early_mask.sum())

            early_examples = []
            if early_count > 0:
                for _, r in sv_df[early_mask].head(5).iterrows():
                    early_examples.append({
                        "code_module": str(r.get("code_module", "")),
                        "code_presentation": str(r.get("code_presentation", "")),
                        "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                        "id_site": int(r["id_site"]) if pd.notna(r.get("id_site")) else None,
                        "date": r.get("date"),
                    })
                warnings.append(
                    f"student_vle contains {early_count} interaction(s) occurring before Day -25 (earliest documented pre-course access)."
                )

            checks.append({
                "table": "student_vle",
                "check_name": "precourse_interaction_dates_plausible",
                "severity": "WARNING",
                "count": early_count,
                "is_valid": True,  # Non-fatal audit property
                "examples": early_examples,
                "rationale": "Documented pre-course VLE access window begins at Day -25 relative to presentation start.",
            })

            # Late bound: date > presentation_length (or 269 if courses not provided)
            late_count = 0
            late_examples = []
            if "courses" in tables and "module_presentation_length" in tables["courses"].columns:
                courses_df = tables["courses"]
                course_lengths = {
                    f"{r.code_module}_{r.code_presentation}": r.module_presentation_length
                    for r in courses_df.itertuples()
                }
                key_series = sv_df["code_module"].astype(str) + "_" + sv_df["code_presentation"].astype(str)
                pres_lengths = key_series.map(course_lengths)
                max_bound = pres_lengths.fillna(269)
                late_mask = valid_dates_mask & (numeric_dates > max_bound)
                late_count = int(late_mask.sum())
                if late_count > 0:
                    for _, r in sv_df[late_mask].head(5).iterrows():
                        late_examples.append({
                            "code_module": str(r.get("code_module", "")),
                            "code_presentation": str(r.get("code_presentation", "")),
                            "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                            "id_site": int(r["id_site"]) if pd.notna(r.get("id_site")) else None,
                            "date": r.get("date"),
                            "module_presentation_length": course_lengths.get(
                                f"{r.get('code_module')}_{r.get('code_presentation')}"
                            ),
                        })
            else:
                late_mask = valid_dates_mask & (numeric_dates > 269)
                late_count = int(late_mask.sum())
                if late_count > 0:
                    for _, r in sv_df[late_mask].head(5).iterrows():
                        late_examples.append({
                            "code_module": str(r.get("code_module", "")),
                            "code_presentation": str(r.get("code_presentation", "")),
                            "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                            "id_site": int(r["id_site"]) if pd.notna(r.get("id_site")) else None,
                            "date": r.get("date"),
                        })

            if late_count > 0:
                warnings.append(
                    f"student_vle contains {late_count} interaction(s) occurring after module presentation completion."
                )

            checks.append({
                "table": "student_vle",
                "check_name": "postcourse_interaction_dates_plausible",
                "severity": "WARNING",
                "count": late_count,
                "is_valid": True,  # Non-fatal audit property
                "examples": late_examples,
                "rationale": "VLE interactions recorded after module presentation length exceed formal course duration.",
            })

    total_fatal_errors = len(fatal_violations)
    total_warnings = len(warnings)

    return {
        "is_valid": total_fatal_errors == 0,
        "total_fatal_errors": total_fatal_errors,
        "total_warnings": total_warnings,
        "checks": checks,
        "violations": fatal_violations,
        "warnings": warnings,
    }
