"""Cohort construction and deterministic eligibility exclusion pipeline."""
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.cleaning.missing_values import audit_missing_values
from student_retention.cleaning.normalization import normalize_table
from student_retention.config import get_project_config, get_quality_rules_config
from student_retention.labels.withdrawal_target import (
    assert_no_target_leakage,
    create_withdrawal_target,
    parse_and_validate_temporal_date,
    validate_canonical_keys,
    validate_final_result,
    validate_withdrawal_target,
)
from student_retention.logging_config import get_logger
from student_retention.paths import (
    DATA_RAW_DIR,
    INTERIM_NORMALIZED_DIR,
    PROCESSED_COHORT_DIR,
    QUALITY_REPORTS_DIR,
)
from student_retention.validation.relationship_checks import validate_relationship
from student_retention.validation.schema_checks import (
    validate_courses_reference_table,
    validate_table_schema,
)

logger = get_logger(__name__)

CONTRADICTORY_RECORD_KEY = ("BBB", "2013J", 554243)


class CohortInvariantError(ValueError, RuntimeError):
    """Raised when any cohort construction invariant or reconciliation check is violated."""


class CohortBuilder:
    """Builds the eligible student cohort at prediction cutoff under Policy C or Policy B."""

    def __init__(
        self,
        cutoff_day: int | None = None,
        policy: str = "policy_c",
        data_dir: Path | str | None = None,
        courses_df: pd.DataFrame | None = None,
    ):
        project_cfg = get_project_config()
        self.cutoff_day = (
            cutoff_day
            if cutoff_day is not None
            else project_cfg.get("prediction", {}).get("prediction_cutoff_relative_days", 35)
        )
        self.policy = policy.lower()
        if self.policy not in {"policy_c", "policy_b"}:
            raise ValueError(f"Unknown cohort policy '{self.policy}'. Supported policies: 'policy_c', 'policy_b'")

        self.data_dir = Path(data_dir) if data_dir else INTERIM_NORMALIZED_DIR
        self.quality_rules_cfg = get_quality_rules_config()
        if courses_df is not None:
            validate_courses_reference_table(courses_df, error_cls=CohortInvariantError)
        self.courses_df = courses_df

    def _load_courses(self) -> pd.DataFrame:
        """Load courses DataFrame from normalized parquet or raw CSV fallback, validating schema and keys."""
        if self.courses_df is not None:
            validate_courses_reference_table(self.courses_df, error_cls=CohortInvariantError)
            return self.courses_df

        courses_parquet = self.data_dir / "courses.parquet"
        if courses_parquet.exists():
            df = pd.read_parquet(courses_parquet)
            validate_courses_reference_table(df, error_cls=CohortInvariantError)
            return df

        courses_csv = self.data_dir / "courses.csv" if (self.data_dir / "courses.csv").exists() else DATA_RAW_DIR / "courses.csv"
        if courses_csv.exists():
            raw_courses = pd.read_csv(courses_csv, na_values=["?"], keep_default_na=False)
            audit_res = audit_missing_values("courses", raw_courses, quality_rules_cfg=self.quality_rules_cfg)
            if not audit_res["is_valid"]:
                raise CohortInvariantError(
                    f"Raw fallback courses missing-value/forbidden-marker violations: {audit_res['violations']}"
                )
            df = normalize_table("courses", raw_courses)
            validate_courses_reference_table(df, error_cls=CohortInvariantError)
            return df

        raise CohortInvariantError(
            f"Cannot validate referential integrity against courses: table not found in {self.data_dir} or {DATA_RAW_DIR}"
        )

    def load_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load student_info and student_registration from parquet or raw CSV.

        When falling back to raw CSV, explicitly audits missingness against quality_rules.yaml
        before normalization, normalizes types, validates schemas, and checks referential integrity
        against courses to enforce data contract equivalence with the normalized interim path.
        """
        parquet_info = self.data_dir / "student_info.parquet"
        parquet_reg = self.data_dir / "student_registration.parquet"

        if parquet_info.exists() and parquet_reg.exists():
            logger.info("Loading student_info and student_registration from normalized parquet: %s", self.data_dir)
            info_df = pd.read_parquet(parquet_info)
            reg_df = pd.read_parquet(parquet_reg)
        else:
            logger.info("Normalized parquet not found at %s; loading from raw CSV: %s", self.data_dir, DATA_RAW_DIR)

            raw_info_path = self.data_dir / "studentInfo.csv" if (self.data_dir / "studentInfo.csv").exists() else DATA_RAW_DIR / "studentInfo.csv"
            raw_reg_path = self.data_dir / "studentRegistration.csv" if (self.data_dir / "studentRegistration.csv").exists() else DATA_RAW_DIR / "studentRegistration.csv"

            info_raw = pd.read_csv(raw_info_path, na_values=["?"], keep_default_na=False)
            reg_raw = pd.read_csv(raw_reg_path, na_values=["?"], keep_default_na=False)

            # 1. Audit missing values and forbidden markers BEFORE normalization can alter/erase them
            audit_info = audit_missing_values("student_info", info_raw, quality_rules_cfg=self.quality_rules_cfg)
            audit_reg = audit_missing_values("student_registration", reg_raw, quality_rules_cfg=self.quality_rules_cfg)
            if not audit_info["is_valid"]:
                raise CohortInvariantError(
                    f"Raw fallback student_info missing-value/forbidden-marker violations: {audit_info['violations']}"
                )
            if not audit_reg["is_valid"]:
                raise CohortInvariantError(
                    f"Raw fallback student_registration missing-value/forbidden-marker violations: {audit_reg['violations']}"
                )

            # 2. Normalize types into nullable extension dtypes
            info_df = normalize_table("student_info", info_raw)
            reg_df = normalize_table("student_registration", reg_raw)

            # 3. Validate schema contracts
            v_info = validate_table_schema("student_info", info_df)
            v_reg = validate_table_schema("student_registration", reg_df)
            if not v_info["is_valid"]:
                raise CohortInvariantError(f"Raw fallback student_info schema violations: {v_info['violations']}")
            if not v_reg["is_valid"]:
                raise CohortInvariantError(f"Raw fallback student_registration schema violations: {v_reg['violations']}")

            # 4. Validate referential integrity against courses
            courses_df = self._load_courses()
            rel_info = validate_relationship("courses", courses_df, "student_info", info_df, ["code_module", "code_presentation"], allow_orphans=False)
            rel_reg = validate_relationship("courses", courses_df, "student_registration", reg_df, ["code_module", "code_presentation"], allow_orphans=False)
            if not rel_info["is_valid"]:
                raise CohortInvariantError(f"Raw fallback student_info referential integrity violations: {rel_info['violations']}")
            if not rel_reg["is_valid"]:
                raise CohortInvariantError(f"Raw fallback student_registration referential integrity violations: {rel_reg['violations']}")

        return info_df, reg_df

    def build_cohort(
        self,
        student_info_df: pd.DataFrame,
        student_registration_df: pd.DataFrame,
        courses_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute deterministic eligibility exclusions and attach binary withdrawal target.

        Args:
            student_info_df: student_info DataFrame.
            student_registration_df: student_registration DataFrame.
            courses_df: Optional courses DataFrame to validate course referential integrity.

        Returns:
            Dictionary containing:
            - 'cohort_df': Filtered eligible cohort with schema fields.
            - 'exclusions_df': Excluded records with step, code, and reason.
            - 'audit_report': Complete audit summary and reconciliation.

        Raises:
            CohortInvariantError: If input tables contain missing canonical keys, duplicate keys,
                or mismatched key sets, or if any cohort grain, target completeness, class-count,
                referential integrity against courses, or source reconciliation invariant is violated.
        """
        grain = ["code_module", "code_presentation", "id_student"]

        # Reject missing/null/blank canonical keys and validate id_student as positive integer
        validate_canonical_keys(student_info_df, "student_info_df", CohortInvariantError)
        validate_canonical_keys(student_registration_df, "student_registration_df", CohortInvariantError)

        # Validate final_result presence, string-typing, non-nullability, and domain
        validate_final_result(student_info_df, "student_info_df", CohortInvariantError)

        # Validate required registration columns presence
        for reg_col in ["date_registration", "date_unregistration"]:
            if reg_col not in student_registration_df.columns:
                raise CohortInvariantError(
                    f"student_registration_df is missing required column '{reg_col}'."
                )

        # Validate referential integrity against courses
        ref_courses = courses_df if courses_df is not None else self._load_courses()
        validate_courses_reference_table(ref_courses, error_cls=CohortInvariantError)

        rel_info = validate_relationship(
            parent_name="courses",
            parent_df=ref_courses,
            child_name="student_info_df",
            child_df=student_info_df,
            join_keys=["code_module", "code_presentation"],
            allow_orphans=False,
        )
        if not rel_info["is_valid"]:
            raise CohortInvariantError(
                f"Referential integrity invariant violated: student_info_df references invalid courses: {rel_info['violations']}"
            )

        rel_reg = validate_relationship(
            parent_name="courses",
            parent_df=ref_courses,
            child_name="student_registration_df",
            child_df=student_registration_df,
            join_keys=["code_module", "code_presentation"],
            allow_orphans=False,
        )
        if not rel_reg["is_valid"]:
            raise CohortInvariantError(
                f"Referential integrity invariant violated: student_registration_df references invalid courses: {rel_reg['violations']}"
            )

        # Independently validate source-record counts before joining
        raw_info_count = len(student_info_df)
        raw_reg_count = len(student_registration_df)

        # Guard against duplicate canonical keys in input tables
        info_dups = int(student_info_df.duplicated(subset=grain).sum())
        if info_dups > 0:
            raise CohortInvariantError(
                f"student_info_df contains {info_dups} duplicate canonical keys on {grain}."
            )

        reg_dups = int(student_registration_df.duplicated(subset=grain).sum())
        if reg_dups > 0:
            raise CohortInvariantError(
                f"student_registration_df contains {reg_dups} duplicate canonical keys on {grain}."
            )

        # Explicitly verify that canonical key sets match exactly between tables before joining
        info_keys = set(zip(student_info_df["code_module"], student_info_df["code_presentation"], student_info_df["id_student"]))
        reg_keys = set(zip(student_registration_df["code_module"], student_registration_df["code_presentation"], student_registration_df["id_student"]))

        only_in_info = info_keys - reg_keys
        only_in_reg = reg_keys - info_keys
        if only_in_info or only_in_reg:
            err_parts = []
            if only_in_info:
                sample_info = sorted(only_in_info)[:3]
                err_parts.append(
                    f"{len(only_in_info)} keys in student_info missing from student_registration (sample: {sample_info})"
                )
            if only_in_reg:
                sample_reg = sorted(only_in_reg)[:3]
                err_parts.append(
                    f"{len(only_in_reg)} keys in student_registration missing from student_info (sample: {sample_reg})"
                )

            raise CohortInvariantError(
                f"Canonical key mismatch between student_info and student_registration on {grain}: {'; '.join(err_parts)}. "
                f"Input tables must possess identical canonical key sets."
            )

        if raw_info_count != raw_reg_count:
            raise CohortInvariantError(
                f"Source count mismatch: student_info has {raw_info_count} records but student_registration has {raw_reg_count} records."
            )

        # Define independent, validated source record count
        total_source_records = raw_info_count

        # Merge base demographic/result fields with registration timestamps with strict 1:1 validation
        merged = pd.merge(
            student_info_df,
            student_registration_df[grain + ["date_registration", "date_unregistration"]],
            on=grain,
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != total_source_records:
            raise CohortInvariantError(
                f"Source merge count mismatch: merged {len(merged)} rows, but independent source count is {total_source_records}."
            )

        # Convert and strictly validate temporal fields
        date_reg = parse_and_validate_temporal_date(
            merged["date_registration"],
            col_name="date_registration",
            table_name="student_registration",
            error_cls=CohortInvariantError,
        )
        date_unreg = parse_and_validate_temporal_date(
            merged["date_unregistration"],
            col_name="date_unregistration",
            table_name="student_registration",
            error_cls=CohortInvariantError,
        )
        merged["date_registration"] = date_reg
        merged["date_unregistration"] = date_unreg
        final_results = merged["final_result"].astype(str)

        # Initialize tracking series for mutually exclusive exclusions
        exclusion_step = pd.Series(index=merged.index, dtype="Int64", data=pd.NA)
        exclusion_code = pd.Series(index=merged.index, dtype="string", data=pd.NA)
        exclusion_reason = pd.Series(index=merged.index, dtype="string", data=pd.NA)

        # -------------------------------------------------------------------------
        # Precedence Step 1: Pre-cutoff unregistration (date_unregistration <= cutoff_day)
        # -------------------------------------------------------------------------
        step1_mask = date_unreg.notna() & (date_unreg <= self.cutoff_day)
        unassigned = exclusion_step.isna()
        apply_step1 = step1_mask & unassigned
        exclusion_step.loc[apply_step1] = 1
        exclusion_code.loc[apply_step1] = "EXCL_PRE_CUTOFF_UNREG"
        exclusion_reason.loc[apply_step1] = (
            f"Student unregistered on or before prediction cutoff (Day {self.cutoff_day}). Already left course."
        )

        # -------------------------------------------------------------------------
        # Precedence Step 2: Late registration (date_registration > cutoff_day)
        # -------------------------------------------------------------------------
        step2_mask = date_reg.notna() & (date_reg > self.cutoff_day)
        unassigned = exclusion_step.isna()
        apply_step2 = step2_mask & unassigned
        exclusion_step.loc[apply_step2] = 2
        exclusion_code.loc[apply_step2] = "EXCL_LATE_REGISTRATION"
        exclusion_reason.loc[apply_step2] = (
            f"Student registered after prediction cutoff (Day {self.cutoff_day}). Not yet enrolled at cutoff."
        )

        # -------------------------------------------------------------------------
        # Precedence Step 3: Missing registration date (date_registration is null)
        # Excluded in Policy C; retained in Policy B.
        # -------------------------------------------------------------------------
        if self.policy == "policy_c":
            step3_mask = date_reg.isna()
            unassigned = exclusion_step.isna()
            apply_step3 = step3_mask & unassigned
            exclusion_step.loc[apply_step3] = 3
            exclusion_code.loc[apply_step3] = "EXCL_MISSING_REGISTRATION_DATE"
            exclusion_reason.loc[apply_step3] = (
                f"Registration date is unrecorded. Enrollment presence on Day {self.cutoff_day} cannot be confirmed."
            )

        # -------------------------------------------------------------------------
        # Precedence Step 4: Withdrawn with missing unregistration date
        # (final_result == 'Withdrawn' and date_unregistration is null)
        # -------------------------------------------------------------------------
        step4_mask = (final_results == "Withdrawn") & date_unreg.isna()
        unassigned = exclusion_step.isna()
        apply_step4 = step4_mask & unassigned
        exclusion_step.loc[apply_step4] = 4
        exclusion_code.loc[apply_step4] = "EXCL_WITHDRAWN_MISSING_UNREG_DATE"
        exclusion_reason.loc[apply_step4] = (
            "Recorded as Withdrawn but withdrawal timing is unrecorded. Cannot verify post-cutoff status."
        )

        # -------------------------------------------------------------------------
        # Precedence Step 5: Contradictory outcome record (BBB / 2013J / 554243)
        # Excluded in Policy C; retained in Policy B.
        # -------------------------------------------------------------------------
        if self.policy == "policy_c":
            mod_b, pres_b, id_b = CONTRADICTORY_RECORD_KEY
            step5_mask = (
                (merged["code_module"] == mod_b)
                & (merged["code_presentation"] == pres_b)
                & (merged["id_student"] == id_b)
            )
            unassigned = exclusion_step.isna()
            apply_step5 = step5_mask & unassigned
            exclusion_step.loc[apply_step5] = 5
            exclusion_code.loc[apply_step5] = "EXCL_CONTRADICTORY_RECORD"
            exclusion_reason.loc[apply_step5] = (
                "Final result recorded as Fail but date_unregistration is Day 166. Contradictory outcome semantics."
            )

        # Partition into eligible cohort and excluded records
        is_excluded = exclusion_step.notna()
        is_eligible = ~is_excluded

        exclusions_df = merged[is_excluded].copy()
        exclusions_df["exclusion_step"] = exclusion_step[is_excluded]
        exclusions_df["exclusion_code"] = exclusion_code[is_excluded]
        exclusions_df["exclusion_reason"] = exclusion_reason[is_excluded]

        eligible_df = merged[is_eligible].copy()

        # Compute withdrawal target
        targets_df = create_withdrawal_target(
            student_info_df=eligible_df,
            student_registration_df=eligible_df,
            cutoff_day=self.cutoff_day,
        )

        # Explicitly validate target outputs using validate_withdrawal_target
        validate_withdrawal_target(targets_df, raise_on_invalid=True)

        # Guard target join against duplicate keys and row multiplication
        targets_dups = int(targets_df.duplicated(subset=grain).sum())
        if targets_dups > 0:
            raise CohortInvariantError(
                f"targets_df contains {targets_dups} duplicate canonical keys on {grain}."
            )

        pre_target_count = len(eligible_df)
        eligible_df = pd.merge(eligible_df, targets_df, on=grain, how="left", validate="one_to_one")
        if len(eligible_df) != pre_target_count:
            raise CohortInvariantError(
                f"Row multiplication detected during target merge: expected {pre_target_count} rows, got {len(eligible_df)}."
            )
        eligible_df["policy_version"] = self.policy

        # Reconcile counts and metrics
        eligible_count = len(eligible_df)
        total_excluded_count = len(exclusions_df)

        pos_count = int((eligible_df["withdrawn_after_cutoff"] == 1).sum())
        neg_count = int((eligible_df["withdrawn_after_cutoff"] == 0).sum())
        base_rate = round(pos_count / eligible_count * 100, 4) if eligible_count > 0 else 0.0

        # Detailed breakdown of exclusions by step
        exclusion_steps_summary = {}
        for step in [1, 2, 3, 4, 5]:
            step_mask = exclusions_df["exclusion_step"] == step
            step_count = int(step_mask.sum())
            if step_count > 0 or self.policy == "policy_c":
                code = exclusions_df.loc[step_mask, "exclusion_code"].iloc[0] if step_count > 0 else f"STEP_{step}"
                reason = exclusions_df.loc[step_mask, "exclusion_reason"].iloc[0] if step_count > 0 else "N/A"
                exclusion_steps_summary[f"step_{step}"] = {
                    "step": step,
                    "code": code,
                    "reason": reason,
                    "count": step_count,
                    "pct_of_raw": round(step_count / total_source_records * 100, 4),
                }

        neg_breakdown = eligible_df[eligible_df["withdrawn_after_cutoff"] == 0]["final_result"].value_counts().to_dict()

        # Enforce all runtime invariants (fails fast if any invariant is violated)
        self._assert_cohort_invariants(
            eligible_df=eligible_df,
            exclusions_df=exclusions_df,
            total_source_records=total_source_records,
            raw_info_count=raw_info_count,
            raw_reg_count=raw_reg_count,
            exclusion_steps_summary=exclusion_steps_summary,
            pos_count=pos_count,
            neg_count=neg_count,
            neg_breakdown=neg_breakdown,
        )

        audit_report = {
            "dataset_name": "Open University Learning Analytics Dataset (OULAD)",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "policy": self.policy,
            "prediction_cutoff_relative_days": self.cutoff_day,
            "reconciliation": {
                "raw_student_info_records": raw_info_count,
                "raw_student_registration_records": raw_reg_count,
                "total_source_records": total_source_records,
                "total_excluded_records": total_excluded_count,
                "pct_excluded": round(total_excluded_count / total_source_records * 100, 4),
                "eligible_cohort_records": eligible_count,
                "pct_eligible": round(eligible_count / total_source_records * 100, 4),
                "is_reconciled": (
                    total_source_records == (total_excluded_count + eligible_count)
                    and raw_info_count == raw_reg_count == total_source_records
                ),
            },
            "target_distribution": {
                "positive_cases": pos_count,
                "negative_cases": neg_count,
                "positive_base_rate_pct": base_rate,
                "negative_class_breakdown": {str(k): int(v) for k, v in neg_breakdown.items()},
            },
            "exclusion_precedence_hierarchy": exclusion_steps_summary,
        }

        logger.info(
            "Built cohort [%s]: %d eligible (%d pos, %d neg, base rate: %.2f%%), %d excluded.",
            self.policy,
            eligible_count,
            pos_count,
            neg_count,
            base_rate,
            total_excluded_count,
        )

        return {
            "cohort_df": eligible_df,
            "exclusions_df": exclusions_df,
            "audit_report": audit_report,
        }

    def _assert_cohort_invariants(
        self,
        eligible_df: pd.DataFrame,
        exclusions_df: pd.DataFrame,
        total_source_records: int,
        exclusion_steps_summary: dict[str, Any],
        pos_count: int,
        neg_count: int,
        neg_breakdown: dict[str, int],
        raw_info_count: int | None = None,
        raw_reg_count: int | None = None,
    ) -> None:
        """Enforce strict runtime invariants across cohort outputs."""
        grain = ["code_module", "code_presentation", "id_student"]

        if raw_info_count is None:
            raw_info_count = total_source_records
        if raw_reg_count is None:
            raw_reg_count = total_source_records

        # 1. Source Input Reconciliation Invariants
        if raw_info_count != total_source_records or raw_reg_count != total_source_records:
            raise CohortInvariantError(
                f"Source input reconciliation invariant violated: raw_info ({raw_info_count}) "
                f"or raw_reg ({raw_reg_count}) != total_source_records ({total_source_records})."
            )

        # 2. Cohort Grain Invariants
        eligible_grain_dups = int(eligible_df.duplicated(subset=grain).sum())
        if eligible_grain_dups > 0:
            raise CohortInvariantError(
                f"Cohort grain invariant violated: eligible cohort contains {eligible_grain_dups} duplicate keys on {grain}."
            )

        excl_grain_dups = int(exclusions_df.duplicated(subset=grain).sum())
        if excl_grain_dups > 0:
            raise CohortInvariantError(
                f"Exclusions grain invariant violated: exclusions dataset contains {excl_grain_dups} duplicate keys on {grain}."
            )

        eligible_keys = set(zip(eligible_df["code_module"], eligible_df["code_presentation"], eligible_df["id_student"]))
        excl_keys = set(zip(exclusions_df["code_module"], exclusions_df["code_presentation"], exclusions_df["id_student"]))
        overlap_keys = eligible_keys.intersection(excl_keys)
        if overlap_keys:
            raise CohortInvariantError(
                f"Cohort disjointness invariant violated: {len(overlap_keys)} keys exist in both eligible and excluded datasets."
            )

        # 3. Target Completeness Invariants
        na_target_count = int(eligible_df["withdrawn_after_cutoff"].isna().sum())
        if na_target_count > 0:
            raise CohortInvariantError(
                f"Target completeness invariant violated: {na_target_count} eligible cohort records have unassigned or null target values."
            )

        # Target must not be boolean and must be in {0, 1}
        if pd.api.types.is_bool_dtype(eligible_df["withdrawn_after_cutoff"].dtype):
            raise CohortInvariantError(
                "Target domain invariant violated: target column has boolean dtype; expected integer {0, 1}."
            )

        target_domain = set(eligible_df["withdrawn_after_cutoff"].dropna().unique())
        if not target_domain.issubset({0, 1}):
            raise CohortInvariantError(
                f"Target domain invariant violated: unexpected target values {target_domain - {0, 1}} found in eligible cohort."
            )

        # 4. Class-Count Reconciliation Invariants
        eligible_count = len(eligible_df)
        if (pos_count + neg_count) != eligible_count:
            raise CohortInvariantError(
                f"Class-count reconciliation invariant violated: positive ({pos_count}) + negative ({neg_count}) "
                f"!= total eligible cohort ({eligible_count})."
            )

        neg_breakdown_sum = sum(neg_breakdown.values())
        if neg_breakdown_sum != neg_count:
            raise CohortInvariantError(
                f"Negative class breakdown reconciliation violated: sum of negative outcomes ({neg_breakdown_sum}) "
                f"!= total negative count ({neg_count})."
            )

        if self.policy == "policy_c":
            pos_non_withdrawn = int(((eligible_df["withdrawn_after_cutoff"] == 1) & (eligible_df["final_result"] != "Withdrawn")).sum())
            if pos_non_withdrawn > 0:
                raise CohortInvariantError(
                    f"Positive class label invariant violated: {pos_non_withdrawn} positive records under Policy C have final_result != 'Withdrawn'."
                )

        # 5. Source/Output Reconciliation Invariants
        total_excluded_count = len(exclusions_df)
        if total_source_records != (eligible_count + total_excluded_count):
            raise CohortInvariantError(
                f"Source-to-output reconciliation invariant violated: total source records ({total_source_records}) "
                f"!= eligible ({eligible_count}) + excluded ({total_excluded_count})."
            )

        step_count_sum = sum(s["count"] for s in exclusion_steps_summary.values())
        if step_count_sum != total_excluded_count:
            raise CohortInvariantError(
                f"Exclusion hierarchy reconciliation invariant violated: sum of step counts ({step_count_sum}) "
                f"!= total excluded records ({total_excluded_count})."
            )

        if (
            exclusions_df["exclusion_step"].isna().any()
            or exclusions_df["exclusion_code"].isna().any()
            or exclusions_df["exclusion_reason"].isna().any()
        ):
            raise CohortInvariantError(
                "Exclusion metadata invariant violated: null values detected in exclusion_step, exclusion_code, or exclusion_reason."
            )

        # 6. Anti-Leakage Invariants
        # Verify no outcome columns leaked into non-metadata, non-target feature fields
        exempt_metadata_and_target = {
            "code_module",
            "code_presentation",
            "id_student",
            "withdrawn_after_cutoff",
            "target_withdrawn",
            "withdrawal_target",
            "date_registration",
            "date_unregistration",
            "final_result",
            "policy_version",
        }
        candidate_feature_cols = [c for c in eligible_df.columns if c not in exempt_metadata_and_target]
        assert_no_target_leakage(candidate_feature_cols)

        early_unregs = int((eligible_df["date_unregistration"].notna() & (eligible_df["date_unregistration"] <= self.cutoff_day)).sum())
        if early_unregs > 0:
            raise CohortInvariantError(
                f"Anti-leakage invariant violated: {early_unregs} eligible records have date_unregistration <= cutoff ({self.cutoff_day})."
            )

        early_pos = int(((eligible_df["withdrawn_after_cutoff"] == 1) & (eligible_df["date_unregistration"].notna()) & (eligible_df["date_unregistration"] <= self.cutoff_day)).sum())
        if early_pos > 0:
            raise CohortInvariantError(
                f"Anti-leakage invariant violated: {early_pos} records with date_unregistration <= cutoff ({self.cutoff_day}) received target 1."
            )

    def run(
        self,
        save_outputs: bool = True,
        output_dir: Path | None = None,
        report_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Execute end-to-end cohort construction pipeline and optionally persist results."""
        out_dir = Path(output_dir) if output_dir else PROCESSED_COHORT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        info_df, reg_df = self.load_inputs()
        results = self.build_cohort(info_df, reg_df)

        if save_outputs:
            cohort_df = results["cohort_df"]
            exclusions_df = results["exclusions_df"]
            audit_report = results["audit_report"]

            prefix = f"cohort_{self.policy}"
            parquet_path = out_dir / f"{prefix}.parquet"
            csv_path = out_dir / f"{prefix}.csv"

            cohort_df.to_parquet(parquet_path, index=False)
            cohort_df.to_csv(csv_path, index=False)

            excl_parquet = out_dir / f"{prefix}_exclusions.parquet"
            excl_csv = out_dir / f"{prefix}_exclusions.csv"
            exclusions_df.to_parquet(excl_parquet, index=False)
            exclusions_df.to_csv(excl_csv, index=False)

            if report_dir is not None:
                rep_dir = Path(report_dir)
            elif output_dir is not None and Path(output_dir).resolve() != PROCESSED_COHORT_DIR.resolve():
                rep_dir = Path(output_dir)
            else:
                rep_dir = QUALITY_REPORTS_DIR

            report_path = rep_dir / f"{prefix}_audit_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(audit_report, f, indent=2)

            logger.info("Persisted cohort outputs to %s and report to %s", out_dir, report_path)
            results["saved_paths"] = {
                "cohort_parquet": str(parquet_path),
                "cohort_csv": str(csv_path),
                "exclusions_parquet": str(excl_parquet),
                "exclusions_csv": str(excl_csv),
                "audit_report_json": str(report_path),
            }

        return results
