"""Feature table assembly and orchestration pipeline for Day 35 early warning."""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    CANONICAL_TARGET,
    DEMOGRAPHIC_AUDIT_FEATURES,
)
from student_retention.features.feature_validation import validate_feature_table
from student_retention.paths import (
    INTERIM_NORMALIZED_DIR,
    PROCESSED_COHORT_DIR,
    PROCESSED_FEATURES_DIR,
    QUALITY_REPORTS_DIR,
)
from student_retention.transformation.assessment_features import (
    extract_assessment_features,
)
from student_retention.transformation.registration_features import (
    extract_registration_features,
)
from student_retention.transformation.vle_features import extract_vle_features


class FeatureTableBuilder:
    """Orchestrates temporal feature extraction and produces verified feature matrices."""

    def __init__(self, cutoff_day: int = 35) -> None:
        if cutoff_day != 35:
            raise ValueError(
                f"Unsupported cutoff_day={cutoff_day}. The Day-35 early warning feature engineering pipeline "
                f"strictly specifies and enforces cutoff_day=35."
            )
        self.cutoff_day = cutoff_day

    def build_feature_table(
        self,
        cohort_df: pd.DataFrame,
        student_vle_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        student_assessment_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        student_registration_df: pd.DataFrame | None = None,
        policy_version: str = "Policy C",
        enforce_registration_opportunity: bool = True,
        include_demographics: bool = True,
    ) -> pd.DataFrame:
        """Construct the complete standardized feature table for the cohort.

        Args:
            cohort_df: Cohort DataFrame with canonical keys and target.
            student_vle_df: VLE interaction logs.
            vle_df: VLE site metadata.
            student_assessment_df: Assessment submissions.
            assessments_df: Assessment schedules.
            student_registration_df: Optional registration table.
            policy_version: Policy identifier ("Policy C" or "Policy B").
            enforce_registration_opportunity: Whether to apply opportunity adjustment for late registrants.
            include_demographics: Whether to retain demographic audit features in output table.

        Returns:
            DataFrame: Validated feature table containing canonical grain, 22 primary features, target,
                       and optional demographic audit attributes.
        """
        initial_len = len(cohort_df)

        # 1. Course Context Features (4 features)
        # code_module, presentation_season, studied_credits, num_of_prev_attempts
        context_df = cohort_df[CANONICAL_GRAIN].copy()
        context_df["code_module"] = cohort_df["code_module"].astype(str)
        context_df["presentation_season"] = cohort_df["code_presentation"].astype(str).str[-1]
        context_df["studied_credits"] = cohort_df["studied_credits"].astype("Int64")
        context_df["num_of_prev_attempts"] = cohort_df["num_of_prev_attempts"].astype("Int64")

        # 2. Registration Features (2 features)
        # registration_lead_time_days, is_late_registration
        reg_df = extract_registration_features(
            cohort_df=cohort_df,
            student_registration_df=student_registration_df,
            cutoff_day=self.cutoff_day,
        )

        # 3. VLE Digital Engagement Features (11 features)
        vle_feats_df = extract_vle_features(
            cohort_df=cohort_df,
            student_vle_df=student_vle_df,
            vle_df=vle_df,
            cutoff_day=self.cutoff_day,
        )

        # 4. Assessment Progress Features (5 features)
        assess_feats_df = extract_assessment_features(
            cohort_df=cohort_df,
            student_assessment_df=student_assessment_df,
            assessments_df=assessments_df,
            cutoff_day=self.cutoff_day,
            enforce_registration_opportunity=enforce_registration_opportunity,
        )

        # 5. Assemble all feature groups onto cohort using canonical grain
        merged = context_df.merge(reg_df, on=CANONICAL_GRAIN, how="inner")
        merged = merged.merge(vle_feats_df, on=CANONICAL_GRAIN, how="inner")
        merged = merged.merge(assess_feats_df, on=CANONICAL_GRAIN, how="inner")

        # Verify zero row multiplication
        if len(merged) != initial_len:
            raise ValueError(
                f"Feature assembly resulted in row multiplication/loss: "
                f"expected {initial_len}, got {len(merged)}"
            )

        # Attach target variable
        if CANONICAL_TARGET in cohort_df.columns:
            merged[CANONICAL_TARGET] = cohort_df[CANONICAL_TARGET].values

        # Attach demographic audit attributes if requested
        if include_demographics:
            for demo_col in DEMOGRAPHIC_AUDIT_FEATURES:
                if demo_col in cohort_df.columns:
                    merged[demo_col] = cohort_df[demo_col].values

        # 6. Validate the assembled table
        validate_feature_table(
            df=merged,
            expected_row_count=initial_len,
            policy_version=policy_version,
        )

        return merged

    def run_pipeline(
        self,
        cohort_path: Path | None = None,
        interim_dir: Path | None = None,
        output_dir: Path | None = None,
        report_dir: Path | None = None,
        quality_report_path: Path | None = None,
        policy_version: str = "Policy C",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load normalized data, build feature matrix, validate, and write artifacts.

        Args:
            cohort_path: Path to input cohort parquet file.
            interim_dir: Optional custom directory containing normalized interim tables.
            output_dir: Directory to save feature matrix and quality report.
            report_dir: Optional custom directory to save quality report.
            quality_report_path: Optional explicit path to quality report JSON file.
            policy_version: "Policy C" or "Policy B".

        Returns:
            tuple[pd.DataFrame, dict]: Assembled feature DataFrame and validation report dict.
        """
        if cohort_path is None:
            filename = "cohort_policy_c.parquet" if policy_version == "Policy C" else "cohort_policy_b.parquet"
            cohort_path = PROCESSED_COHORT_DIR / filename

        if interim_dir is None:
            interim_dir = INTERIM_NORMALIZED_DIR

        if output_dir is None:
            output_dir = PROCESSED_FEATURES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load cohort and normalized datasets
        cohort_df = pd.read_parquet(cohort_path)
        student_vle_df = pd.read_parquet(interim_dir / "student_vle.parquet")
        vle_df = pd.read_parquet(interim_dir / "vle.parquet")
        student_assessment_df = pd.read_parquet(interim_dir / "student_assessment.parquet")
        assessments_df = pd.read_parquet(interim_dir / "assessments.parquet")

        feature_df = self.build_feature_table(
            cohort_df=cohort_df,
            student_vle_df=student_vle_df,
            vle_df=vle_df,
            student_assessment_df=student_assessment_df,
            assessments_df=assessments_df,
            policy_version=policy_version,
        )

        # Run validation and produce quality report
        val_report = validate_feature_table(
            df=feature_df,
            expected_row_count=len(cohort_df),
            policy_version=policy_version,
        )

        # Save artifacts
        prefix = "feature_matrix_policy_c" if policy_version == "Policy C" else "feature_matrix_policy_b"
        parquet_out = output_dir / f"{prefix}.parquet"
        csv_out = output_dir / f"{prefix}.csv"

        feature_df.to_parquet(parquet_out, index=False)
        feature_df.to_csv(csv_out, index=False)

        # Determine destination for quality report
        # When custom output_dir is passed and no custom report path given, isolate to output_dir
        if quality_report_path is not None:
            report_dest = quality_report_path
        elif report_dir is not None:
            report_dest = report_dir / "feature_quality_report.json"
        elif output_dir != PROCESSED_FEATURES_DIR:
            report_dest = output_dir / "feature_quality_report.json"
        else:
            report_dest = QUALITY_REPORTS_DIR / "feature_quality_report.json"

        report_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(report_dest, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        return feature_df, val_report
