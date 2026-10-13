"""Referential integrity and relationship constraint validation for OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.config import get_quality_rules_config
from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def validate_relationship(
    parent_name: str,
    parent_df: pd.DataFrame,
    child_name: str,
    child_df: pd.DataFrame,
    join_keys: list[str],
    allow_orphans: bool = False,
    cardinality: str | None = None,
) -> dict[str, Any]:
    """Validate referential integrity between a parent and child table.

    Uses efficient key-level deduplication to scale cleanly to large event tables.

    Args:
        parent_name: Name of parent table
        parent_df: Parent DataFrame
        child_name: Name of child table
        child_df: Child DataFrame
        join_keys: List of column names used to join
        allow_orphans: Whether child keys missing from parent are allowed
        cardinality: Expected cardinality (e.g. '1:1')

    Returns:
        Structured dictionary with relationship validation results.
    """
    violations: list[str] = []

    # Check key existence
    missing_parent_keys = [k for k in join_keys if k not in parent_df.columns]
    missing_child_keys = [k for k in join_keys if k not in child_df.columns]

    if missing_parent_keys or missing_child_keys:
        if missing_parent_keys:
            violations.append(f"Parent '{parent_name}' missing join keys: {missing_parent_keys}")
        if missing_child_keys:
            violations.append(f"Child '{child_name}' missing join keys: {missing_child_keys}")
        return {
            "parent": parent_name,
            "child": child_name,
            "join_keys": join_keys,
            "is_valid": False,
            "orphan_child_rows": 0,
            "orphan_child_keys": 0,
            "violations": violations,
        }

    # Extract distinct non-null keys from parent and child
    parent_keys = parent_df[join_keys].dropna().drop_duplicates()
    child_keys = child_df[join_keys].dropna().drop_duplicates()

    # Find child keys missing from parent via left join
    merged_child = child_keys.merge(parent_keys, on=join_keys, how="left", indicator=True)
    orphan_keys_df = merged_child[merged_child["_merge"] != "both"]
    orphan_keys = len(orphan_keys_df)

    if orphan_keys > 0:
        # Determine actual row count in child_df that have orphan keys
        orphan_child_rows = int(child_df.merge(orphan_keys_df[join_keys], on=join_keys, how="inner").shape[0])
        sample_orphans = orphan_keys_df[join_keys].head(3).to_dict("records")
        if not allow_orphans:
            violations.append(
                f"Child '{child_name}' has {orphan_child_rows} row(s) ({orphan_keys} distinct key(s)) "
                f"not found in parent '{parent_name}' on join keys {join_keys}. Examples: {sample_orphans}"
            )
    else:
        orphan_child_rows = 0

    # Cardinality check: 1:1
    if cardinality == "1:1":
        parent_dups = int(parent_df.duplicated(subset=join_keys).sum())
        if parent_dups > 0:
            violations.append(f"Expected 1:1 cardinality but parent '{parent_name}' has {parent_dups} duplicate keys")

        child_dups = int(child_df.duplicated(subset=join_keys).sum())
        if child_dups > 0:
            violations.append(f"Expected 1:1 cardinality but child '{child_name}' has {child_dups} duplicate keys")

        # Symmetric check: parent records missing from child
        merged_parent = parent_keys.merge(child_keys, on=join_keys, how="left", indicator=True)
        orphan_parents_df = merged_parent[merged_parent["_merge"] != "both"]
        orphan_parent_keys = len(orphan_parents_df)

        if orphan_parent_keys > 0:
            orphan_parent_rows = int(parent_df.merge(orphan_parents_df[join_keys], on=join_keys, how="inner").shape[0])
            violations.append(
                f"Expected 1:1 cardinality but parent '{parent_name}' has {orphan_parent_rows} row(s) "
                f"({orphan_parent_keys} distinct key(s)) missing from child '{child_name}'"
            )

    is_valid = len(violations) == 0

    return {
        "parent": parent_name,
        "child": child_name,
        "join_keys": join_keys,
        "cardinality": cardinality,
        "is_valid": is_valid,
        "orphan_child_rows": orphan_child_rows,
        "orphan_child_keys": orphan_keys,
        "violations": violations,
    }


def validate_student_assessment_enrollment(
    student_assessment_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    student_info_df: pd.DataFrame,
    allow_orphans: bool = False,
) -> dict[str, Any]:
    """Validate student_assessment compatibility with student_info enrollment.

    Resolves id_assessment through assessments to obtain (code_module, code_presentation),
    then verifies that (code_module, code_presentation, id_student) exists in student_info.

    Args:
        student_assessment_df: student_assessment DataFrame
        assessments_df: assessments DataFrame
        student_info_df: student_info DataFrame
        allow_orphans: Whether unlinked enrollment combinations are allowed

    Returns:
        Structured dictionary with validation results and violations.
    """
    violations: list[str] = []

    # Check required columns
    sa_required = ["id_assessment", "id_student"]
    assess_required = ["id_assessment", "code_module", "code_presentation"]
    info_required = ["code_module", "code_presentation", "id_student"]

    missing_sa = [c for c in sa_required if c not in student_assessment_df.columns]
    missing_assess = [c for c in assess_required if c not in assessments_df.columns]
    missing_info = [c for c in info_required if c not in student_info_df.columns]

    if missing_sa or missing_assess or missing_info:
        if missing_sa:
            violations.append(f"student_assessment missing columns: {missing_sa}")
        if missing_assess:
            violations.append(f"assessments missing columns: {missing_assess}")
        if missing_info:
            violations.append(f"student_info missing columns: {missing_info}")
        return {
            "parent": "student_info",
            "child": "student_assessment",
            "via": "assessments",
            "join_keys": info_required,
            "bridge_keys": ["id_assessment"],
            "is_valid": False,
            "orphan_child_rows": 0,
            "orphan_child_keys": 0,
            "violations": violations,
        }

    # Efficient key-level resolution
    sa_keys = student_assessment_df[["id_assessment", "id_student"]].dropna().drop_duplicates()
    assess_keys = assessments_df[["id_assessment", "code_module", "code_presentation"]].dropna().drop_duplicates()
    info_keys = student_info_df[["code_module", "code_presentation", "id_student"]].dropna().drop_duplicates()

    # Step 1: Bridge student_assessment -> assessments
    resolved = sa_keys.merge(assess_keys, on="id_assessment", how="inner")

    # Step 2: Check resolved against student_info
    merged = resolved.merge(
        info_keys,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        indicator=True,
    )
    unmatched_keys_df = merged[merged["_merge"] != "both"]
    orphan_keys = len(unmatched_keys_df)

    if orphan_keys > 0:
        # Calculate matching rows in student_assessment
        orphan_rows = int(
            student_assessment_df.merge(
                unmatched_keys_df[["id_assessment", "id_student"]],
                on=["id_assessment", "id_student"],
                how="inner",
            ).shape[0]
        )
        sample_unmatched = unmatched_keys_df[
            ["id_assessment", "id_student", "code_module", "code_presentation"]
        ].head(3).to_dict("records")
        if not allow_orphans:
            violations.append(
                f"student_assessment has {orphan_rows} row(s) ({orphan_keys} distinct pairs) whose "
                f"assessment does not match an active student_info enrollment. Examples: {sample_unmatched}"
            )
    else:
        orphan_rows = 0

    is_valid = len(violations) == 0

    return {
        "parent": "student_info",
        "child": "student_assessment",
        "via": "assessments",
        "join_keys": ["code_module", "code_presentation", "id_student"],
        "bridge_keys": ["id_assessment"],
        "is_valid": is_valid,
        "orphan_child_rows": orphan_rows,
        "orphan_child_keys": orphan_keys,
        "violations": violations,
    }


def validate_all_relationships(
    tables: dict[str, pd.DataFrame],
    rules_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all configured referential integrity rules across tables."""
    quality_rules = rules_cfg or get_quality_rules_config()
    source_rules = quality_rules.get("source_quality_rules", {})
    ref_rules = source_rules.get("referential_integrity", [])

    results: list[dict[str, Any]] = []
    all_valid = True
    total_violations = 0

    for rule in ref_rules:
        parent_name = rule["parent"]
        child_name = rule["child"]
        allow_orphans = rule.get("allow_orphans", False)

        # Check for bridged / indirect relationship
        if rule.get("via_parent") == "assessments" and child_name == "student_assessment":
            required_tables = ["student_assessment", "assessments", "student_info"]
            missing_tables = [t for t in required_tables if t not in tables]
            if missing_tables:
                results.append({
                    "parent": parent_name,
                    "child": child_name,
                    "via": "assessments",
                    "join_keys": rule.get("join_keys", []),
                    "is_valid": False,
                    "orphan_child_rows": 0,
                    "orphan_child_keys": 0,
                    "violations": [
                        f"Missing required table(s) for bridged relationship check: {missing_tables}"
                    ],
                })
                all_valid = False
                total_violations += 1
            else:
                res = validate_student_assessment_enrollment(
                    student_assessment_df=tables["student_assessment"],
                    assessments_df=tables["assessments"],
                    student_info_df=tables["student_info"],
                    allow_orphans=allow_orphans,
                )
                results.append(res)
                if not res["is_valid"]:
                    all_valid = False
                    total_violations += len(res["violations"])
            continue

        if "join_keys" not in rule:
            results.append({
                "parent": parent_name,
                "child": child_name,
                "is_valid": False,
                "violations": ["Configured rule is missing 'join_keys' and is unsupported."],
            })
            all_valid = False
            total_violations += 1
            continue

        join_keys = rule["join_keys"]
        cardinality = rule.get("cardinality")

        if parent_name not in tables or child_name not in tables:
            missing = [name for name in (parent_name, child_name) if name not in tables]
            results.append({
                "parent": parent_name,
                "child": child_name,
                "join_keys": join_keys,
                "is_valid": False,
                "violations": [f"Missing table(s) in dictionary: {missing}"],
            })
            all_valid = False
            total_violations += 1
            continue

        res = validate_relationship(
            parent_name=parent_name,
            parent_df=tables[parent_name],
            child_name=child_name,
            child_df=tables[child_name],
            join_keys=join_keys,
            allow_orphans=allow_orphans,
            cardinality=cardinality,
        )
        results.append(res)
        if not res["is_valid"]:
            all_valid = False
            total_violations += len(res["violations"])

    return {
        "is_valid": all_valid and (len(results) == len(ref_rules)),
        "configured_rules_count": len(ref_rules),
        "executed_rules_count": len(results),
        "total_violations": total_violations,
        "relationships": results,
    }
