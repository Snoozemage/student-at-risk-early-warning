#!/usr/bin/env python
"""Generate high-level markdown deliverables for the Smart Student Retention project."""

import sys

from student_retention.logging_config import get_logger
from student_retention.paths import DELIVERABLES_DIR
from student_retention.reporting.generate_report import (
    ReportingDataError,
    generate_all_deliverables,
)

logger = get_logger(__name__)


def main() -> None:
    """Entry point for deliverable generation."""
    logger.info("Generating project deliverables in %s...", DELIVERABLES_DIR)

    try:
        results = generate_all_deliverables()
        logger.info("Successfully generated all %d deliverables:", len(results))
        for key, path in results.items():
            size_kb = path.stat().st_size / 1024
            logger.info("  - %-22s: %s (%.1f KB)", key, path.name, size_kb)
    except (FileNotFoundError, ReportingDataError) as e:
        logger.error("Cannot generate deliverables due to missing or invalid inputs: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Failed to generate deliverables")
        sys.exit(1)


if __name__ == "__main__":
    main()
