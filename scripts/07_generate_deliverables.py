#!/usr/bin/env python
"""Generate project deliverables (compatibility alias for scripts/08_generate_deliverables.py)."""

from student_retention.logging_config import get_logger
from student_retention.reporting.generate_report import generate_all_deliverables

logger = get_logger(__name__)


def main() -> None:
    logger.info("Running deliverable generation...")
    generate_all_deliverables()


if __name__ == "__main__":
    main()
