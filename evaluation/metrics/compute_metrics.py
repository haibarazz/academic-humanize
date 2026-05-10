"""
Compute Academic Humanize metrics from a saved prediction report.

This is the preferred entrypoint for the separated evaluation workflow.
It intentionally performs no API calls and no model generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.recompute_from_report import main


if __name__ == "__main__":
    raise SystemExit(main())
