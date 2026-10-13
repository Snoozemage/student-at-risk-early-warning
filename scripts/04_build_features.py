"""CLI script alias for Phase 5 Temporal Feature Engineering (referenced in PROJECT_PLAN.md)."""
import importlib.util
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

script_path = PROJECT_ROOT / "scripts" / "05_build_features.py"
spec = importlib.util.spec_from_file_location("build_features", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == "__main__":
    sys.exit(module.main())
