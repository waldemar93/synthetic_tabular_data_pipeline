from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Backwards-compatible name for external scripts that imported PROJECT_DIR.
PROJECT_DIR = PROJECT_ROOT.name

