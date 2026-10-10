"""Project paths, resolved from this file location (cwd-independent)."""

from __future__ import annotations

from pathlib import Path

# notebooks/src/underwriting/paths.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
