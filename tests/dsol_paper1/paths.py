"""One explicit repository root, independent of individual test module depth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
