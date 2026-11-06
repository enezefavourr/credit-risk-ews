"""Project-wide paths and settings.

Set the environment variable DATA_DIR to point at the folder holding the real
Kaggle files. If it is not set, the pipeline falls back to the synthetic sample
in data/sample/ so the code always runs end to end.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

_raw = ROOT / "data" / "raw"
_sample = ROOT / "data" / "sample"


def _pick_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    if (_raw / "application_train.csv").exists():
        return _raw
    return _sample


DATA_DIR = _pick_data_dir()
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
POWERBI = ROOT / "powerbi" / "data"

for p in (ARTIFACTS, REPORTS, FIGURES, POWERBI):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Scorecard scaling: 600 points = 50:1 good:bad odds, 20 points to double the odds
SCORE_BASE = 600
ODDS_BASE = 50
PDO = 20

# Illustrative Basel-style assumption for unsecured consumer lending
LGD = 0.45

# DPD buckets used across DPD analysis, roll-rates and early warning
DPD_BUCKETS = [(-1, 0, "Current"), (0, 30, "1-30"), (30, 60, "31-60"),
               (60, 90, "61-90"), (90, 10**6, "90+")]

# Protected / non-decision attributes deliberately excluded from model features
EXCLUDED_FEATURES = ["CODE_GENDER"]
