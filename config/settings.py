import sys
from pathlib import Path
from core.data.legacy import LEGACY_FEATURES, LEGACY_TARGETS, LEGACY_NOISE

if getattr(sys, "frozen", False):
    ROOT = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).resolve().parent
    RESULTS_DIR = EXE_DIR / "results"
    _external_data = EXE_DIR / "data.xlsx"
    DATA_PATH = _external_data if _external_data.exists() else ROOT / "config" / "data.xlsx"
else:
    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"
    DATA_PATH = ROOT / "config" / "data.xlsx"

SEED = 1337
# Deprecated compatibility aliases; runtime code uses DatasetSchema.
FEATURES = list(LEGACY_FEATURES)
TARGETS = dict(LEGACY_TARGETS)

SYNTH = {
    "enabled": True,
    "samples_per_profile": 15,
    "noise": dict(LEGACY_NOISE),
}
ACTIVE_MODELS = [
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "bio_ga",
]
