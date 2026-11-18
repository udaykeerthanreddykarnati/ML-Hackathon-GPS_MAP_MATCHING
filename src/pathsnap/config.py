"""Central configuration and path resolution for PathSnap."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDENT_DATA_ROOT = Path(
    os.environ.get("PATHSNAP_DATA_ROOT", str(PROJECT_ROOT / "Student folder"))
)
TRAIN_DIR = STUDENT_DATA_ROOT / "train"
STUDENT_TEST_DIR = STUDENT_DATA_ROOT / "student_test"

# For final evaluation on Laptop 1, set PATHSNAP_EVAL_DIR to the reserved evaluator folder.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(STUDENT_TEST_DIR)))
EVAL_SPLIT_NAME = os.environ.get("PATHSNAP_EVAL_SPLIT_NAME", EVAL_DIR.name)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42
TOP_K_CANDIDATES = 14
MAX_CAND_RADIUS_M = 150.0
ARC_SAMPLE_STEP_M = 25.0
CV_FOLDS = 5
TRAIN_MAX_ALIGN_DIST_M = 50.0
