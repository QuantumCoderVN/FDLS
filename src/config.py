import os

OUTPUT_DIR = "outputs"
MATRIX_DIR = os.path.join(OUTPUT_DIR, "matrices")
RESULT_TEXT_FILE = os.path.join(OUTPUT_DIR, "run_results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)


# ============================================================
# FDLS settings
# ============================================================

FDLS_MAX_ITER = 12
FDLS_TOL = 1e-6


# ============================================================
# VQLS no-CZ settings
# ============================================================

VQLS_LAYERS = 3
VQLS_STEPS = 100
VQLS_RESTARTS = 5
VQLS_SEED = 0
VQLS_Q_DELTA = 0.001


# ============================================================
# HHL settings
# ============================================================

HHL_PHASE_QUBITS = 6
HHL_TIME = 0.6
HHL_TARGET_MAX_EIGENVALUE = 5.0