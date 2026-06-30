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


# ============================================================
# QUBO / Annealing settings - improved
# ============================================================

QUBO_BITS_PER_VAR = 10
QUBO_NUM_READS = 80
QUBO_SWEEPS = 3000
QUBO_SEED = 0

QUBO_TEMP_START = 2.0
QUBO_TEMP_END = 0.001

# Nghiệm Δθ trong IEEE14 one-step cỡ 1e-4,
# nên range cũ ±1.8e-2 quá rộng.
QUBO_RANGE_FACTOR = 0.20
QUBO_MIN_ABS_RANGE = 1e-5
QUBO_MAX_ABS_RANGE = 0.002

# Iterative residual QUBO
QUBO_REFINEMENT_STEPS = 5
QUBO_REFINEMENT_SHRINK = 0.45
QUBO_DAMPING = 0.8

# Không cho QUBO update quá lớn trong FDLS
QUBO_MAX_STEP_NORM_FACTOR = 2.0


# ============================================================
# DC-OPF / Quantum Optimization Power Flow settings
# ============================================================

RUN_DC_OPF = True

# Nên để False trước. Sau khi Classical OPF chạy ổn thì bật True.
RUN_QUANTUM_OPF = True 

# Không nên bật line limits ngay từ đầu vì KKT lớn hơn.
OPF_INCLUDE_LINE_LIMITS = False

OPF_MAX_ITER = 20
OPF_QUANTUM_MAX_ITER = 6
OPF_TOL = 1e-7
OPF_SIGMA = 0.1
OPF_FRACTION_TO_BOUNDARY = 0.99
OPF_KKT_REG = 1e-9