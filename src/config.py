import os

# ============================================================
# Output folders
# ============================================================

OUTPUT_DIR = "outputs"
MATRIX_DIR = os.path.join(OUTPUT_DIR, "matrices")
RESULT_TEXT_FILE = os.path.join(OUTPUT_DIR, "run_results.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)


# ============================================================
# Power system test case
# ============================================================

POWER_CASE = "case6ww"
POWER_CASE_NAME = POWER_CASE


# ============================================================
# Main execution switches
# ============================================================

RUN_ONE_STEP_BENCHMARK = False
RUN_FDLS_BENCHMARK = False
RUN_DC_OPF = True


# ============================================================
# One-step benchmark solver switches
# ============================================================

RUN_ONE_STEP_VQLS = True
RUN_ONE_STEP_HHL = False
RUN_ONE_STEP_QUBO = False


# ============================================================
# FDLS benchmark solver switches
# ============================================================

RUN_FDLS_VQLS = True
RUN_FDLS_HHL = False
RUN_FDLS_QUBO = False


# ============================================================
# FDLS settings
# ============================================================

FDLS_MAX_ITER = 12
FDLS_TOL = 1e-6


# ============================================================
# DC-OPF / Quantum Optimization Power Flow settings
# ============================================================

RUN_QUANTUM_OPF = True

# RAW TEST: bật cả 3 solver.
RUN_OPF_VQLS = True
RUN_OPF_HHL = True
RUN_OPF_QUBO = True

OPF_INCLUDE_LINE_LIMITS = False

# Raw test không nên chạy quá dài.
# Nếu solver đi sai hướng, chạy dài chỉ làm số nổ rất lớn.
OPF_MAX_ITER = 20
OPF_QUANTUM_MAX_ITER = 8

OPF_TOL = 1e-6

# Khôi phục setting chuẩn để classical baseline vẫn hội tụ tốt.
OPF_SIGMA = 0.1
OPF_FRACTION_TO_BOUNDARY = 0.95
OPF_KKT_REG = 1e-7

# NO CLASSICAL FALLBACK:
# Đặt rất lớn để VQLS/HHL/QUBO không bị thay bằng classical solve
# trong solve_symmetric_preconditioned_kkt().
OPF_QUANTUM_FALLBACK_RELRES = 999.0

OPF_MAX_QUANTUM_KKT_DIM = 64


# ============================================================
# VQLS settings
# ============================================================
# Cấu hình tương đối mạnh nhưng vẫn chạy được trên case6ww.

VQLS_LAYERS = 2
VQLS_STEPS = 80
VQLS_RESTARTS = 2
VQLS_SEED = 0
VQLS_Q_DELTA = 0.001


# ============================================================
# HHL settings
# ============================================================
# HHL padded KKT 14 -> 16.
# 4 phase qubits chạy nhanh hơn; nếu HHL có vẻ ổn, sau đó tăng lên 5.

HHL_PHASE_QUBITS = 4
HHL_TIME = 0.6
HHL_TARGET_MAX_EIGENVALUE = 5.0


# ============================================================
# QUBO / Annealing settings
# ============================================================
# QUBO OPF-KKT rất dễ chậm/nổ.
# Giữ bit thấp để chạy khảo sát nhanh.

QUBO_BITS_PER_VAR = 3
QUBO_NUM_READS = 8
QUBO_SWEEPS = 200
QUBO_SEED = 0

QUBO_TEMP_START = 2.0
QUBO_TEMP_END = 0.001

QUBO_RANGE_FACTOR = 0.20
QUBO_MIN_ABS_RANGE = 1e-5
QUBO_MAX_ABS_RANGE = 0.002

QUBO_REFINEMENT_STEPS = 1
QUBO_REFINEMENT_SHRINK = 0.5
QUBO_DAMPING = 1.0

# Không còn dùng để clamp nếu bạn sửa qubo_p_solver raw bên dưới.
QUBO_MAX_STEP_NORM_FACTOR = 999.0