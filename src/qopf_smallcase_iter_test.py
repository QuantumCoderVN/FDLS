# ============================================================
# IEEE 14-bus FDLS: Classical vs VQLS no-CZ vs HHL
# ============================================================

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, RYGate
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator

import importlib


warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATRIX_DIR = os.path.join(OUTPUT_DIR, "matrices")
os.makedirs(MATRIX_DIR, exist_ok=True)

RESULT_TEXT_FILE = os.path.join(OUTPUT_DIR, "run_results.txt")


# ============================================================
# Settings
# ============================================================

# Small-case / speed settings
POWER_CASE = "auto_smallest"   # auto_smallest -> tries case4gs, case6ww, case9, case14
RUN_FDLS = False                # False = skip FDLS to focus on QOPF speed test
RUN_OPF = True                  # True = run DC-OPF/IPM

FDLS_MAX_ITER = 60              # increased to test whether FDLS reaches 1e-6
FDLS_TOL = 1e-6

# VQLS no-CZ settings
VQLS_LAYERS = 2
VQLS_STEPS = 40
VQLS_RESTARTS = 1
VQLS_SEED = 0
VQLS_Q_DELTA = 0.001

# HHL settings
HHL_PHASE_QUBITS = 4
HHL_TIME = 0.5
HHL_TARGET_MAX_EIGENVALUE = 5.0


# ============================================================
# Text logger
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================
# MATPOWER / PYPOWER column indices
# ============================================================

BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 7
VA = 8

PQ = 1
PV = 2
REF = 3

GEN_BUS = 0
PG = 1
QG = 2
VG = 5
GEN_STATUS = 7

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 8
SHIFT = 9
BR_STATUS = 10
RATE_A = 5

PMAX = 8
PMIN = 9

GENCOST_MODEL = 0
GENCOST_NCOST = 3
GENCOST_C2 = 4
GENCOST_C1 = 5
GENCOST_C0 = 6


# ============================================================
# 1. Load selected PYPOWER case
# ============================================================

def import_pypower_case(case_name):
    """
    Import a PYPOWER case by name, for example: case4gs, case6ww, case9, case14.
    """
    module = importlib.import_module(f"pypower.{case_name}")
    return getattr(module, case_name)


def resolve_power_case(case_name="auto_smallest"):
    """
    Choose the smallest available PYPOWER case.

    Priority:
        case4gs -> case6ww -> case9 -> case14

    If your PYPOWER installation does not include case4gs, the function
    automatically falls back to the next available case.
    """
    if case_name != "auto_smallest":
        return case_name

    for candidate in ["case4gs", "case6ww", "case9", "case14"]:
        try:
            import_pypower_case(candidate)
            return candidate
        except Exception:
            continue

    raise ImportError("No supported PYPOWER case found. Tried case4gs, case6ww, case9, case14.")


def load_power_case(case_name=POWER_CASE):
    """
    Load a PYPOWER/MATPOWER case.

    This replaces the old fixed load_ieee14_case(). For fast QOPF tests,
    use POWER_CASE = "auto_smallest" or "case4gs".
    """
    resolved_name = resolve_power_case(case_name)
    case_fn = import_pypower_case(resolved_name)
    mpc = case_fn()

    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)

    if "gencost" not in mpc:
        raise ValueError(f"{resolved_name} does not contain gencost; OPF needs gencost.")

    gencost = np.asarray(mpc["gencost"], dtype=float)

    bus_numbers = bus[:, BUS_I].astype(int)
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_numbers)}

    nbus = bus.shape[0]

    slack = np.where(bus[:, BUS_TYPE].astype(int) == REF)[0]
    pv = np.where(bus[:, BUS_TYPE].astype(int) == PV)[0]
    pq = np.where(bus[:, BUS_TYPE].astype(int) == PQ)[0]

    if len(slack) != 1:
        raise ValueError("This script expects exactly one slack bus.")

    slack = int(slack[0])
    non_slack = [i for i in range(nbus) if i != slack]

    return {
        "case_name": resolved_name,
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
        "bus_id_to_idx": bus_id_to_idx,
        "nbus": nbus,
        "slack": slack,
        "pv": list(pv),
        "pq": list(pq),
        "non_slack": non_slack,
    }


# Backward-compatible alias: old code can still call load_ieee14_case().
def load_ieee14_case():
    return load_power_case(POWER_CASE)


def make_ybus(case):
    """
    Build full AC Ybus from MATPOWER branch data.

    Includes:
        - line resistance r
        - line reactance x
        - line charging b
        - transformer tap and phase shift
        - bus shunts Gs + jBs
    """

    nbus = case["nbus"]
    base_mva = case["base_mva"]
    bus = case["bus"]
    branch = case["branch"]
    bus_id_to_idx = case["bus_id_to_idx"]

    Ybus = np.zeros((nbus, nbus), dtype=np.complex128)

    for br in branch:
        if int(br[BR_STATUS]) == 0:
            continue

        f_ext = int(br[F_BUS])
        t_ext = int(br[T_BUS])

        f = bus_id_to_idx[f_ext]
        t = bus_id_to_idx[t_ext]

        r = br[BR_R]
        x = br[BR_X]
        b = br[BR_B]

        z = r + 1j * x
        y = 1.0 / z

        tap_mag = br[TAP]
        if abs(tap_mag) < 1e-14:
            tap_mag = 1.0

        shift_rad = np.deg2rad(br[SHIFT])
        tap = tap_mag * np.exp(1j * shift_rad)

        y_shunt = 1j * b / 2.0

        Yff = (y + y_shunt) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        Ytf = -y / tap
        Ytt = y + y_shunt

        Ybus[f, f] += Yff
        Ybus[f, t] += Yft
        Ybus[t, f] += Ytf
        Ybus[t, t] += Ytt

    # Bus shunts: Gs + jBs in MW/MVAr at V=1.0, convert to p.u.
    for i in range(nbus):
        gs = bus[i, GS] / base_mva
        bs = bus[i, BS] / base_mva
        Ybus[i, i] += gs + 1j * bs

    return Ybus


def make_specified_power(case):
    """
    Build P_spec and Q_spec in p.u.

        P_spec = (sum Pg - Pd) / baseMVA
        Q_spec = (sum Qg - Qd) / baseMVA
    """

    base_mva = case["base_mva"]
    bus = case["bus"]
    gen = case["gen"]
    bus_id_to_idx = case["bus_id_to_idx"]

    nbus = case["nbus"]

    P_spec = -bus[:, PD] / base_mva
    Q_spec = -bus[:, QD] / base_mva

    for g in gen:
        if int(g[GEN_STATUS]) == 0:
            continue

        bus_ext = int(g[GEN_BUS])
        idx = bus_id_to_idx[bus_ext]

        P_spec[idx] += g[PG] / base_mva
        Q_spec[idx] += g[QG] / base_mva

    return P_spec, Q_spec


def initial_voltage(case):
    """
    Use MATPOWER initial voltage magnitude and angle.
    """

    bus = case["bus"]

    vm = bus[:, VM].astype(float).copy()
    va = np.deg2rad(bus[:, VA].astype(float).copy())

    return vm, va


def calculated_power(Ybus, vm, va):
    """
    AC power injection:

        S = V * conj(Ybus V)

    Returns P_calc and Q_calc in p.u.
    """

    V = vm * np.exp(1j * va)
    S = V * np.conj(Ybus @ V)

    return np.real(S), np.imag(S)


def make_fdls_matrices(Ybus, case):
    """
    Simple FDLS B matrices:

        B = -imag(Ybus)
        B'  = B without slack bus
        B'' = B restricted to PQ buses
    """

    B = -np.imag(Ybus)

    non_slack = case["non_slack"]
    pq = case["pq"]

    Bprime = B[np.ix_(non_slack, non_slack)]
    Bdouble = B[np.ix_(pq, pq)]

    # Symmetrize for quantum solvers
    Bprime = 0.5 * (Bprime + Bprime.T)
    Bdouble = 0.5 * (Bdouble + Bdouble.T)

    return Bprime, Bdouble


# ============================================================
# 2. Export matrices
# ============================================================

def save_complex_matrix_csv(path_real, path_imag, matrix):
    np.savetxt(path_real, np.real(matrix), delimiter=",", fmt="%.12e")
    np.savetxt(path_imag, np.imag(matrix), delimiter=",", fmt="%.12e")


def export_matrices(Ybus, Bprime, Bdouble, rhs_p0, case):
    save_complex_matrix_csv(
        os.path.join(MATRIX_DIR, "Ybus_real.csv"),
        os.path.join(MATRIX_DIR, "Ybus_imag.csv"),
        Ybus,
    )

    np.savetxt(os.path.join(MATRIX_DIR, "Bprime.csv"), Bprime, delimiter=",", fmt="%.12e")
    np.savetxt(os.path.join(MATRIX_DIR, "Bdouble_prime.csv"), Bdouble, delimiter=",", fmt="%.12e")
    np.savetxt(os.path.join(MATRIX_DIR, "rhs_p_initial.csv"), rhs_p0, delimiter=",", fmt="%.12e")

    with open(os.path.join(MATRIX_DIR, "matrix_info.txt"), "w", encoding="utf-8") as f:
        f.write("IEEE 14-bus matrix export\n")
        f.write("=" * 80 + "\n")
        f.write(f"nbus = {case['nbus']}\n")
        f.write(f"baseMVA = {case['base_mva']}\n")
        f.write(f"slack bus internal index = {case['slack']}\n")
        f.write(f"PV internal indices = {case['pv']}\n")
        f.write(f"PQ internal indices = {case['pq']}\n")
        f.write(f"non-slack internal indices = {case['non_slack']}\n")
        f.write(f"Ybus shape = {Ybus.shape}\n")
        f.write(f"Bprime shape = {Bprime.shape}\n")
        f.write(f"Bdouble_prime shape = {Bdouble.shape}\n")
        f.write(f"rhs_p_initial shape = {rhs_p0.shape}\n")


# ============================================================
# 3. Utilities for quantum linear solvers
# ============================================================

def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def pad_linear_system(A, b):
    """
    Pad A x = b to dimension 2^n.

    Original IEEE 14 P-theta system has size 13.
    Quantum statevector solvers need dimension 2^n.
    Therefore:
        13 -> 16
    """

    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    n = A.shape[0]
    dim = next_power_of_two(n)

    A_pad = np.eye(dim, dtype=np.complex128)
    b_pad = np.zeros(dim, dtype=np.complex128)

    A_pad[:n, :n] = A
    b_pad[:n] = b

    A_pad = 0.5 * (A_pad + A_pad.conj().T)

    return A_pad, b_pad, n


def normalize_state(v):
    v = np.asarray(v, dtype=np.complex128)
    norm = np.linalg.norm(v)

    if norm < 1e-14:
        raise ValueError("Cannot normalize zero vector.")

    return v / norm


def recover_scaled_solution(A, b, x_state):
    """
    Recover classical-scale solution from normalized quantum state.

        x = k |x_state>

        k = <A x_state | b> / <A x_state | A x_state>
    """

    x_state = normalize_state(x_state)

    Ax = A @ x_state
    denom = np.vdot(Ax, Ax)

    if abs(denom) < 1e-14:
        return np.zeros_like(b), 0.0

    k = np.vdot(Ax, b) / denom
    x = k * x_state

    return x, k


def project_real_direction(x_state):
    """
    The IEEE FDLS matrix and RHS are real.
    Use real direction to remove unnecessary complex global phase.
    """

    x_state = np.asarray(x_state, dtype=np.complex128)
    real_part = np.real(x_state)

    if np.linalg.norm(real_part) < 1e-14:
        return normalize_state(x_state)

    return normalize_state(real_part)


def clean_real_vector(x):
    x = np.asarray(x, dtype=np.complex128)
    x = np.real_if_close(x, tol=1000)

    if np.iscomplexobj(x):
        return np.real(x)

    return np.asarray(x, dtype=float)


def infer_n_qubits_from_matrix(A):
    dim = A.shape[0]
    n_qubits = int(np.log2(dim))

    if 2 ** n_qubits != dim:
        raise ValueError("Matrix dimension must be power of two.")

    return n_qubits


def vector_metrics(A, b, x, x_ref):
    residual = np.linalg.norm(A @ x - b)
    rel_residual = residual / max(np.linalg.norm(b), 1e-14)

    error = np.linalg.norm(x - x_ref)
    rel_error = error / max(np.linalg.norm(x_ref), 1e-14)

    fidelity = np.abs(
        np.vdot(normalize_state(x_ref), normalize_state(x))
    ) ** 2

    return {
        "residual": float(residual),
        "rel_residual": float(rel_residual),
        "error": float(error),
        "rel_error": float(rel_error),
        "fidelity": float(fidelity),
    }


# ============================================================
# 4. Classical solver
# ============================================================

def solve_classical(A, b):
    return np.linalg.solve(A, b)


# ============================================================
# 5. VQLS no-CZ solver
# ============================================================

def pauli_decompose_matrix(A):
    n_qubits = infer_n_qubits_from_matrix(A)

    op = Operator(
        A,
        input_dims=(2,) * n_qubits,
        output_dims=(2,) * n_qubits,
    )

    return SparsePauliOp.from_operator(op, atol=1e-10, rtol=1e-10)


def apply_vqls_ansatz_no_cz(qc, params, qubits, layers):
    """
    VQLS ansatz without CZ gates.

    Gates used:
        H, RY, RZ, CX
    """

    params = np.asarray(params, dtype=float)

    for q in qubits:
        qc.h(q)

    k = 0

    for layer in range(layers):
        for q in qubits:
            qc.ry(params[k], q)
            k += 1

        for q in qubits:
            qc.rz(params[k], q)
            k += 1

        if layer < layers - 1:
            for q1, q2 in zip(qubits[:-1], qubits[1:]):
                qc.cx(q1, q2)

            for q1, q2 in zip(reversed(qubits[:-1]), reversed(qubits[1:])):
                qc.cx(q2, q1)


def vqls_state(params, n_qubits, layers):
    qc = QuantumCircuit(n_qubits)
    apply_vqls_ansatz_no_cz(qc, params, list(range(n_qubits)), layers)
    state = Statevector.from_instruction(qc).data
    return normalize_state(state)


def vqls_global_cost(params, A_scaled, b_norm, n_qubits, layers):
    x = vqls_state(params, n_qubits, layers)
    psi = A_scaled @ x

    denom = np.vdot(psi, psi).real

    if denom < 1e-14:
        return 1.0

    numerator = np.abs(np.vdot(b_norm, psi)) ** 2
    return float(1.0 - numerator / denom)


def solve_vqls_no_cz(A_original, b_original, verbose=False):
    """
    Solve A x = b using VQLS no-CZ.

    If A is not 2^n, it is padded:
        13 x 13 -> 16 x 16
    """

    A_pad, b_pad, original_dim = pad_linear_system(A_original, b_original)

    n_qubits = infer_n_qubits_from_matrix(A_pad)
    n_params = 2 * n_qubits * VQLS_LAYERS

    A_scale = np.linalg.norm(A_pad, ord=2)
    A_scaled = A_pad / A_scale
    b_norm = normalize_state(b_pad)

    pauli_op = pauli_decompose_matrix(A_scaled)

    if verbose:
        print("\n" + "=" * 80)
        print("VQLS NO-CZ SOLVER")
        print("=" * 80)
        print(f"Original dimension = {original_dim}")
        print(f"Padded dimension   = {A_pad.shape[0]}")
        print(f"System qubits      = {n_qubits}")
        print(f"Layers             = {VQLS_LAYERS}")
        print(f"Parameters         = {n_params}")
        print(f"Steps              = {VQLS_STEPS}")
        print(f"Restarts           = {VQLS_RESTARTS}")
        print(f"Pauli terms        = {len(pauli_op.coeffs)}")
        print("No CZ gate is used.")
        print("\nPauli decomposition of padded scaled A:")
        print(pauli_op)

    rng = np.random.default_rng(VQLS_SEED)

    best = None

    for restart in range(VQLS_RESTARTS):
        if restart == 0:
            w0 = VQLS_Q_DELTA * rng.standard_normal(n_params)
        else:
            w0 = rng.uniform(-np.pi, np.pi, size=n_params)

        cost_history = []

        def objective(w):
            cost = vqls_global_cost(
                w,
                A_scaled=A_scaled,
                b_norm=b_norm,
                n_qubits=n_qubits,
                layers=VQLS_LAYERS,
            )
            cost_history.append(cost)
            return cost

        res = minimize(
            objective,
            w0,
            method="COBYLA",
            options={
                "rhobeg": 1.0,
                "maxiter": VQLS_STEPS,
                "catol": 1e-10,
            },
        )

        raw_state = vqls_state(res.x, n_qubits, VQLS_LAYERS)
        real_state = project_real_direction(raw_state)

        x_pad, k = recover_scaled_solution(A_pad, b_pad, real_state)
        x = clean_real_vector(x_pad[:original_dim])

        residual = np.linalg.norm(A_original @ x - b_original)
        rel_residual = residual / max(np.linalg.norm(b_original), 1e-14)

        candidate = {
            "restart": restart,
            "x": x,
            "x_state": real_state,
            "k": k,
            "cost": float(res.fun),
            "cost_history": cost_history,
            "rel_residual": float(rel_residual),
        }

        if verbose:
            print(
                f"Restart {restart:2d} | "
                f"cost={res.fun:.6e} | "
                f"relative residual={rel_residual:.6e}"
            )

        if best is None or candidate["rel_residual"] < best["rel_residual"]:
            best = candidate

    if best is None:
        raise RuntimeError("VQLS failed.")

    if verbose:
        print("\nBest VQLS:")
        print(f"restart           = {best['restart']}")
        print(f"cost              = {best['cost']:.6e}")
        print(f"relative residual = {best['rel_residual']:.6e}")
        print(f"k                 = {best['k']}")

    plt.figure(figsize=(7, 4))
    plt.plot(best["cost_history"], marker="o")
    plt.yscale("log")
    plt.xlabel("COBYLA evaluation")
    plt.ylabel("Global VQLS cost")
    plt.title("VQLS no-CZ cost convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vqls_no_cz_cost.png"), dpi=150)
    plt.close()

    return best["x"], best


# ============================================================
# 6. HHL solver
# ============================================================

def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    U_power = np.linalg.matrix_power(U, power)

    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(U_power, list(range(len(target_qubits))))

    qc.append(gate.to_gate().control(1), [control_qubit] + list(target_qubits))


def qpe_circuit(qc, U, phase_reg, target_reg):
    for q in phase_reg:
        qc.h(q)

    for i, ctrl in enumerate(reversed(phase_reg)):
        controlled_unitary_power(qc, U, 2 ** i, ctrl, target_reg)

    iqft = QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False)
    qc.append(iqft, phase_reg)


def control_rotation_gate(qc, control_reg, target_qubit, t, C):
    n = len(control_reg)

    for d in range(1, 2 ** n):
        bin_str = f"{d:0{n}b}"

        phi = sum(int(bit) * 2 ** (-(j + 1)) for j, bit in enumerate(bin_str))

        if phi > 0.5:
            phi -= 1.0

        lam = (2 * np.pi / t) * phi

        if abs(lam) < 1e-12:
            continue

        amp = C / lam

        if abs(amp) > 1:
            continue

        theta = 2 * np.arcsin(amp)
        cry = RYGate(theta).control(n, ctrl_state=bin_str)
        qc.append(cry, list(control_reg) + [target_qubit])


def extract_hhl_amplitude_state(statevector, phase_qubits, n_system_qubits, ancilla_index):
    dim = 2 ** n_system_qubits
    raw = np.zeros(dim, dtype=np.complex128)

    ancilla_bit = 1 << ancilla_index

    for target_basis in range(dim):
        full_index = ancilla_bit | (target_basis << phase_qubits)
        raw[target_basis] = statevector[full_index]

    success_probability = np.vdot(raw, raw).real

    if success_probability < 1e-14:
        raise ValueError("HHL success probability too small.")

    return normalize_state(raw / np.sqrt(success_probability)), float(success_probability)


def solve_hhl(A_original, b_original, verbose=False):
    """
    Solve A x = b using HHL.

    If A is not 2^n, it is padded:
        13 x 13 -> 16 x 16
    """

    A_pad, b_pad, original_dim = pad_linear_system(A_original, b_original)

    n_qubits = infer_n_qubits_from_matrix(A_pad)
    dim = 2 ** n_qubits

    eigvals = np.linalg.eigvalsh(A_pad)
    max_abs_eig = np.max(np.abs(eigvals))

    scale_factor = HHL_TARGET_MAX_EIGENVALUE / max_abs_eig
    A_hhl = A_pad * scale_factor

    eig_hhl = np.linalg.eigvalsh(A_hhl)
    nonzero = np.abs(eig_hhl[np.abs(eig_hhl) > 1e-12])
    C = min(1.0, 0.95 * np.min(nonzero))

    b_norm = normalize_state(b_pad)

    phase_qubits = HHL_PHASE_QUBITS
    total_qubits = phase_qubits + n_qubits + 1

    phase_indices = list(range(phase_qubits))
    target_indices = list(range(phase_qubits, phase_qubits + n_qubits))
    ancilla_index = total_qubits - 1

    U = expm(1j * A_hhl * HHL_TIME)

    qc = QuantumCircuit(total_qubits, name="HHL")
    qc.initialize(b_norm, target_indices)

    qc.barrier(label="QPE")
    qpe_circuit(qc, U, phase_indices, target_indices)

    qc.barrier(label="AQE")
    control_rotation_gate(qc, phase_indices, ancilla_index, HHL_TIME, C)

    qc.barrier(label="QPE†")
    qpe_only = QuantumCircuit(phase_qubits + n_qubits)
    qpe_circuit(
        qpe_only,
        U,
        list(range(phase_qubits)),
        list(range(phase_qubits, phase_qubits + n_qubits)),
    )
    qc.append(qpe_only.inverse(), phase_indices + target_indices)

    sv = Statevector.from_instruction(qc)

    x_state, success_probability = extract_hhl_amplitude_state(
        sv.data,
        phase_qubits=phase_qubits,
        n_system_qubits=n_qubits,
        ancilla_index=ancilla_index,
    )

    x_state = project_real_direction(x_state)

    x_pad, k = recover_scaled_solution(A_pad, b_pad, x_state)
    x = clean_real_vector(x_pad[:original_dim])

    if verbose:
        print("\n" + "=" * 80)
        print("HHL SOLVER")
        print("=" * 80)
        print(f"Original dimension    = {original_dim}")
        print(f"Padded dimension      = {A_pad.shape[0]}")
        print(f"System qubits         = {n_qubits}")
        print(f"Phase qubits          = {phase_qubits}")
        print(f"time                  = {HHL_TIME}")
        print(f"scale factor          = {scale_factor:.8e}")
        print(f"rotation constant     = {C:.8e}")
        print(f"success probability   = {success_probability:.8e}")
        print(f"k                     = {k}")

    return x, {
        "success_probability": success_probability,
        "k": k,
    }



# ============================================================
# 6B. Starter DC-OPF with primal-dual IPM
#     Quantum hook: replace KKT solve by Classical / VQLS / HHL
# ============================================================

OPF_MAX_ITER = 80               # increased to test convergence down to 1e-6
OPF_QUANTUM_MAX_ITER = 30       # quantum OPF outer iterations; lower if VQLS/HHL are slow
OPF_TOL = 1e-6
OPF_SIGMA = 0.1
OPF_FRACTION_TO_BOUNDARY = 0.99
OPF_KKT_REG = 1e-9

# Start with False. If True, KKT becomes much larger.
OPF_INCLUDE_LINE_LIMITS = False

# Start with False. Turn on only after Classical OPF is stable.
RUN_QUANTUM_OPF = True
QUANTUM_OPF_SOLVERS = ["HHL"]   # fast test: ["HHL"]. Try ["VQLS"] after HHL works.


def make_dc_bbus_bf(case):
    """
    Build DC Bbus and branch-flow matrix Bf.

    DC approximation:
        Pbus = Bbus @ theta
        Pft  = Bf @ theta
    """

    nbus = case["nbus"]
    branch = case["branch"]
    bus_id_to_idx = case["bus_id_to_idx"]

    Bbus = np.zeros((nbus, nbus), dtype=float)
    bf_rows = []
    rates = []

    for br in branch:
        if int(br[BR_STATUS]) == 0:
            continue

        x = float(br[BR_X])
        if abs(x) < 1e-14:
            continue

        f = bus_id_to_idx[int(br[F_BUS])]
        t = bus_id_to_idx[int(br[T_BUS])]

        tap = float(br[TAP])
        if abs(tap) < 1e-14:
            tap = 1.0

        b = 1.0 / x

        Bbus[f, f] += b / (tap * tap)
        Bbus[t, t] += b
        Bbus[f, t] -= b / tap
        Bbus[t, f] -= b / tap

        row = np.zeros(nbus, dtype=float)
        row[f] = b / tap
        row[t] = -b
        bf_rows.append(row)

        rate_a = float(br[RATE_A])
        rates.append(np.inf if rate_a <= 0 else rate_a / case["base_mva"])

    if len(bf_rows) == 0:
        return Bbus, np.zeros((0, nbus), dtype=float), np.zeros(0, dtype=float)

    return Bbus, np.vstack(bf_rows), np.asarray(rates, dtype=float)


def get_online_generators(case):
    online = np.where(case["gen"][:, GEN_STATUS].astype(int) > 0)[0]
    if len(online) == 0:
        raise ValueError("No online generator found.")
    return list(online)


def make_generator_connection(case, online_gens):
    """Cg[i, j] = 1 if online generator j is connected to bus i."""

    Cg = np.zeros((case["nbus"], len(online_gens)), dtype=float)

    for local_j, gen_idx in enumerate(online_gens):
        bus_ext = int(case["gen"][gen_idx, GEN_BUS])
        bus_int = case["bus_id_to_idx"][bus_ext]
        Cg[bus_int, local_j] = 1.0

    return Cg


def make_quadratic_cost_pu(case, online_gens):
    """
    MATPOWER cost uses MW:
        c2*PG_MW^2 + c1*PG_MW + c0

    OPF variable here uses p.u.:
        PG_MW = baseMVA * PG_pu
    """

    base = case["base_mva"]
    gencost = case["gencost"]

    Hdiag = []
    cvec = []
    const = 0.0

    for gen_idx in online_gens:
        row = gencost[gen_idx]
        model = int(row[GENCOST_MODEL])
        ncost = int(row[GENCOST_NCOST])

        if model != 2 or ncost != 3:
            raise ValueError("This starter OPF expects quadratic MATPOWER gencost rows.")

        c2 = float(row[GENCOST_C2])
        c1 = float(row[GENCOST_C1])
        c0 = float(row[GENCOST_C0])

        # Objective is 0.5*x^T H x + c^T x.
        Hdiag.append(2.0 * c2 * base * base)
        cvec.append(c1 * base)
        const += c0

    return np.asarray(Hdiag), np.asarray(cvec), float(const)


def economic_dispatch_initial_pg(case, online_gens):
    """Simple initial Pg in p.u. inside generator bounds."""

    base = case["base_mva"]
    total_load = np.sum(case["bus"][:, PD]) / base

    gen = case["gen"]
    pmin = gen[online_gens, PMIN] / base
    pmax = gen[online_gens, PMAX] / base

    pg = gen[online_gens, PG] / base
    pg = np.clip(pg, pmin + 1e-4, pmax - 1e-4)

    for _ in range(50):
        gap = total_load - np.sum(pg)
        if abs(gap) < 1e-10:
            break

        if gap > 0:
            room = pmax - pg
        else:
            room = pg - pmin

        free = room > 1e-10
        if not np.any(free):
            break

        weights = room[free] / np.sum(room[free])
        pg[free] += gap * weights
        pg = np.clip(pg, pmin + 1e-4, pmax - 1e-4)

    return pg


def build_dc_opf_qp(case, include_line_limits=False):
    """
    Build convex DC-OPF QP:
        minimize    0.5 x^T H x + c^T x
        subject to  Aeq x = beq
                    G x <= h

    Variables:
        x = [theta_non_slack, Pg_online]
    """

    base = case["base_mva"]
    bus = case["bus"]
    gen = case["gen"]
    nbus = case["nbus"]
    non_slack = case["non_slack"]

    online_gens = get_online_generators(case)
    ng = len(online_gens)
    nt = len(non_slack)
    nx = nt + ng

    Bbus, Bf, rates = make_dc_bbus_bf(case)
    Cg = make_generator_connection(case, online_gens)
    Hdiag_pg, c_pg, cost_const = make_quadratic_cost_pu(case, online_gens)

    H = np.zeros((nx, nx), dtype=float)
    c = np.zeros(nx, dtype=float)

    # Tiny regularization for theta variables.
    H[:nt, :nt] = np.eye(nt) * 1e-10

    for j in range(ng):
        H[nt + j, nt + j] = max(Hdiag_pg[j], 1e-10)
        c[nt + j] = c_pg[j]

    Pd_pu = bus[:, PD] / base

    # Bbus*theta - Cg*Pg = -Pd
    Aeq = np.zeros((nbus, nx), dtype=float)
    Aeq[:, :nt] = Bbus[:, non_slack]
    Aeq[:, nt:] = -Cg
    beq = -Pd_pu

    G_rows = []
    h_vals = []

    pmin = gen[online_gens, PMIN] / base
    pmax = gen[online_gens, PMAX] / base

    # Pg <= Pmax
    for j in range(ng):
        row = np.zeros(nx, dtype=float)
        row[nt + j] = 1.0
        G_rows.append(row)
        h_vals.append(pmax[j])

    # -Pg <= -Pmin
    for j in range(ng):
        row = np.zeros(nx, dtype=float)
        row[nt + j] = -1.0
        G_rows.append(row)
        h_vals.append(-pmin[j])

    if include_line_limits and Bf.shape[0] > 0:
        Bf_red = Bf[:, non_slack]

        for k in range(Bf_red.shape[0]):
            if not np.isfinite(rates[k]):
                continue

            row = np.zeros(nx, dtype=float)
            row[:nt] = Bf_red[k]
            G_rows.append(row)
            h_vals.append(rates[k])

            row = np.zeros(nx, dtype=float)
            row[:nt] = -Bf_red[k]
            G_rows.append(row)
            h_vals.append(rates[k])

    G = np.vstack(G_rows)
    h = np.asarray(h_vals, dtype=float)

    x0 = np.zeros(nx, dtype=float)
    pg0 = economic_dispatch_initial_pg(case, online_gens)
    x0[nt:] = pg0

    inj = Cg @ pg0 - Pd_pu
    Bred = Bbus[np.ix_(non_slack, non_slack)]

    try:
        theta0 = np.linalg.solve(Bred, inj[non_slack])
    except np.linalg.LinAlgError:
        theta0 = np.linalg.lstsq(Bred, inj[non_slack], rcond=None)[0]

    x0[:nt] = theta0

    return {
        "H": H,
        "c": c,
        "Aeq": Aeq,
        "beq": beq,
        "G": G,
        "h": h,
        "x0": x0,
        "online_gens": online_gens,
        "non_slack": non_slack,
        "Bbus": Bbus,
        "Bf": Bf,
        "rates": rates,
        "cost_const": cost_const,
        "nt": nt,
        "ng": ng,
        "nx": nx,
    }


def split_opf_kkt_delta(delta, nx, m_ineq, n_eq):
    p1 = nx
    p2 = p1 + m_ineq
    p3 = p2 + n_eq
    p4 = p3 + m_ineq
    return delta[:p1], delta[p1:p2], delta[p2:p3], delta[p3:p4]


def fraction_to_boundary_step(v, dv, tau=0.99):
    negative = dv < 0
    if not np.any(negative):
        return 1.0
    return min(1.0, tau * np.min(-v[negative] / dv[negative]))


def solve_symmetric_preconditioned_kkt(K, rhs, linear_solver, reg=OPF_KKT_REG):
    """
    Symmetric diagonal preconditioning:
        Kp = D K D
        rp = D rhs
        solve Kp y = rp
        delta = D y

    This preserves symmetry better than left-only preconditioning.
    """

    K = np.asarray(K, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if np.linalg.norm(rhs) < 1e-14:
        return np.zeros_like(rhs)

    K = 0.5 * (K + K.T)
    K = K + reg * np.eye(K.shape[0])

    row_norm = np.sum(np.abs(K), axis=1)
    row_norm = np.maximum(row_norm, 1e-12)
    D = 1.0 / np.sqrt(row_norm)

    Kp = (D[:, None] * K) * D[None, :]
    rp = D * rhs

    y = linear_solver(Kp, rp)
    y = clean_real_vector(y)
    return clean_real_vector(D * y)


def dc_opf_primal_dual_ipm(
    case,
    linear_solver,
    name="DC-OPF",
    include_line_limits=False,
    max_iter=OPF_MAX_ITER,
    tol=OPF_TOL,
    sigma=OPF_SIGMA,
    verbose=True,
):
    """Starter primal-dual IPM for DC-OPF."""

    data = build_dc_opf_qp(case, include_line_limits=include_line_limits)

    H = data["H"]
    c = data["c"]
    Aeq = data["Aeq"]
    beq = data["beq"]
    G = data["G"]
    h = data["h"]

    x = data["x0"].copy()

    nx = data["nx"]
    n_eq = Aeq.shape[0]
    m_ineq = G.shape[0]

    z = h - G @ x
    z = np.maximum(z, 1.0)
    lmbda = np.zeros(n_eq, dtype=float)
    mu = np.ones(m_ineq, dtype=float)

    history = []

    for it in range(max_iter):
        rx = H @ x + c + Aeq.T @ lmbda + G.T @ mu
        rh = Aeq @ x - beq
        rg = G @ x + z - h

        gap = float(np.dot(z, mu) / m_ineq)
        gamma = sigma * gap
        rz = mu - gamma / z

        gradcond = np.linalg.norm(rx, ord=np.inf) / (
            1.0 + max(np.linalg.norm(lmbda, ord=np.inf), np.linalg.norm(mu, ord=np.inf))
        )
        feascond = max(np.linalg.norm(rh, ord=np.inf), np.linalg.norm(rg, ord=np.inf)) / (
            1.0 + max(np.linalg.norm(beq, ord=np.inf), np.linalg.norm(h, ord=np.inf))
        )
        compcond = gap
        cost_pu_objective = 0.5 * float(x @ H @ x) + float(c @ x) + data["cost_const"]

        history.append({
            "iteration": it,
            "cost": cost_pu_objective,
            "gradcond": float(gradcond),
            "feascond": float(feascond),
            "compcond": float(compcond),
            "kkt_dim": int(nx + m_ineq + n_eq + m_ineq),
        })

        if verbose:
            print(
                f"{name:18s} | iter={it:02d} | cost={cost_pu_objective:.6f} | "
                f"grad={gradcond:.3e} | feas={feascond:.3e} | gap={compcond:.3e}"
            )

        if max(gradcond, feascond, compcond) < tol:
            break

        K11 = H
        K12 = np.zeros((nx, m_ineq))
        K13 = Aeq.T
        K14 = G.T

        K21 = np.zeros((m_ineq, nx))
        K22 = np.diag(mu / z)
        K23 = np.zeros((m_ineq, n_eq))
        K24 = np.eye(m_ineq)

        K31 = Aeq
        K32 = np.zeros((n_eq, m_ineq))
        K33 = np.zeros((n_eq, n_eq))
        K34 = np.zeros((n_eq, m_ineq))

        K41 = G
        K42 = np.eye(m_ineq)
        K43 = np.zeros((m_ineq, n_eq))
        K44 = np.zeros((m_ineq, m_ineq))

        KKT = np.block([
            [K11, K12, K13, K14],
            [K21, K22, K23, K24],
            [K31, K32, K33, K34],
            [K41, K42, K43, K44],
        ])

        rhs = -np.concatenate([rx, rz, rh, rg])

        delta = solve_symmetric_preconditioned_kkt(
            KKT,
            rhs,
            linear_solver=linear_solver,
        )

        dx, dz, dlmbda, dmu = split_opf_kkt_delta(delta, nx, m_ineq, n_eq)

        alpha_p = fraction_to_boundary_step(z, dz, OPF_FRACTION_TO_BOUNDARY)
        alpha_d = fraction_to_boundary_step(mu, dmu, OPF_FRACTION_TO_BOUNDARY)

        # Quantum/simulator directions may be noisier, so damp them.
        if linear_solver is not classical_p_solver:
            alpha_p = min(alpha_p, 0.5)
            alpha_d = min(alpha_d, 0.5)

        x = x + alpha_p * dx
        z = z + alpha_p * dz
        lmbda = lmbda + alpha_d * dlmbda
        mu = mu + alpha_d * dmu

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.all(np.isfinite(lmbda)) and np.all(np.isfinite(mu))):
            raise FloatingPointError(f"{name} produced non-finite values.")

        z = np.maximum(z, 1e-12)
        mu = np.maximum(mu, 1e-12)

    nt = data["nt"]
    ng = data["ng"]
    non_slack = data["non_slack"]
    online_gens = data["online_gens"]

    theta = np.zeros(case["nbus"], dtype=float)
    theta[non_slack] = x[:nt]

    pg_pu = x[nt:nt + ng]
    pg_mw = pg_pu * case["base_mva"]

    flows_pu = data["Bf"] @ theta
    flows_mw = flows_pu * case["base_mva"]

    Cg = make_generator_connection(case, online_gens)
    balance = data["Bbus"] @ theta - Cg @ pg_pu + case["bus"][:, PD] / case["base_mva"]

    pmin_mw = case["gen"][online_gens, PMIN]
    pmax_mw = case["gen"][online_gens, PMAX]
    gen_violation = np.maximum(pg_mw - pmax_mw, 0.0) + np.maximum(pmin_mw - pg_mw, 0.0)

    if data["Bf"].shape[0] > 0:
        finite_rates = np.isfinite(data["rates"])
        if np.any(finite_rates):
            line_violation = np.maximum(np.abs(flows_pu[finite_rates]) - data["rates"][finite_rates], 0.0)
            max_line_violation_mw = float(np.max(line_violation) * case["base_mva"])
        else:
            max_line_violation_mw = 0.0
    else:
        max_line_violation_mw = 0.0

    # Report MATPOWER-style cost in dollars/hour units.
    true_cost = 0.0
    for j, gen_idx in enumerate(online_gens):
        cost_row = case["gencost"][gen_idx]
        true_cost += (
            cost_row[GENCOST_C2] * pg_mw[j] ** 2
            + cost_row[GENCOST_C1] * pg_mw[j]
            + cost_row[GENCOST_C0]
        )

    return {
        "name": name,
        "theta_rad": theta,
        "theta_deg": np.rad2deg(theta),
        "online_gens": online_gens,
        "pg_pu": pg_pu,
        "pg_mw": pg_mw,
        "flows_mw": flows_mw,
        "cost": float(true_cost),
        "balance_inf": float(np.linalg.norm(balance, ord=np.inf)),
        "max_gen_violation_mw": float(np.max(gen_violation)),
        "max_line_violation_mw": max_line_violation_mw,
        "history": history,
        "iterations": len(history),
        "converged": max(history[-1]["gradcond"], history[-1]["feascond"], history[-1]["compcond"]) < tol,
        "include_line_limits": include_line_limits,
        "kkt_dim": history[0]["kkt_dim"],
    }


def print_dc_opf_result(result):
    print("\n" + "=" * 80)
    print(result["name"])
    print("=" * 80)
    print(f"converged             = {result['converged']}")
    print(f"iterations            = {result['iterations']}")
    print(f"KKT dimension         = {result['kkt_dim']}  -> padded to {next_power_of_two(result['kkt_dim'])}")
    print(f"cost                  = {result['cost']:.6f}")
    print(f"balance_inf           = {result['balance_inf']:.6e}")
    print(f"max_gen_violation_mw  = {result['max_gen_violation_mw']:.6e}")
    print(f"max_line_violation_mw = {result['max_line_violation_mw']:.6e}")
    print(f"include_line_limits   = {result['include_line_limits']}")

    print("\nGenerator dispatch Pg MW:")
    for local_j, gen_idx in enumerate(result["online_gens"]):
        print(f"  gen row {gen_idx:2d}: Pg = {result['pg_mw'][local_j]: .6f} MW")

    print("\nVoltage angles theta degrees:")
    print(result["theta_deg"])


def plot_opf_gradcond(opf_results):
    plt.figure(figsize=(8, 5))

    for result in opf_results:
        grad = [item["gradcond"] for item in result["history"]]
        plt.plot(grad, marker="o", linewidth=2, label=result["name"])

    plt.yscale("log")
    plt.xlabel("OPF IPM iteration")
    plt.ylabel("gradcond")
    plt.title("Starter DC-OPF gradcond convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "starter_dc_opf_gradcond.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def run_starter_dc_opf_comparison(case):
    print("\n" + "=" * 80)
    print("STARTER DC-OPF: CLASSICAL / SELECTED QUANTUM KKT SOLVERS")
    print("=" * 80)
    print(f"case_name               = {case.get('case_name', 'unknown')}")
    print(f"nbus                    = {case['nbus']}")
    print(f"OPF_MAX_ITER            = {OPF_MAX_ITER}")
    print(f"OPF_QUANTUM_MAX_ITER    = {OPF_QUANTUM_MAX_ITER}")
    print(f"OPF_TOL                 = {OPF_TOL}")
    print(f"OPF_INCLUDE_LINE_LIMITS = {OPF_INCLUDE_LINE_LIMITS}")
    print(f"RUN_QUANTUM_OPF         = {RUN_QUANTUM_OPF}")
    print(f"QUANTUM_OPF_SOLVERS     = {QUANTUM_OPF_SOLVERS}")

    result_classical = dc_opf_primal_dual_ipm(
        case,
        linear_solver=classical_p_solver,
        name="Classical DC-OPF",
        include_line_limits=OPF_INCLUDE_LINE_LIMITS,
        max_iter=OPF_MAX_ITER,
        tol=OPF_TOL,
        verbose=True,
    )

    results = [result_classical]

    if RUN_QUANTUM_OPF:
        if "VQLS" in QUANTUM_OPF_SOLVERS:
            result_vqls = dc_opf_primal_dual_ipm(
                case,
                linear_solver=vqls_p_solver,
                name="VQLS DC-OPF",
                include_line_limits=OPF_INCLUDE_LINE_LIMITS,
                max_iter=OPF_QUANTUM_MAX_ITER,
                tol=OPF_TOL,
                verbose=True,
            )
            results.append(result_vqls)

        if "HHL" in QUANTUM_OPF_SOLVERS:
            result_hhl = dc_opf_primal_dual_ipm(
                case,
                linear_solver=hhl_p_solver,
                name="HHL DC-OPF",
                include_line_limits=OPF_INCLUDE_LINE_LIMITS,
                max_iter=OPF_QUANTUM_MAX_ITER,
                tol=OPF_TOL,
                verbose=True,
            )
            results.append(result_hhl)

    for result in results:
        print_dc_opf_result(result)

    plot_opf_gradcond(results)
    return results


# ============================================================
# 7. True FDLS
# ============================================================

def fdls_power_flow(case, Ybus, Bprime, Bdouble, p_solver, name):
    """
    True FDLS loop:

        P-theta step:
            B' Δθ = ΔP / |V|

        Q-V step:
            B'' ΔV = ΔQ / |V|

    Here:
        P-theta uses Classical / VQLS / HHL.
        Q-V uses classical solve because B'' is 9x9 in IEEE 14 and
        this script focuses quantum solvers on the P-theta matrix.
    """

    non_slack = case["non_slack"]
    pq = case["pq"]
    slack = case["slack"]

    P_spec, Q_spec = make_specified_power(case)
    vm, va = initial_voltage(case)

    loss_history = []

    for iteration in range(FDLS_MAX_ITER):
        P_calc, Q_calc = calculated_power(Ybus, vm, va)

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        mismatch_p = dP[non_slack]
        mismatch_q = dQ[pq]

        loss = np.sqrt(
            np.sum(mismatch_p ** 2) +
            np.sum(mismatch_q ** 2)
        )

        loss_history.append(float(loss))

        if loss < FDLS_TOL:
            break

        rhs_p = mismatch_p / vm[non_slack]
        dtheta = p_solver(Bprime, rhs_p)
        dtheta = clean_real_vector(dtheta)

        va[non_slack] += dtheta

        rhs_q = mismatch_q / vm[pq]
        dV = np.linalg.solve(Bdouble, rhs_q)

        vm[pq] += dV

        # Slack fixed
        va[slack] = np.deg2rad(case["bus"][slack, VA])
        vm[slack] = case["bus"][slack, VM]

        # PV and REF voltage magnitudes fixed
        for idx in case["pv"]:
            vm[idx] = case["bus"][idx, VM]

    P_final, Q_final = calculated_power(Ybus, vm, va)

    final_dP = P_spec - P_final
    final_dQ = Q_spec - Q_final

    final_loss = np.sqrt(
        np.sum(final_dP[non_slack] ** 2) +
        np.sum(final_dQ[pq] ** 2)
    )

    return {
        "name": name,
        "vm": vm,
        "va": va,
        "va_deg": np.rad2deg(va),
        "loss_history": loss_history,
        "final_loss": float(final_loss),
        "converged": final_loss < FDLS_TOL,
        "iterations": len(loss_history),
        "P_calc": P_final,
        "Q_calc": Q_final,
    }


def classical_p_solver(A, b):
    return solve_classical(A, b)


def vqls_p_solver(A, b):
    x, _ = solve_vqls_no_cz(A, b, verbose=False)
    return x


def hhl_p_solver(A, b):
    x, _ = solve_hhl(A, b, verbose=False)
    return x


# ============================================================
# 8. Plots
# ============================================================

def plot_one_step_solution(non_slack_buses, x_classical, x_vqls, x_hhl):
    labels = [str(i + 1) for i in non_slack_buses]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, x_classical, width, label="Classical")
    plt.bar(x, x_vqls, width, label="VQLS no-CZ")
    plt.bar(x + width, x_hhl, width, label="HHL")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Initial Δθ solution")
    plt.title("One-step P-theta solution from IEEE 14-bus matrix")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "one_step_delta_theta_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_fdls_loss(results):
    plt.figure(figsize=(8, 5))

    for result in results:
        plt.plot(
            result["loss_history"],
            marker="o",
            linewidth=2,
            label=result["name"],
        )

    plt.yscale("log")
    plt.xlabel("FDLS iteration")
    plt.ylabel("Loss = sqrt(||ΔP||² + ||ΔQ||²)")
    plt.title("IEEE 14-bus FDLS convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_fdls_loss_comparison.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_final_angle_solution(results, nbus):
    labels = [str(i + 1) for i in range(nbus)]
    x = np.arange(nbus)
    width = 0.25

    plt.figure(figsize=(12, 5))

    for k, result in enumerate(results):
        plt.bar(
            x + (k - 1) * width,
            result["va_deg"],
            width,
            label=result["name"],
        )

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Voltage angle θ (degrees)")
    plt.title("IEEE 14-bus final voltage angle solution")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_final_angle_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_final_voltage_solution(results, nbus):
    labels = [str(i + 1) for i in range(nbus)]
    x = np.arange(nbus)
    width = 0.25

    plt.figure(figsize=(12, 5))

    for k, result in enumerate(results):
        plt.bar(
            x + (k - 1) * width,
            result["vm"],
            width,
            label=result["name"],
        )

    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Voltage magnitude |V|")
    plt.title("IEEE 14-bus final voltage magnitude solution")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_final_voltage_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


# ============================================================
# 9. Main
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("SMALL-CASE POWER FLOW / QOPF CONVERGENCE TEST")
    print("=" * 80)
    print(f"POWER_CASE = {POWER_CASE}")
    print(f"RUN_FDLS   = {RUN_FDLS}")
    print(f"RUN_OPF    = {RUN_OPF}")

    case = load_power_case(POWER_CASE)
    Ybus = make_ybus(case)
    Bprime, Bdouble = make_fdls_matrices(Ybus, case)

    P_spec, Q_spec = make_specified_power(case)
    vm0, va0 = initial_voltage(case)
    P_calc0, Q_calc0 = calculated_power(Ybus, vm0, va0)

    non_slack = case["non_slack"]
    rhs_p0 = (P_spec - P_calc0)[non_slack] / vm0[non_slack]

    export_matrices(Ybus, Bprime, Bdouble, rhs_p0, case)

    print("\nCase information")
    print("-" * 80)
    print(f"case_name    = {case.get('case_name', 'unknown')}")
    print(f"baseMVA      = {case['base_mva']}")
    print(f"nbus         = {case['nbus']}")
    print(f"ngen         = {case['gen'].shape[0]}")
    print(f"nbranch      = {case['branch'].shape[0]}")
    print(f"slack index  = {case['slack']}  bus number {case['slack'] + 1}")
    print(f"PV indices   = {case['pv']}")
    print(f"PQ indices   = {case['pq']}")
    print(f"non-slack    = {case['non_slack']}")

    print("\nMatrix sizes")
    print("-" * 80)
    print(f"Ybus shape        = {Ybus.shape}")
    print(f"Bprime shape      = {Bprime.shape}")
    print(f"Bdouble shape     = {Bdouble.shape}")
    print(f"Initial rhs shape = {rhs_p0.shape}")
    print(f"Quantum FDLS pad  = {Bprime.shape[0]} -> {next_power_of_two(Bprime.shape[0])}")

    if RUN_FDLS:
        print("\n" + "=" * 80)
        print("ONE-STEP P-THETA MATRIX SOLVE")
        print("=" * 80)

        x_classical = solve_classical(Bprime, rhs_p0)
        x_vqls, vqls_info = solve_vqls_no_cz(Bprime, rhs_p0, verbose=True)
        x_hhl, hhl_info = solve_hhl(Bprime, rhs_p0, verbose=True)

        vqls_metrics = vector_metrics(Bprime, rhs_p0, x_vqls, x_classical)
        hhl_metrics = vector_metrics(Bprime, rhs_p0, x_hhl, x_classical)

        print("\nOne-step metrics vs classical")
        print("-" * 80)
        print(f"VQLS relative residual = {vqls_metrics['rel_residual']:.6e}")
        print(f"HHL  relative residual = {hhl_metrics['rel_residual']:.6e}")
        print(f"HHL success prob       = {hhl_info['success_probability']:.6e}")

        print("\n" + "=" * 80)
        print("RUNNING FDLS")
        print("=" * 80)
        print(f"FDLS_MAX_ITER = {FDLS_MAX_ITER}")
        print(f"FDLS_TOL      = {FDLS_TOL}")

        result_classical = fdls_power_flow(case, Ybus, Bprime, Bdouble, classical_p_solver, "Classical FDLS")
        result_vqls = fdls_power_flow(case, Ybus, Bprime, Bdouble, vqls_p_solver, "VQLS no-CZ FDLS")
        result_hhl = fdls_power_flow(case, Ybus, Bprime, Bdouble, hhl_p_solver, "HHL FDLS")

        results = [result_classical, result_vqls, result_hhl]

        for result in results:
            print("\n" + result["name"])
            print("-" * 80)
            print(f"converged  = {result['converged']}")
            print(f"iterations = {result['iterations']}")
            print(f"final loss = {result['final_loss']:.6e}")

        plot_one_step_solution(non_slack, x_classical, x_vqls, x_hhl)
        plot_fdls_loss(results)
        plot_final_angle_solution(results, case["nbus"])
        plot_final_voltage_solution(results, case["nbus"])

    if RUN_OPF:
        opf_results = run_starter_dc_opf_comparison(case)

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Text result file: {RESULT_TEXT_FILE}")
    print(f"Matrix directory: {MATRIX_DIR}")
    print("Plots are saved inside OUTPUT_DIR.")


if __name__ == "__main__":
    with open(RESULT_TEXT_FILE, "w", encoding="utf-8") as result_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, result_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout
