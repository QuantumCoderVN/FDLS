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

from pypower.case14 import case14


warnings.filterwarnings("ignore", category=DeprecationWarning)

OUTPUT_DIR = "D:\1. QUANTUM MACHINE LEARNING\3. QUANTUM FOWER FLOW\fdls-5bus\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATRIX_DIR = os.path.join(OUTPUT_DIR, "matrices")
os.makedirs(MATRIX_DIR, exist_ok=True)

RESULT_TEXT_FILE = os.path.join(OUTPUT_DIR, "run_results.txt")


# ============================================================
# Settings
# ============================================================

FDLS_MAX_ITER = 12
FDLS_TOL = 1e-6

# VQLS no-CZ settings
VQLS_LAYERS = 3
VQLS_STEPS = 100
VQLS_RESTARTS = 5
VQLS_SEED = 0
VQLS_Q_DELTA = 0.001

# HHL settings
HHL_PHASE_QUBITS = 6
HHL_TIME = 0.6
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


# ============================================================
# 1. Load IEEE 14-bus data
# ============================================================

def load_ieee14_case():
    """
    Load IEEE 14-bus power-flow case from PYPOWER/MATPOWER case14.
    """

    mpc = case14()

    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)

    # Convert external bus numbers to internal indices 0..n-1
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
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "bus_id_to_idx": bus_id_to_idx,
        "nbus": nbus,
        "slack": slack,
        "pv": list(pv),
        "pq": list(pq),
        "non_slack": non_slack,
    }


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
    print("IEEE 14-BUS DATA + FDLS + CLASSICAL/VQLS/HHL")
    print("=" * 80)

    case = load_ieee14_case()
    Ybus = make_ybus(case)
    Bprime, Bdouble = make_fdls_matrices(Ybus, case)

    P_spec, Q_spec = make_specified_power(case)
    vm0, va0 = initial_voltage(case)
    P_calc0, Q_calc0 = calculated_power(Ybus, vm0, va0)

    non_slack = case["non_slack"]
    pq = case["pq"]

    rhs_p0 = (P_spec - P_calc0)[non_slack] / vm0[non_slack]

    export_matrices(Ybus, Bprime, Bdouble, rhs_p0, case)

    print("\nCase information")
    print("-" * 80)
    print(f"baseMVA      = {case['base_mva']}")
    print(f"nbus         = {case['nbus']}")
    print(f"slack index  = {case['slack']}  bus number {case['slack'] + 1}")
    print(f"PV indices   = {case['pv']}")
    print(f"PQ indices   = {case['pq']}")
    print(f"non-slack    = {case['non_slack']}")

    print("\nMatrix sizes")
    print("-" * 80)
    print(f"Ybus shape       = {Ybus.shape}")
    print(f"Bprime shape     = {Bprime.shape}")
    print(f"Bdouble shape    = {Bdouble.shape}")
    print(f"Initial rhs shape = {rhs_p0.shape}")
    print("Quantum solvers pad Bprime from 13x13 to 16x16.")

    print("\nBprime matrix:")
    print(Bprime)

    print("\nInitial rhs_p = ΔP/|V|:")
    print(rhs_p0)

    # ------------------------------------------------------------
    # One-step P-theta matrix solve
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ONE-STEP IEEE 14 P-THETA MATRIX SOLVE")
    print("=" * 80)

    x_classical = solve_classical(Bprime, rhs_p0)
    x_vqls, vqls_info = solve_vqls_no_cz(Bprime, rhs_p0, verbose=True)
    x_hhl, hhl_info = solve_hhl(Bprime, rhs_p0, verbose=True)

    print("\nClassical Δθ:")
    print(x_classical)

    print("\nVQLS no-CZ Δθ:")
    print(x_vqls)

    print("\nHHL Δθ:")
    print(x_hhl)

    print("\nSign comparison")
    print("Classical:", np.sign(x_classical))
    print("VQLS:     ", np.sign(x_vqls))
    print("HHL:      ", np.sign(x_hhl))

    vqls_metrics = vector_metrics(Bprime, rhs_p0, x_vqls, x_classical)
    hhl_metrics = vector_metrics(Bprime, rhs_p0, x_hhl, x_classical)

    print("\nOne-step metrics vs classical")
    print("-" * 80)
    print("VQLS no-CZ:")
    print(f"  residual          = {vqls_metrics['residual']:.6e}")
    print(f"  relative residual = {vqls_metrics['rel_residual']:.6e}")
    print(f"  relative error    = {vqls_metrics['rel_error']:.6e}")
    print(f"  fidelity          = {vqls_metrics['fidelity']:.8f}")

    print("HHL:")
    print(f"  residual          = {hhl_metrics['residual']:.6e}")
    print(f"  relative residual = {hhl_metrics['rel_residual']:.6e}")
    print(f"  relative error    = {hhl_metrics['rel_error']:.6e}")
    print(f"  fidelity          = {hhl_metrics['fidelity']:.8f}")
    print(f"  success prob      = {hhl_info['success_probability']:.6e}")

    # ------------------------------------------------------------
    # True FDLS with three P-theta solvers
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING TRUE IEEE 14 FDLS")
    print("=" * 80)

    result_classical = fdls_power_flow(
        case,
        Ybus,
        Bprime,
        Bdouble,
        classical_p_solver,
        "Classical FDLS",
    )

    result_vqls = fdls_power_flow(
        case,
        Ybus,
        Bprime,
        Bdouble,
        vqls_p_solver,
        "VQLS no-CZ FDLS",
    )

    result_hhl = fdls_power_flow(
        case,
        Ybus,
        Bprime,
        Bdouble,
        hhl_p_solver,
        "HHL FDLS",
    )

    results = [result_classical, result_vqls, result_hhl]

    for result in results:
        print("\n" + result["name"])
        print("-" * 80)
        print(f"converged  = {result['converged']}")
        print(f"iterations = {result['iterations']}")
        print(f"final loss = {result['final_loss']:.6e}")

        print("\nVoltage angles θ in degrees:")
        print(result["va_deg"])

        print("\nVoltage magnitudes |V|:")
        print(result["vm"])

    print("\n" + "=" * 80)
    print("FDLS FINAL SOLUTION ERROR VS CLASSICAL FDLS")
    print("=" * 80)

    ref_va = result_classical["va"]
    ref_vm = result_classical["vm"]

    for result in [result_vqls, result_hhl]:
        angle_error = np.linalg.norm(result["va"] - ref_va)
        voltage_error = np.linalg.norm(result["vm"] - ref_vm)

        print("\n" + result["name"])
        print("-" * 80)
        print(f"angle error   = {angle_error:.6e}")
        print(f"voltage error = {voltage_error:.6e}")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_one_step_solution(non_slack, x_classical, x_vqls, x_hhl)
    plot_fdls_loss(results)
    plot_final_angle_solution(results, case["nbus"])
    plot_final_voltage_solution(results, case["nbus"])

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Text result file: {RESULT_TEXT_FILE}")
    print("Matrix files:")
    print(f"  {MATRIX_DIR}/Ybus_real.csv")
    print(f"  {MATRIX_DIR}/Ybus_imag.csv")
    print(f"  {MATRIX_DIR}/Bprime.csv")
    print(f"  {MATRIX_DIR}/Bdouble_prime.csv")
    print(f"  {MATRIX_DIR}/rhs_p_initial.csv")
    print("Plots:")
    print("  outputs/one_step_delta_theta_solution.png")
    print("  outputs/ieee14_fdls_loss_comparison.png")
    print("  outputs/ieee14_final_angle_solution.png")
    print("  outputs/ieee14_final_voltage_solution.png")
    print("  outputs/vqls_no_cz_cost.png")


if __name__ == "__main__":
    with open(RESULT_TEXT_FILE, "w", encoding="utf-8") as result_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, result_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout