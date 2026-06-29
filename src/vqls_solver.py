import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator

from config import (
    OUTPUT_DIR,
    VQLS_LAYERS,
    VQLS_STEPS,
    VQLS_RESTARTS,
    VQLS_SEED,
    VQLS_Q_DELTA,
)

from math_utils import (
    pad_linear_system,
    normalize_state,
    recover_scaled_solution,
    project_real_direction,
    clean_real_vector,
    infer_n_qubits_from_matrix,
)


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

    No CZ gate is used.
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