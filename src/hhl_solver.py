import numpy as np

from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, RYGate
from qiskit.quantum_info import Statevector

from config import (
    HHL_PHASE_QUBITS,
    HHL_TIME,
    HHL_TARGET_MAX_EIGENVALUE,
)

from math_utils import (
    pad_linear_system,
    normalize_state,
    recover_scaled_solution,
    project_real_direction,
    clean_real_vector,
    infer_n_qubits_from_matrix,
)


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