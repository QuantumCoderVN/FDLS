import numpy as np


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def pad_linear_system(A, b):
    """
    Pad A x = b to dimension 2^n.

    Example:
        13 x 13 -> 16 x 16
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
    FDLS matrices and RHS are real.
    Use real direction to remove unnecessary complex phase.
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