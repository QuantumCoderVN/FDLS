import os
import numpy as np
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    QUBO_BITS_PER_VAR,
    QUBO_NUM_READS,
    QUBO_SWEEPS,
    QUBO_SEED,
    QUBO_TEMP_START,
    QUBO_TEMP_END,
    QUBO_RANGE_FACTOR,
    QUBO_MIN_ABS_RANGE,
    QUBO_MAX_ABS_RANGE,
    QUBO_REFINEMENT_STEPS,
    QUBO_REFINEMENT_SHRINK,
    QUBO_DAMPING,
    QUBO_MAX_STEP_NORM_FACTOR,
)

from math_utils import clean_real_vector


def estimate_symmetric_bound(A, b):
    """
    Estimate a small search range for correction dx.

    For Ax=b:
        ||x|| <= ||b|| / lambda_min(A)

    QUBO should not search a huge interval.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    b_norm = np.linalg.norm(b)

    if b_norm < 1e-14:
        return QUBO_MIN_ABS_RANGE

    eigvals = np.linalg.eigvalsh(0.5 * (A + A.T))
    nonzero = np.abs(eigvals[np.abs(eigvals) > 1e-12])

    if len(nonzero) == 0:
        return QUBO_MAX_ABS_RANGE

    lambda_min = np.min(nonzero)
    bound = QUBO_RANGE_FACTOR * b_norm / lambda_min

    return float(np.clip(bound, QUBO_MIN_ABS_RANGE, QUBO_MAX_ABS_RANGE))


def build_binary_encoding(n_vars, bits_per_var, lower_bound, upper_bound):
    """
    x = offset + E z

    z in {0,1}
    """

    n_binary = n_vars * bits_per_var

    offset = np.full(n_vars, lower_bound, dtype=float)
    E = np.zeros((n_vars, n_binary), dtype=float)

    levels = 2**bits_per_var - 1
    step = (upper_bound - lower_bound) / levels

    for i in range(n_vars):
        for k in range(bits_per_var):
            col = i * bits_per_var + k
            E[i, col] = step * (2**k)

    return offset, E, step


def build_qubo_from_least_squares(A, b, bits_per_var, bound):
    """
    Build QUBO for:

        min_x ||A x - b||^2

    with:

        x = offset + E z

    Expand:

        ||A(offset + Ez) - b||^2
        = z^T Q z + const

    Linear terms are placed on Q diagonal because z_i^2 = z_i.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n_vars = A.shape[1]

    lower_bound = -bound
    upper_bound = bound

    offset, E, step = build_binary_encoding(
        n_vars=n_vars,
        bits_per_var=bits_per_var,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    c = A @ offset - b

    Q = E.T @ A.T @ A @ E
    linear = 2.0 * (E.T @ A.T @ c)

    Q = 0.5 * (Q + Q.T)

    for i in range(Q.shape[0]):
        Q[i, i] += linear[i]

    scale = np.max(np.abs(Q))

    if scale < 1e-14:
        scale = 1.0

    Q_scaled = Q / scale

    return Q_scaled, {
        "offset": offset,
        "E": E,
        "step": step,
        "bound": bound,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "scale": scale,
        "Q_scaled": Q_scaled,
    }


def qubo_energy(Q, z):
    z = np.asarray(z, dtype=float)
    return float(z @ Q @ z)


def greedy_local_search(Q, z):
    """
    Deterministic 1-bit improvement after annealing.
    This is very important for QUBO quality.
    """

    Q = 0.5 * (Q + Q.T)
    z = z.astype(float).copy()

    qz = Q @ z
    energy = float(z @ qz)

    improved = True

    while improved:
        improved = False

        for i in range(len(z)):
            zi = z[i]
            row_no_diag = qz[i] - Q[i, i] * zi

            if zi == 0:
                delta = Q[i, i] + 2.0 * row_no_diag
                flip_direction = 1.0
            else:
                delta = -Q[i, i] - 2.0 * row_no_diag
                flip_direction = -1.0

            if delta < -1e-14:
                z[i] = 1.0 - zi
                qz = qz + flip_direction * Q[:, i]
                energy += delta
                improved = True

    return z.astype(int), float(energy)


def simulated_annealing_qubo(
    Q,
    num_reads=QUBO_NUM_READS,
    sweeps=QUBO_SWEEPS,
    temp_start=QUBO_TEMP_START,
    temp_end=QUBO_TEMP_END,
    seed=QUBO_SEED,
):
    """
    Classical simulated annealing for QUBO.
    """

    Q = np.asarray(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)

    n = Q.shape[0]
    rng = np.random.default_rng(seed)

    best_z = None
    best_energy = float("inf")
    best_history = None

    for read in range(num_reads):
        z = rng.integers(0, 2, size=n).astype(float)

        qz = Q @ z
        energy = float(z @ qz)

        history = []

        for sweep in range(sweeps):
            frac = sweep / max(sweeps - 1, 1)
            temperature = temp_start * (temp_end / temp_start) ** frac

            for i in rng.permutation(n):
                zi = z[i]
                row_no_diag = qz[i] - Q[i, i] * zi

                if zi == 0:
                    delta = Q[i, i] + 2.0 * row_no_diag
                    flip_direction = 1.0
                else:
                    delta = -Q[i, i] - 2.0 * row_no_diag
                    flip_direction = -1.0

                if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-14)):
                    z[i] = 1.0 - zi
                    qz = qz + flip_direction * Q[:, i]
                    energy += delta

            history.append(float(energy))

        # Polishing
        z_int, polished_energy = greedy_local_search(Q, z.astype(int))

        if polished_energy < best_energy:
            best_energy = polished_energy
            best_z = z_int.copy()
            best_history = history

    return best_z.astype(int), best_energy, best_history


def decode_solution(z, offset, E):
    return offset + E @ z


def solve_qubo_once(A, b, bound, seed_shift=0):
    """
    Solve one bounded least-squares QUBO:

        min_dx ||A dx - b||^2
    """

    Q, meta = build_qubo_from_least_squares(
        A,
        b,
        bits_per_var=QUBO_BITS_PER_VAR,
        bound=bound,
    )

    z_best, energy_best, history = simulated_annealing_qubo(
        Q,
        num_reads=QUBO_NUM_READS,
        sweeps=QUBO_SWEEPS,
        temp_start=QUBO_TEMP_START,
        temp_end=QUBO_TEMP_END,
        seed=QUBO_SEED + seed_shift,
    )

    dx = decode_solution(
        z_best,
        meta["offset"],
        meta["E"],
    )

    return clean_real_vector(dx), {
        "z": z_best,
        "energy": energy_best,
        "energy_history": history,
        "Q_scaled": meta["Q_scaled"],
        "step": meta["step"],
        "bound": meta["bound"],
        "lower_bound": meta["lower_bound"],
        "upper_bound": meta["upper_bound"],
        "num_binary_variables": len(z_best),
        "bits_per_var": QUBO_BITS_PER_VAR,
    }


def solve_qubo_annealing(A_original, b_original, verbose=False):
    """
    Improved QUBO solver.

    Instead of solving x all at once, solve corrections:

        x <- x + damping * dx
        r = b - A x
        dx ≈ argmin ||A dx - r||^2

    This is much more stable for FDLS.
    """

    A = np.asarray(A_original, dtype=float)
    b = np.asarray(b_original, dtype=float)

    n = len(b)

    if np.linalg.norm(b) < 1e-14:
        return np.zeros_like(b), {
            "relative_residual": 0.0,
            "note": "zero rhs",
        }

    x = np.zeros(n, dtype=float)

    initial_bound = estimate_symmetric_bound(A, b)
    bound = initial_bound

    residual_history = []
    best_x = x.copy()
    best_rel = float("inf")
    best_info = None

    for ref_it in range(QUBO_REFINEMENT_STEPS):
        r = b - A @ x

        rel_r = np.linalg.norm(r) / max(np.linalg.norm(b), 1e-14)
        residual_history.append(float(rel_r))

        if rel_r < best_rel:
            best_rel = float(rel_r)
            best_x = x.copy()

        dx, info = solve_qubo_once(
            A,
            r,
            bound=bound,
            seed_shift=1000 * ref_it,
        )

        # Damping prevents FDLS from exploding.
        candidate_x = x + QUBO_DAMPING * dx
        candidate_rel = np.linalg.norm(A @ candidate_x - b) / max(np.linalg.norm(b), 1e-14)

        # Accept only if residual improves. Otherwise shrink search box.
        if candidate_rel < rel_r:
            x = candidate_x

            if candidate_rel < best_rel:
                best_rel = float(candidate_rel)
                best_x = x.copy()
                best_info = info
        else:
            bound *= QUBO_REFINEMENT_SHRINK

        bound = max(bound * QUBO_REFINEMENT_SHRINK, QUBO_MIN_ABS_RANGE)

    x = clean_real_vector(best_x)

    residual = np.linalg.norm(A @ x - b)
    rel_residual = residual / max(np.linalg.norm(b), 1e-14)

    if best_info is None:
        _, best_info = solve_qubo_once(A, b, bound=initial_bound)

    info = dict(best_info)
    info.update({
        "x": x,
        "residual": float(residual),
        "relative_residual": float(rel_residual),
        "residual_history": residual_history,
        "initial_bound": initial_bound,
        "final_bound": bound,
    })

    if verbose:
        print("\n" + "=" * 80)
        print("IMPROVED QUBO / ANNEALING SOLVER")
        print("=" * 80)
        print("Solves correction form: x <- x + dx")
        print("This is still classical simulated annealing, not D-Wave.")
        print(f"variables              = {n}")
        print(f"bits per variable      = {QUBO_BITS_PER_VAR}")
        print(f"binary variables       = {info['num_binary_variables']}")
        print(f"initial bound          = {initial_bound:.8e}")
        print(f"final bound            = {bound:.8e}")
        print(f"step last              = {info['step']:.8e}")
        print(f"relative residual      = {rel_residual:.8e}")
        print("residual history       =", residual_history)

    plt.figure(figsize=(7, 4))
    plt.plot(residual_history, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("QUBO refinement iteration")
    plt.ylabel("relative residual")
    plt.title("Improved QUBO residual refinement")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qubo_refinement_residual.png"), dpi=150)
    plt.close()

    if "energy_history" in info and info["energy_history"] is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(info["energy_history"], linewidth=2)
        plt.xlabel("Annealing sweep")
        plt.ylabel("QUBO energy")
        plt.title("QUBO simulated annealing energy")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "qubo_annealing_energy.png"), dpi=150)
        plt.close()

    return x, info