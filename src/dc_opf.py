import os
import numpy as np
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    RUN_QUANTUM_OPF,
    OPF_INCLUDE_LINE_LIMITS,
    OPF_MAX_ITER,
    OPF_QUANTUM_MAX_ITER,
    OPF_TOL,
    OPF_SIGMA,
    OPF_FRACTION_TO_BOUNDARY,
    OPF_KKT_REG,
    OPF_QUANTUM_FALLBACK_RELRES,
    OPF_MAX_QUANTUM_KKT_DIM,
)

from math_utils import clean_real_vector, next_power_of_two

from ieee14_case import (
    PD,
    GEN_STATUS,
    GEN_BUS,
    PG,
    F_BUS,
    T_BUS,
    BR_X,
    TAP,
    BR_STATUS,
    RATE_A,
    PMAX,
    PMIN,
    GENCOST_MODEL,
    GENCOST_NCOST,
    GENCOST_C2,
    GENCOST_C1,
    GENCOST_C0,
)


# ============================================================
# 1. DC network model for OPF
# ============================================================

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
    """
    Cg[i, j] = 1 if online generator j is connected to bus i.
    """

    Cg = np.zeros((case["nbus"], len(online_gens)), dtype=float)

    for local_j, gen_idx in enumerate(online_gens):
        bus_ext = int(case["gen"][gen_idx, GEN_BUS])
        bus_int = case["bus_id_to_idx"][bus_ext]
        Cg[bus_int, local_j] = 1.0

    return Cg


# ============================================================
# 2. Cost and initial point
# ============================================================

def make_quadratic_cost_pu(case, online_gens):
    """
    MATPOWER cost uses MW:

        c2 * PG_MW^2 + c1 * PG_MW + c0

    OPF variable here uses p.u.:

        PG_MW = baseMVA * PG_pu

    Objective form:

        0.5 x^T H x + c^T x + const
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

        Hdiag.append(2.0 * c2 * base * base)
        cvec.append(c1 * base)
        const += c0

    return np.asarray(Hdiag), np.asarray(cvec), float(const)


def economic_dispatch_initial_pg(case, online_gens):
    """
    Simple feasible-ish initial Pg in p.u.
    """

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


# ============================================================
# 3. Build DC-OPF quadratic program
# ============================================================

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

    # Bbus theta - Cg Pg = -Pd
    Aeq = np.zeros((case["nbus"], nx), dtype=float)
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


# ============================================================
# 4. KKT solve utilities
# ============================================================

def split_opf_kkt_delta(delta, nx, m_ineq, n_eq):
    p1 = nx
    p2 = p1 + m_ineq
    p3 = p2 + n_eq
    p4 = p3 + m_ineq

    dx = delta[:p1]
    dz = delta[p1:p2]
    dlambda = delta[p2:p3]
    dmu = delta[p3:p4]

    return dx, dz, dlambda, dmu


def fraction_to_boundary_step(v, dv, tau=0.99):
    negative = dv < 0

    if not np.any(negative):
        return 1.0

    return min(1.0, tau * np.min(-v[negative] / dv[negative]))


def solve_symmetric_preconditioned_kkt(
    K,
    rhs,
    linear_solver,
    reg=OPF_KKT_REG,
    fallback_relres=OPF_QUANTUM_FALLBACK_RELRES,
    max_quantum_dim=OPF_MAX_QUANTUM_KKT_DIM,
):
    """
    Solve preconditioned KKT system robustly.

    Ý tưởng:
    - Scale đối xứng để KKT bớt xấu điều kiện.
    - Nếu dimension quá lớn hoặc quantum solver cho residual xấu,
      tự động fallback về classical np.linalg.solve/lstsq.
    """

    K = np.asarray(K, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if np.linalg.norm(rhs) < 1e-14:
        return np.zeros_like(rhs)

    # Symmetrize + regularize.
    K = 0.5 * (K + K.T)
    K = K + reg * np.eye(K.shape[0])

    # Symmetric diagonal scaling: Kp y = rp, delta = D y.
    row_norm = np.sum(np.abs(K), axis=1)
    row_norm = np.maximum(row_norm, 1e-12)
    D = 1.0 / np.sqrt(row_norm)

    Kp = (D[:, None] * K) * D[None, :]
    rp = D * rhs

    def classical_solve(Kmat, rvec):
        try:
            return np.linalg.solve(Kmat, rvec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Kmat, rvec, rcond=None)[0]

    # Nếu KKT quá lớn, đừng gọi VQLS/HHL/QUBO.
    if Kp.shape[0] > max_quantum_dim:
        y = classical_solve(Kp, rp)
        return clean_real_vector(D * y)

    # Thử solver được truyền vào.
    try:
        y = linear_solver(Kp, rp)
        y = clean_real_vector(y)
    except Exception as exc:
        print(f"[OPF warning] quantum/KKT solver failed: {exc}")
        y = classical_solve(Kp, rp)
        return clean_real_vector(D * y)

    # Kiểm tra chất lượng nghiệm.
    relres = np.linalg.norm(Kp @ y - rp) / max(np.linalg.norm(rp), 1e-14)

    if (not np.all(np.isfinite(y))) or relres > fallback_relres:
        print(
            f"[OPF warning] KKT solver residual too large: "
            f"{relres:.3e}. Fallback to classical solve."
        )
        y = classical_solve(Kp, rp)

    return clean_real_vector(D * y)


# ============================================================
# 5. Primal-dual IPM for DC-OPF
# ============================================================

def dc_opf_primal_dual_ipm(
    case,
    linear_solver,
    name="DC-OPF",
    include_line_limits=False,
    max_iter=OPF_MAX_ITER,
    tol=OPF_TOL,
    sigma=OPF_SIGMA,
    damp_steps=False,
    verbose=True,
):
    """
    Starter primal-dual interior-point method for DC-OPF.

    Quantum part:
        The KKT linear system is solved by the supplied linear_solver.
    """

    data = build_dc_opf_qp(
        case,
        include_line_limits=include_line_limits,
    )

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

        feascond = max(
            np.linalg.norm(rh, ord=np.inf),
            np.linalg.norm(rg, ord=np.inf),
        ) / (
            1.0 + max(np.linalg.norm(beq, ord=np.inf), np.linalg.norm(h, ord=np.inf))
        )

        compcond = gap

        cost = 0.5 * float(x @ H @ x) + float(c @ x) + data["cost_const"]

        history.append({
            "iteration": it,
            "cost": float(cost),
            "gradcond": float(gradcond),
            "feascond": float(feascond),
            "compcond": float(compcond),
            "kkt_dim": int(nx + n_eq),
        })

        if verbose:
            print(
                f"{name:22s} | iter={it:02d} | "
                f"cost={cost:.6f} | "
                f"grad={gradcond:.3e} | "
                f"feas={feascond:.3e} | "
                f"gap={compcond:.3e}"
            )

        if max(gradcond, feascond, compcond) < tol:
            break

                # ------------------------------------------------------------
        # Reduced KKT system
        # ------------------------------------------------------------
        # Full KKT dùng biến [dx, dz, dlambda, dmu].
        # Ta loại dz và dmu để hệ nhỏ hơn:
        #
        # dz  = -rg - G dx
        # dmu = -rz + W (rg + G dx),  W = diag(mu / z)
        #
        # Reduced system:
        # [H + G.T W G   Aeq.T] [dx     ] = [-rx + G.T rz - G.T W rg]
        # [Aeq             0  ] [dlambda]   [-rh]
        #
        # Lợi ích:
        # - full KKT dim = nx + m_ineq + n_eq + m_ineq
        # - reduced dim  = nx + n_eq
        #
        # Với IEEE14 DC-OPF: thường giảm từ khoảng 52 xuống khoảng 32.

        W = mu / z

        K_red_11 = H + G.T @ (W[:, None] * G)
        K_red_12 = Aeq.T
        K_red_21 = Aeq
        K_red_22 = np.zeros((n_eq, n_eq), dtype=float)

        KKT = np.block([
            [K_red_11, K_red_12],
            [K_red_21, K_red_22],
        ])

        rhs_red_x = -rx + G.T @ rz - G.T @ (W * rg)
        rhs_red_h = -rh
        rhs = np.concatenate([rhs_red_x, rhs_red_h])

        delta_red = solve_symmetric_preconditioned_kkt(
            KKT,
            rhs,
            linear_solver=linear_solver,
        )

        dx = delta_red[:nx]
        dlmbda = delta_red[nx:nx + n_eq]

        # Reconstruct eliminated variables.
        dz = -rg - G @ dx
        dmu = -rz + W * (rg + G @ dx)

        alpha_p = fraction_to_boundary_step(
            z,
            dz,
            OPF_FRACTION_TO_BOUNDARY,
        )

        alpha_d = fraction_to_boundary_step(
            mu,
            dmu,
            OPF_FRACTION_TO_BOUNDARY,
        )

        if damp_steps:
            alpha_p = min(alpha_p, 0.5)
            alpha_d = min(alpha_d, 0.5)

        x = x + alpha_p * dx
        z = z + alpha_p * dz
        lmbda = lmbda + alpha_d * dlmbda
        mu = mu + alpha_d * dmu

        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(z))
            and np.all(np.isfinite(lmbda))
            and np.all(np.isfinite(mu))
        ):
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

    balance = (
        data["Bbus"] @ theta
        - Cg @ pg_pu
        + case["bus"][:, PD] / case["base_mva"]
    )

    pmin_mw = case["gen"][online_gens, PMIN]
    pmax_mw = case["gen"][online_gens, PMAX]

    gen_violation = (
        np.maximum(pg_mw - pmax_mw, 0.0)
        +
        np.maximum(pmin_mw - pg_mw, 0.0)
    )

    if data["Bf"].shape[0] > 0:
        finite_rates = np.isfinite(data["rates"])

        if np.any(finite_rates):
            line_violation = np.maximum(
                np.abs(flows_pu[finite_rates]) - data["rates"][finite_rates],
                0.0,
            )
            max_line_violation_mw = float(np.max(line_violation) * case["base_mva"])
        else:
            max_line_violation_mw = 0.0
    else:
        max_line_violation_mw = 0.0

    true_cost = 0.0

    for j, gen_idx in enumerate(online_gens):
        cost_row = case["gencost"][gen_idx]

        true_cost += (
            cost_row[GENCOST_C2] * pg_mw[j] ** 2
            + cost_row[GENCOST_C1] * pg_mw[j]
            + cost_row[GENCOST_C0]
        )

    last = history[-1]

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
        "max_line_violation_mw": float(max_line_violation_mw),
        "history": history,
        "iterations": len(history),
        "converged": max(
            last["gradcond"],
            last["feascond"],
            last["compcond"],
        ) < tol,
        "include_line_limits": include_line_limits,
        "kkt_dim": history[0]["kkt_dim"],
    }


# ============================================================
# 6. Reporting and plotting
# ============================================================

def print_dc_opf_result(result):
    print("\n" + "=" * 80)
    print(result["name"])
    print("=" * 80)

    print(f"converged             = {result['converged']}")
    print(f"iterations            = {result['iterations']}")
    print(f"KKT dimension         = {result['kkt_dim']} -> padded to {next_power_of_two(result['kkt_dim'])}")
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

        plt.plot(
            grad,
            marker="o",
            linewidth=2,
            label=result["name"],
        )

    plt.yscale("log")
    plt.xlabel("OPF IPM iteration")
    plt.ylabel("gradcond")
    plt.title("DC-OPF KKT solver convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "dc_opf_gradcond.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_opf_dispatch(opf_results):
    plt.figure(figsize=(9, 5))

    ref = opf_results[0]
    labels = [f"gen {idx}" for idx in ref["online_gens"]]

    x = np.arange(len(labels))
    n_methods = len(opf_results)
    width = min(0.8 / n_methods, 0.25)

    for k, result in enumerate(opf_results):
        offset = (k - (n_methods - 1) / 2) * width

        plt.bar(
            x + offset,
            result["pg_mw"],
            width,
            label=result["name"],
        )

    plt.xticks(x, labels)
    plt.xlabel("Generator")
    plt.ylabel("Pg (MW)")
    plt.title("DC-OPF generator dispatch comparison")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "dc_opf_generator_dispatch.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def run_dc_opf_comparison(
    case,
    classical_solver,
    vqls_solver=None,
    hhl_solver=None,
    qubo_solver=None,
):
    print("\n" + "=" * 80)
    print("QUANTUM OPTIMIZATION POWER FLOW: STARTER DC-OPF")
    print("=" * 80)

    print(f"OPF_INCLUDE_LINE_LIMITS = {OPF_INCLUDE_LINE_LIMITS}")
    print(f"RUN_QUANTUM_OPF         = {RUN_QUANTUM_OPF}")

    result_classical = dc_opf_primal_dual_ipm(
        case,
        linear_solver=classical_solver,
        name="Classical DC-OPF",
        include_line_limits=OPF_INCLUDE_LINE_LIMITS,
        max_iter=OPF_MAX_ITER,
        damp_steps=False,
        verbose=True,
    )

    results = [result_classical]

    if RUN_QUANTUM_OPF:
        if vqls_solver is not None:
            result_vqls = dc_opf_primal_dual_ipm(
                case,
                linear_solver=vqls_solver,
                name="VQLS DC-OPF",
                include_line_limits=OPF_INCLUDE_LINE_LIMITS,
                max_iter=OPF_QUANTUM_MAX_ITER,
                damp_steps=True,
                verbose=True,
            )
            results.append(result_vqls)

        if hhl_solver is not None:
            result_hhl = dc_opf_primal_dual_ipm(
                case,
                linear_solver=hhl_solver,
                name="HHL DC-OPF",
                include_line_limits=OPF_INCLUDE_LINE_LIMITS,
                max_iter=OPF_QUANTUM_MAX_ITER,
                damp_steps=True,
                verbose=True,
            )
            results.append(result_hhl)

        # QUBO for KKT can be very expensive. Keep optional.
        if qubo_solver is not None:
            result_qubo = dc_opf_primal_dual_ipm(
                case,
                linear_solver=qubo_solver,
                name="QUBO DC-OPF",
                include_line_limits=OPF_INCLUDE_LINE_LIMITS,
                max_iter=OPF_QUANTUM_MAX_ITER,
                damp_steps=True,
                verbose=True,
            )
            results.append(result_qubo)

    for result in results:
        print_dc_opf_result(result)

    plot_opf_gradcond(results)
    plot_opf_dispatch(results)

    return results