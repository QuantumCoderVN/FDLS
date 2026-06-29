import numpy as np

from config import FDLS_MAX_ITER, FDLS_TOL

from ieee14_case import (
    VA,
    VM,
    make_specified_power,
    initial_voltage,
    calculated_power,
)

from math_utils import clean_real_vector


def fdls_power_flow(case, Ybus, Bprime, Bdouble, p_solver, name):
    """
    True FDLS loop:

        P-theta step:
            B' Δθ = ΔP / |V|

        Q-V step:
            B'' ΔV = ΔQ / |V|

    P-theta uses:
        Classical / VQLS / HHL

    Q-V is solved classically.
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
            np.sum(mismatch_p ** 2)
            +
            np.sum(mismatch_q ** 2)
        )

        loss_history.append(float(loss))

        if loss < FDLS_TOL:
            break

        # P-theta step
        rhs_p = mismatch_p / vm[non_slack]
        dtheta = p_solver(Bprime, rhs_p)
        dtheta = clean_real_vector(dtheta)

        va[non_slack] += dtheta

        # Q-V step
        rhs_q = mismatch_q / vm[pq]
        dV = np.linalg.solve(Bdouble, rhs_q)

        vm[pq] += dV

        # Keep slack fixed
        va[slack] = np.deg2rad(case["bus"][slack, VA])
        vm[slack] = case["bus"][slack, VM]

        # Keep PV voltage magnitudes fixed
        for idx in case["pv"]:
            vm[idx] = case["bus"][idx, VM]

    P_final, Q_final = calculated_power(Ybus, vm, va)

    final_dP = P_spec - P_final
    final_dQ = Q_spec - Q_final

    final_loss = np.sqrt(
        np.sum(final_dP[non_slack] ** 2)
        +
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