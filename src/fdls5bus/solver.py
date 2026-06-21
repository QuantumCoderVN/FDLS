"""Fast Decoupled Load-Flow Solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import PQ, PV, SLACK, PowerSystemCase
from .ybus import make_ybus


@dataclass(frozen=True)
class FDLSResult:
    """Result returned by the FDLS solver."""

    converged: bool
    iterations: int
    vm: np.ndarray
    va_rad: np.ndarray
    p_calc: np.ndarray
    q_calc: np.ndarray
    p_spec: np.ndarray
    q_spec: np.ndarray
    p_mismatch: np.ndarray
    q_mismatch: np.ndarray
    history: list[dict[str, float]]
    b_prime: np.ndarray
    b_double_prime: np.ndarray

    @property
    def va_deg(self) -> np.ndarray:
        """Bus voltage angles in degrees."""
        return np.rad2deg(self.va_rad)


def specified_power_injections(case: PowerSystemCase) -> tuple[np.ndarray, np.ndarray]:
    """Return specified net P and Q injections in per unit.

    Injection convention: generation minus load.
    """

    nb = case.bus.shape[0]
    pd = case.bus[:, 2] / case.base_mva
    qd = case.bus[:, 3] / case.base_mva
    pg = np.zeros(nb)
    qg = np.zeros(nb)

    for row in case.gen:
        status = int(row[7])
        if status == 0:
            continue
        bus_index = int(row[0]) - 1
        pg[bus_index] += row[1] / case.base_mva
        qg[bus_index] += row[2] / case.base_mva

    return pg - pd, qg - qd


def bus_sets(case: PowerSystemCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of slack, PV, PQ, and non-slack bus indices."""

    types = case.bus[:, 1].astype(int)
    slack = np.where(types == SLACK)[0]
    pv = np.where(types == PV)[0]
    pq = np.where(types == PQ)[0]
    if slack.size != 1:
        raise ValueError("This solver expects exactly one slack bus.")
    non_slack = np.array([i for i in range(case.bus.shape[0]) if i not in slack], dtype=int)
    return slack, pv, pq, non_slack


def initial_voltage(case: PowerSystemCase) -> tuple[np.ndarray, np.ndarray]:
    """Return initial Vm and Va from case.bus and generator voltage set-points."""

    vm = case.bus[:, 7].astype(float).copy()
    va_rad = np.deg2rad(case.bus[:, 8].astype(float).copy())

    # Respect generator voltage set-points at PV and slack buses.
    types = case.bus[:, 1].astype(int)
    for row in case.gen:
        if int(row[7]) == 0:
            continue
        idx = int(row[0]) - 1
        if types[idx] in (PV, SLACK):
            vm[idx] = row[5]

    return vm, va_rad


def calculated_power(ybus: np.ndarray, vm: np.ndarray, va_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate bus complex-power injection S = V * conj(Ybus V)."""

    voltage = vm * np.exp(1j * va_rad)
    s_calc = voltage * np.conj(ybus @ voltage)
    return s_calc.real, s_calc.imag


def make_b_matrices(case: PowerSystemCase) -> tuple[np.ndarray, np.ndarray]:
    """Construct the constant FDLS matrices B' and B''.

    B' is built from a lossless series-reactance network and excludes the slack
    bus. B'' is the negative imaginary part of the full Ybus restricted to PQ
    buses. This is the common XB fast-decoupled variant.
    """

    _, _, pq, non_slack = bus_sets(case)

    ybus_bp = make_ybus(
        case,
        series_x_only=True,
        include_line_charging=False,
        include_bus_shunts=False,
    )
    ybus_bpp = make_ybus(
        case,
        series_x_only=False,
        include_line_charging=True,
        include_bus_shunts=True,
    )

    b_prime = -ybus_bp.imag[np.ix_(non_slack, non_slack)]
    b_double_prime = -ybus_bpp.imag[np.ix_(pq, pq)]

    return b_prime, b_double_prime


def solve_fdls(
    case: PowerSystemCase,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> FDLSResult:
    """Solve an AC load-flow problem by the Fast Decoupled Load-flow method.

    The solver does not enforce generator reactive-power limits. If a PV bus
    violates Q limits, production-grade tools usually switch that bus to PQ and
    solve again; this pedagogical implementation keeps the core FDLS clear.
    """

    ybus = make_ybus(case)
    slack, _, pq, non_slack = bus_sets(case)
    p_spec, q_spec = specified_power_injections(case)
    vm, va_rad = initial_voltage(case)
    b_prime, b_double_prime = make_b_matrices(case)

    history: list[dict[str, float]] = []
    converged = False

    for iteration in range(max_iterations + 1):
        p_calc, q_calc = calculated_power(ybus, vm, va_rad)
        d_p = p_spec - p_calc
        d_q = q_spec - q_calc

        max_p = float(np.max(np.abs(d_p[non_slack])))
        max_q = float(np.max(np.abs(d_q[pq]))) if pq.size else 0.0
        history.append({"iteration": iteration, "max_dP": max_p, "max_dQ": max_q})

        if max(max_p, max_q) < tolerance:
            converged = True
            break

        # P-theta step: solve B' * Delta theta = Delta P / |V|.
        rhs_p = d_p[non_slack] / vm[non_slack]
        d_theta = np.linalg.solve(b_prime, rhs_p)
        va_rad[non_slack] += d_theta

        # Q-V step: recalculate Q after the angle update, then solve
        # B'' * Delta |V| = Delta Q / |V| for PQ buses only.
        if pq.size:
            _, q_calc = calculated_power(ybus, vm, va_rad)
            d_q = q_spec - q_calc
            rhs_q = d_q[pq] / vm[pq]
            d_vm = np.linalg.solve(b_double_prime, rhs_q)
            vm[pq] += d_vm

    p_calc, q_calc = calculated_power(ybus, vm, va_rad)
    p_mismatch = p_spec - p_calc
    q_mismatch = q_spec - q_calc

    return FDLSResult(
        converged=converged,
        iterations=len(history) - 1,
        vm=vm,
        va_rad=va_rad,
        p_calc=p_calc,
        q_calc=q_calc,
        p_spec=p_spec,
        q_spec=q_spec,
        p_mismatch=p_mismatch,
        q_mismatch=q_mismatch,
        history=history,
        b_prime=b_prime,
        b_double_prime=b_double_prime,
    )
