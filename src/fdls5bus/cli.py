"""Command-line interface for fdls5bus."""

from __future__ import annotations

import argparse

from .data import case5
from .solver import solve_fdls


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FDLS on the MATPOWER/PJM 5-bus case.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance in per unit.")
    parser.add_argument("--max-it", type=int, default=50, help="Maximum number of iterations.")
    args = parser.parse_args()

    result = solve_fdls(case5(), tolerance=args.tol, max_iterations=args.max_it)

    print(f"Converged: {result.converged} in {result.iterations} iterations")
    print("\nBus results")
    print("bus | Vm(pu)    | Va(deg)    | Pcalc(pu)   | Qcalc(pu)")
    print("----+-----------+------------+-------------+------------")
    for i, (vm, va, p, q) in enumerate(
        zip(result.vm, result.va_deg, result.p_calc, result.q_calc), start=1
    ):
        print(f"{i:>3d} | {vm:>9.6f} | {va:>10.6f} | {p:>11.6f} | {q:>10.6f}")

    print("\nMismatch history")
    print("it | max|dP|     | max|dQ|")
    print("---+-------------+------------")
    for row in result.history:
        print(f"{row['iteration']:>2d} | {row['max_dP']:>11.3e} | {row['max_dQ']:>10.3e}")


if __name__ == "__main__":
    main()
