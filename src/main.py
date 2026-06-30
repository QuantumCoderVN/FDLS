import os
import sys
import warnings
import numpy as np
from config import RUN_DC_OPF
from dc_opf import run_dc_opf_comparison
from config import RESULT_TEXT_FILE, MATRIX_DIR
from config import QUBO_MAX_STEP_NORM_FACTOR
from logger import Tee

from ieee14_case import (
    load_ieee14_case,
    make_ybus,
    make_fdls_matrices,
    make_specified_power,
    initial_voltage,
    calculated_power,
)

from exporter import export_matrices

from math_utils import vector_metrics

from classical_solver import solve_classical
from vqls_solver import solve_vqls_no_cz
from hhl_solver import solve_hhl
from qubo_solver import solve_qubo_annealing

from fdls import fdls_power_flow

from plots import (
    plot_one_step_solution,
    plot_fdls_loss,
    plot_final_angle_solution,
    plot_final_voltage_solution,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# Solver wrappers for FDLS P-theta step
# ============================================================

def classical_p_solver(A, b):
    return solve_classical(A, b)


def vqls_p_solver(A, b):
    x, _ = solve_vqls_no_cz(A, b, verbose=False)
    return x


def hhl_p_solver(A, b):
    x, _ = solve_hhl(A, b, verbose=False)
    return x


def qubo_p_solver(A, b):
    x, info = solve_qubo_annealing(A, b, verbose=False)

    # Nếu QUBO vẫn quá tệ thì trả bước nhỏ để không phá FDLS.
    rel = info.get("relative_residual", 999.0)

    if rel > 1.0:
        # Fail-safe: không cho QUBO update phá hệ.
        return np.zeros_like(b)

    # Chặn norm bước update.
    classical_scale = np.linalg.norm(b) / max(np.linalg.norm(A, ord=2), 1e-14)
    max_norm = QUBO_MAX_STEP_NORM_FACTOR * max(classical_scale, 1e-12)

    norm_x = np.linalg.norm(x)

    if norm_x > max_norm:
        x = x * (max_norm / norm_x)

    return x


# ============================================================
# Print metric helper
# ============================================================

def print_metrics(name, metrics):
    print(name + ":")
    print(f"  residual          = {metrics['residual']:.6e}")
    print(f"  relative residual = {metrics['rel_residual']:.6e}")
    print(f"  relative error    = {metrics['rel_error']:.6e}")
    print(f"  fidelity          = {metrics['fidelity']:.8f}")


# ============================================================
# Main program
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("IEEE 14-BUS DATA + FDLS + CLASSICAL/VQLS/HHL/QUBO")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load IEEE 14 data and build matrices
    # ------------------------------------------------------------

    case = load_ieee14_case()
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
    print(f"baseMVA      = {case['base_mva']}")
    print(f"nbus         = {case['nbus']}")
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
    print("VQLS/HHL pad Bprime from 13x13 to 16x16.")
    print("QUBO encodes 13 continuous variables into binary variables.")

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
    x_qubo, qubo_info = solve_qubo_annealing(Bprime, rhs_p0, verbose=True)

    # Save initial QUBO matrix
    if "Q_scaled" in qubo_info:
        np.savetxt(
            os.path.join(MATRIX_DIR, "QUBO_Q_initial_scaled.csv"),
            qubo_info["Q_scaled"],
            delimiter=",",
            fmt="%.12e",
        )

    print("\nClassical Δθ:")
    print(x_classical)

    print("\nVQLS no-CZ Δθ:")
    print(x_vqls)

    print("\nHHL Δθ:")
    print(x_hhl)

    print("\nQUBO annealing Δθ:")
    print(x_qubo)

    print("\nSign comparison")
    print("Classical:", np.sign(x_classical))
    print("VQLS:     ", np.sign(x_vqls))
    print("HHL:      ", np.sign(x_hhl))
    print("QUBO:     ", np.sign(x_qubo))

    vqls_metrics = vector_metrics(Bprime, rhs_p0, x_vqls, x_classical)
    hhl_metrics = vector_metrics(Bprime, rhs_p0, x_hhl, x_classical)
    qubo_metrics = vector_metrics(Bprime, rhs_p0, x_qubo, x_classical)

    print("\nOne-step metrics vs classical")
    print("-" * 80)
    print_metrics("VQLS no-CZ", vqls_metrics)
    print_metrics("HHL", hhl_metrics)
    print_metrics("QUBO annealing", qubo_metrics)
    print(f"  QUBO binary variables = {qubo_info.get('num_binary_variables')}")
    print(f"  QUBO bits per var     = {qubo_info.get('bits_per_var')}")
    print(f"  QUBO step             = {qubo_info.get('step'):.6e}")

    print("\nHHL:")
    print(f"  success prob          = {hhl_info['success_probability']:.6e}")

    # ------------------------------------------------------------
    # True FDLS with four P-theta solvers
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

    result_qubo = fdls_power_flow(
        case,
        Ybus,
        Bprime,
        Bdouble,
        qubo_p_solver,
        "QUBO annealing FDLS",
    )

    results = [
        result_classical,
        result_vqls,
        result_hhl,
        result_qubo,
    ]

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

    # ------------------------------------------------------------
    # Final error vs classical FDLS
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FDLS FINAL SOLUTION ERROR VS CLASSICAL FDLS")
    print("=" * 80)

    ref_va = result_classical["va"]
    ref_vm = result_classical["vm"]

    for result in [result_vqls, result_hhl, result_qubo]:
        angle_error = np.linalg.norm(result["va"] - ref_va)
        voltage_error = np.linalg.norm(result["vm"] - ref_vm)

        print("\n" + result["name"])
        print("-" * 80)
        print(f"angle error   = {angle_error:.6e}")
        print(f"voltage error = {voltage_error:.6e}")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_one_step_solution(
        non_slack,
        {
            "Classical": x_classical,
            "VQLS no-CZ": x_vqls,
            "HHL": x_hhl,
            "QUBO annealing": x_qubo,
        },
    )

    plot_fdls_loss(results)
    plot_final_angle_solution(results, case["nbus"])
    plot_final_voltage_solution(results, case["nbus"])
    # ------------------------------------------------------------
    # Quantum Optimization Power Flow / DC-OPF
    # ------------------------------------------------------------

    if RUN_DC_OPF:
        opf_results = run_dc_opf_comparison(
            case,
            classical_solver=classical_p_solver,
            vqls_solver=vqls_p_solver,
            hhl_solver=hhl_p_solver,
            qubo_solver=qubo_p_solver,
        )
    # ------------------------------------------------------------
    # Output summary
    # ------------------------------------------------------------

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
    print(f"  {MATRIX_DIR}/QUBO_Q_initial_scaled.csv")

    print("Plots:")
    print("  outputs/one_step_delta_theta_solution.png")
    print("  outputs/ieee14_fdls_loss_comparison.png")
    print("  outputs/ieee14_final_angle_solution.png")
    print("  outputs/ieee14_final_voltage_solution.png")
    print("  outputs/vqls_no_cz_cost.png")
    print("  outputs/qubo_annealing_energy.png")

    print("\nImportant note:")
    print("QUBO annealing here means QUBO formulation + classical simulated annealing.")
    print("It is not a real D-Wave quantum annealer run yet.")

    print("  outputs/dc_opf_gradcond.png")
    print("  outputs/dc_opf_generator_dispatch.png")
if __name__ == "__main__":
    with open(RESULT_TEXT_FILE, "w", encoding="utf-8") as result_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, result_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout