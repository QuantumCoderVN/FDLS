import os
import sys
import warnings
import numpy as np

import config as cfg

from dc_opf import run_dc_opf_comparison
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
# Safe config getter
# ============================================================

def get_cfg(name, default):
    """
    Read config variable safely.

    Nếu config.py chưa có biến đó thì dùng default,
    tránh lỗi ImportError / AttributeError.
    """
    return getattr(cfg, name, default)


# ============================================================
# Small helper
# ============================================================

def next_power_of_two(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ============================================================
# Solver wrappers
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
    """
    RAW QUBO solver wrapper.

    Dùng cho thí nghiệm không fallback / không fail-safe.
    Không trả zero khi residual lớn.
    Không clamp norm nghiệm.
    Mục tiêu: xem QUBO tự nó kéo OPF đi đúng hay làm OPF nổ.
    """

    x, info = solve_qubo_annealing(A, b, verbose=False)

    rel = info.get("relative_residual", np.nan)
    num_bin = info.get("num_binary_variables", None)

    print(
        f"[RAW QUBO] rel_residual={rel:.3e}, "
        f"num_binary_variables={num_bin}"
    )

    return x


# ============================================================
# Print helpers
# ============================================================

def print_metrics(name, metrics):
    print(name + ":")
    print(f"  residual          = {metrics['residual']:.6e}")
    print(f"  relative residual = {metrics['rel_residual']:.6e}")
    print(f"  relative error    = {metrics['rel_error']:.6e}")
    print(f"  fidelity          = {metrics['fidelity']:.8f}")


def print_case_info(case, Ybus, Bprime, Bdouble, rhs_p0):
    case_name = case.get("case_name", get_cfg("POWER_CASE", "unknown"))

    print("\nCase information")
    print("-" * 80)
    print(f"case         = {case_name}")
    print(f"baseMVA      = {case['base_mva']}")
    print(f"nbus         = {case['nbus']}")

    slack_idx = case["slack"]
    slack_bus_number = int(case["bus"][slack_idx, 0])

    print(f"slack index  = {slack_idx}  bus number {slack_bus_number}")
    print(f"PV indices   = {case['pv']}")
    print(f"PQ indices   = {case['pq']}")
    print(f"non-slack    = {case['non_slack']}")

    print("\nMatrix sizes")
    print("-" * 80)
    print(f"Ybus shape        = {Ybus.shape}")
    print(f"Bprime shape      = {Bprime.shape}")
    print(f"Bdouble shape     = {Bdouble.shape}")
    print(f"Initial rhs shape = {rhs_p0.shape}")

    pad_n = next_power_of_two(Bprime.shape[0])
    print(
        f"VQLS/HHL pad Bprime from "
        f"{Bprime.shape[0]}x{Bprime.shape[1]} to {pad_n}x{pad_n}."
    )

    bits_per_var = get_cfg("QUBO_BITS_PER_VAR", None)
    if bits_per_var is not None:
        print(
            f"QUBO would encode {Bprime.shape[0]} continuous variables "
            f"into about {Bprime.shape[0] * bits_per_var} binary variables."
        )


# ============================================================
# One-step benchmark
# ============================================================

def run_one_step_benchmark(case, Bprime, rhs_p0):
    print("\n" + "=" * 80)
    print("ONE-STEP P-THETA MATRIX SOLVE")
    print("=" * 80)

    print("\nBprime matrix:")
    print(Bprime)

    print("\nInitial rhs_p = ΔP/|V|:")
    print(rhs_p0)

    x_classical = solve_classical(Bprime, rhs_p0)

    print("\nClassical Δθ:")
    print(x_classical)

    solutions = {
        "Classical": x_classical,
    }

    # VQLS one-step
    if get_cfg("RUN_ONE_STEP_VQLS", True):
        x_vqls, vqls_info = solve_vqls_no_cz(Bprime, rhs_p0, verbose=True)
        solutions["VQLS no-CZ"] = x_vqls

        print("\nVQLS no-CZ Δθ:")
        print(x_vqls)

        vqls_metrics = vector_metrics(Bprime, rhs_p0, x_vqls, x_classical)

        print("\nVQLS metrics vs classical")
        print("-" * 80)
        print_metrics("VQLS no-CZ", vqls_metrics)

    # HHL one-step
    if get_cfg("RUN_ONE_STEP_HHL", False):
        x_hhl, hhl_info = solve_hhl(Bprime, rhs_p0, verbose=True)
        solutions["HHL"] = x_hhl

        print("\nHHL Δθ:")
        print(x_hhl)

        hhl_metrics = vector_metrics(Bprime, rhs_p0, x_hhl, x_classical)

        print("\nHHL metrics vs classical")
        print("-" * 80)
        print_metrics("HHL", hhl_metrics)
        print(f"  success prob = {hhl_info.get('success_probability', np.nan):.6e}")

    # QUBO one-step
    if get_cfg("RUN_ONE_STEP_QUBO", False):
        x_qubo, qubo_info = solve_qubo_annealing(Bprime, rhs_p0, verbose=True)
        solutions["QUBO annealing"] = x_qubo

        print("\nQUBO annealing Δθ:")
        print(x_qubo)

        qubo_metrics = vector_metrics(Bprime, rhs_p0, x_qubo, x_classical)

        print("\nQUBO metrics vs classical")
        print("-" * 80)
        print_metrics("QUBO annealing", qubo_metrics)
        print(f"  QUBO binary variables = {qubo_info.get('num_binary_variables')}")
        print(f"  QUBO bits per var     = {qubo_info.get('bits_per_var')}")

        if "step" in qubo_info:
            print(f"  QUBO step             = {qubo_info.get('step'):.6e}")

        if "Q_scaled" in qubo_info:
            os.makedirs(cfg.MATRIX_DIR, exist_ok=True)
            np.savetxt(
                os.path.join(cfg.MATRIX_DIR, "QUBO_Q_initial_scaled.csv"),
                qubo_info["Q_scaled"],
                delimiter=",",
                fmt="%.12e",
            )

    # Plot only available solutions
    try:
        plot_one_step_solution(case["non_slack"], solutions)
    except Exception as exc:
        print(f"[warning] Could not create one-step plot: {exc}")

    return solutions


# ============================================================
# FDLS benchmark
# ============================================================

def run_fdls_benchmark(case, Ybus, Bprime, Bdouble):
    print("\n" + "=" * 80)
    print("RUNNING TRUE FDLS")
    print("=" * 80)

    results = []

    result_classical = fdls_power_flow(
        case,
        Ybus,
        Bprime,
        Bdouble,
        classical_p_solver,
        "Classical FDLS",
    )
    results.append(result_classical)

    if get_cfg("RUN_FDLS_VQLS", True):
        result_vqls = fdls_power_flow(
            case,
            Ybus,
            Bprime,
            Bdouble,
            vqls_p_solver,
            "VQLS no-CZ FDLS",
        )
        results.append(result_vqls)

    if get_cfg("RUN_FDLS_HHL", False):
        result_hhl = fdls_power_flow(
            case,
            Ybus,
            Bprime,
            Bdouble,
            hhl_p_solver,
            "HHL FDLS",
        )
        results.append(result_hhl)

    if get_cfg("RUN_FDLS_QUBO", False):
        result_qubo = fdls_power_flow(
            case,
            Ybus,
            Bprime,
            Bdouble,
            qubo_p_solver,
            "QUBO annealing FDLS",
        )
        results.append(result_qubo)

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

    # Error vs classical
    print("\n" + "=" * 80)
    print("FDLS FINAL SOLUTION ERROR VS CLASSICAL FDLS")
    print("=" * 80)

    ref_va = result_classical["va"]
    ref_vm = result_classical["vm"]

    for result in results[1:]:
        angle_error = np.linalg.norm(result["va"] - ref_va)
        voltage_error = np.linalg.norm(result["vm"] - ref_vm)

        print("\n" + result["name"])
        print("-" * 80)
        print(f"angle error   = {angle_error:.6e}")
        print(f"voltage error = {voltage_error:.6e}")

    try:
        plot_fdls_loss(results)
        plot_final_angle_solution(results, case["nbus"])
        plot_final_voltage_solution(results, case["nbus"])
    except Exception as exc:
        print(f"[warning] Could not create FDLS plots: {exc}")

    return results


# ============================================================
# DC-OPF benchmark
# ============================================================

def run_opf_benchmark(case):
    print("\n" + "=" * 80)
    print("RUNNING DC-OPF / QUANTUM OPTIMIZATION POWER FLOW")
    print("=" * 80)

    run_vqls = get_cfg("RUN_OPF_VQLS", True)
    run_hhl = get_cfg("RUN_OPF_HHL", False)
    run_qubo = get_cfg("RUN_OPF_QUBO", False)

    print("\nOPF solvers enabled")
    print("-" * 80)
    print(f"Classical = True")
    print(f"VQLS      = {run_vqls}")
    print(f"HHL       = {run_hhl}")
    print(f"QUBO      = {run_qubo}")

    opf_results = run_dc_opf_comparison(
        case,
        classical_solver=classical_p_solver,
        vqls_solver=vqls_p_solver if run_vqls else None,
        hhl_solver=hhl_p_solver if run_hhl else None,
        qubo_solver=qubo_p_solver if run_qubo else None,
    )

    return opf_results


# ============================================================
# Main program
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("POWER FLOW / FDLS / QUANTUM OPF EXPERIMENT")
    print("=" * 80)

    print("\nRun configuration")
    print("-" * 80)
    print(f"POWER_CASE             = {get_cfg('POWER_CASE', 'unknown')}")
    print(f"RUN_ONE_STEP_BENCHMARK = {get_cfg('RUN_ONE_STEP_BENCHMARK', False)}")
    print(f"RUN_FDLS_BENCHMARK     = {get_cfg('RUN_FDLS_BENCHMARK', False)}")
    print(f"RUN_DC_OPF             = {get_cfg('RUN_DC_OPF', True)}")

    # ------------------------------------------------------------
    # Load selected case and build matrices
    # ------------------------------------------------------------

    case = load_ieee14_case()

    Ybus = make_ybus(case)
    Bprime, Bdouble = make_fdls_matrices(Ybus, case)

    P_spec, Q_spec = make_specified_power(case)
    vm0, va0 = initial_voltage(case)

    P_calc0, Q_calc0 = calculated_power(Ybus, vm0, va0)

    non_slack = case["non_slack"]
    rhs_p0 = (P_spec - P_calc0)[non_slack] / vm0[non_slack]

    os.makedirs(cfg.MATRIX_DIR, exist_ok=True)
    export_matrices(Ybus, Bprime, Bdouble, rhs_p0, case)

    print_case_info(case, Ybus, Bprime, Bdouble, rhs_p0)

    # ------------------------------------------------------------
    # Optional: one-step matrix solve
    # ------------------------------------------------------------

    if get_cfg("RUN_ONE_STEP_BENCHMARK", False):
        run_one_step_benchmark(case, Bprime, rhs_p0)
    else:
        print("\n[skip] ONE-STEP benchmark is disabled.")

    # ------------------------------------------------------------
    # Optional: full FDLS
    # ------------------------------------------------------------

    if get_cfg("RUN_FDLS_BENCHMARK", False):
        run_fdls_benchmark(case, Ybus, Bprime, Bdouble)
    else:
        print("\n[skip] FDLS benchmark is disabled.")

    # ------------------------------------------------------------
    # Optional: DC-OPF
    # ------------------------------------------------------------

    if get_cfg("RUN_DC_OPF", True):
        run_opf_benchmark(case)
    else:
        print("\n[skip] DC-OPF benchmark is disabled.")

    # ------------------------------------------------------------
    # Output summary
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(f"Text result file: {cfg.RESULT_TEXT_FILE}")

    print("\nMatrix files:")
    print(f"  {cfg.MATRIX_DIR}/Ybus_real.csv")
    print(f"  {cfg.MATRIX_DIR}/Ybus_imag.csv")
    print(f"  {cfg.MATRIX_DIR}/Bprime.csv")
    print(f"  {cfg.MATRIX_DIR}/Bdouble_prime.csv")
    print(f"  {cfg.MATRIX_DIR}/rhs_p_initial.csv")

    print("\nPossible plots:")
    print("  outputs/one_step_delta_theta_solution.png")
    print("  outputs/ieee14_fdls_loss_comparison.png")
    print("  outputs/ieee14_final_angle_solution.png")
    print("  outputs/ieee14_final_voltage_solution.png")
    print("  outputs/vqls_no_cz_cost.png")
    print("  outputs/qubo_annealing_energy.png")
    print("  outputs/dc_opf_gradcond.png")
    print("  outputs/dc_opf_generator_dispatch.png")

    print("\nImportant note:")
    print("QUBO annealing here means QUBO formulation + classical simulated annealing.")
    print("It is not a real D-Wave quantum annealer run yet.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    os.makedirs(os.path.dirname(cfg.RESULT_TEXT_FILE), exist_ok=True)

    with open(cfg.RESULT_TEXT_FILE, "w", encoding="utf-8") as result_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, result_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout