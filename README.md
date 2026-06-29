# IEEE 14-Bus FDLS with Classical, VQLS no-CZ, and HHL Solvers

This project implements the Fast Decoupled Load Flow Solver (FDLS) on the IEEE 14-bus power system and compares three linear solvers for the (P-\theta) step:

1. Classical solver using `numpy.linalg.solve`
2. VQLS no-CZ solver using Qiskit
3. HHL solver using Qiskit

The project exports matrices, runs FDLS, compares voltage solutions, and saves plots and text results.

## Project Structure

```text
src/
├── main.py
├── config.py
├── logger.py
├── ieee14_case.py
├── exporter.py
├── math_utils.py
├── classical_solver.py
├── vqls_solver.py
├── hhl_solver.py
├── fdls.py
├── core.py
└── plots.py
```

## Problem

FDLS solves the decoupled power-flow equations:

```math
B'\Delta\theta = \frac{\Delta P}{|V|}
```

```math
B''\Delta |V| = \frac{\Delta Q}{|V|}
```

For the IEEE 14-bus system:

* (Ybus): `14 × 14`
* (B'): `13 × 13`
* (B''): `9 × 9`
* Quantum padded (B'): `16 × 16`
* Number of system qubits: `4`

## Installation

Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

```text
numpy
scipy
matplotlib
qiskit
qiskit-aer
pylatexenc
pypower
```

## Run

From the `src` folder:

```bat
cd src
python main.py
```

## Outputs

The program saves results in:

```text
src/outputs/
```

Text output:

```text
outputs/run_results.txt
```

Exported matrices:

```text
outputs/matrices/Ybus_real.csv
outputs/matrices/Ybus_imag.csv
outputs/matrices/Bprime.csv
outputs/matrices/Bdouble_prime.csv
outputs/matrices/rhs_p_initial.csv
outputs/matrices/matrix_info.txt
```

Plots:

```text
outputs/one_step_delta_theta_solution.png
outputs/ieee14_fdls_loss_comparison.png
outputs/ieee14_final_angle_solution.png
outputs/ieee14_final_voltage_solution.png
outputs/vqls_no_cz_cost.png
```

## Notes

The quantum solvers are applied to the (P-\theta) linear system. The (Q-V) step is solved classically.

This project is intended for research and educational comparison. It does not claim quantum advantage because the current implementation uses statevector simulation and a small benchmark system.
