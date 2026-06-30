import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def plot_one_step_solution(non_slack_buses, solutions):
    """
    solutions is a dict:

        {
            "Classical": x_classical,
            "VQLS no-CZ": x_vqls,
            "HHL": x_hhl,
            "QUBO annealing": x_qubo,
        }
    """

    labels = [str(i + 1) for i in non_slack_buses]
    x = np.arange(len(labels))

    names = list(solutions.keys())
    n_methods = len(names)
    width = min(0.8 / n_methods, 0.22)

    plt.figure(figsize=(14, 5))

    for k, name in enumerate(names):
        offset = (k - (n_methods - 1) / 2) * width
        values = np.real(np.asarray(solutions[name], dtype=float))
        plt.bar(x + offset, values, width, label=name)

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Initial Δθ solution")
    plt.title("One-step P-theta solution from IEEE 14-bus matrix")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "one_step_delta_theta_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_fdls_loss(results):
    plt.figure(figsize=(8, 5))

    for result in results:
        plt.plot(
            result["loss_history"],
            marker="o",
            linewidth=2,
            label=result["name"],
        )

    plt.yscale("log")
    plt.xlabel("FDLS iteration")
    plt.ylabel("Loss = sqrt(||ΔP||² + ||ΔQ||²)")
    plt.title("IEEE 14-bus FDLS convergence")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_fdls_loss_comparison.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_final_angle_solution(results, nbus):
    labels = [str(i + 1) for i in range(nbus)]
    x = np.arange(nbus)

    n_methods = len(results)
    width = min(0.8 / n_methods, 0.22)

    plt.figure(figsize=(14, 5))

    for k, result in enumerate(results):
        offset = (k - (n_methods - 1) / 2) * width

        plt.bar(
            x + offset,
            result["va_deg"],
            width,
            label=result["name"],
        )

    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Voltage angle θ (degrees)")
    plt.title("IEEE 14-bus final voltage angle solution")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_final_angle_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def plot_final_voltage_solution(results, nbus):
    labels = [str(i + 1) for i in range(nbus)]
    x = np.arange(nbus)

    n_methods = len(results)
    width = min(0.8 / n_methods, 0.22)

    plt.figure(figsize=(14, 5))

    for k, result in enumerate(results):
        offset = (k - (n_methods - 1) / 2) * width

        plt.bar(
            x + offset,
            result["vm"],
            width,
            label=result["name"],
        )

    plt.xticks(x, labels)
    plt.xlabel("Bus number")
    plt.ylabel("Voltage magnitude |V|")
    plt.title("IEEE 14-bus final voltage magnitude solution")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(OUTPUT_DIR, "ieee14_final_voltage_solution.png")
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")