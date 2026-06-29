import os
import numpy as np

from config import MATRIX_DIR


def save_complex_matrix_csv(path_real, path_imag, matrix):
    np.savetxt(path_real, np.real(matrix), delimiter=",", fmt="%.12e")
    np.savetxt(path_imag, np.imag(matrix), delimiter=",", fmt="%.12e")


def export_matrices(Ybus, Bprime, Bdouble, rhs_p0, case):
    save_complex_matrix_csv(
        os.path.join(MATRIX_DIR, "Ybus_real.csv"),
        os.path.join(MATRIX_DIR, "Ybus_imag.csv"),
        Ybus,
    )

    np.savetxt(
        os.path.join(MATRIX_DIR, "Bprime.csv"),
        Bprime,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        os.path.join(MATRIX_DIR, "Bdouble_prime.csv"),
        Bdouble,
        delimiter=",",
        fmt="%.12e",
    )

    np.savetxt(
        os.path.join(MATRIX_DIR, "rhs_p_initial.csv"),
        rhs_p0,
        delimiter=",",
        fmt="%.12e",
    )

    with open(os.path.join(MATRIX_DIR, "matrix_info.txt"), "w", encoding="utf-8") as f:
        f.write("IEEE 14-bus matrix export\n")
        f.write("=" * 80 + "\n")
        f.write(f"nbus = {case['nbus']}\n")
        f.write(f"baseMVA = {case['base_mva']}\n")
        f.write(f"slack bus internal index = {case['slack']}\n")
        f.write(f"PV internal indices = {case['pv']}\n")
        f.write(f"PQ internal indices = {case['pq']}\n")
        f.write(f"non-slack internal indices = {case['non_slack']}\n")
        f.write(f"Ybus shape = {Ybus.shape}\n")
        f.write(f"Bprime shape = {Bprime.shape}\n")
        f.write(f"Bdouble_prime shape = {Bdouble.shape}\n")
        f.write(f"rhs_p_initial shape = {rhs_p0.shape}\n")