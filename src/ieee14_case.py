import numpy as np
from pypower.case14 import case14


# ============================================================
# MATPOWER / PYPOWER column indices
# ============================================================

BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 7
VA = 8

PQ = 1
PV = 2
REF = 3

GEN_BUS = 0
PG = 1
QG = 2
VG = 5
GEN_STATUS = 7

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 8
SHIFT = 9
BR_STATUS = 10


# ============================================================
# IEEE 14-bus data
# ============================================================

def load_ieee14_case():
    mpc = case14()

    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)

    bus_numbers = bus[:, BUS_I].astype(int)
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_numbers)}

    nbus = bus.shape[0]

    slack = np.where(bus[:, BUS_TYPE].astype(int) == REF)[0]
    pv = np.where(bus[:, BUS_TYPE].astype(int) == PV)[0]
    pq = np.where(bus[:, BUS_TYPE].astype(int) == PQ)[0]

    if len(slack) != 1:
        raise ValueError("This script expects exactly one slack bus.")

    slack = int(slack[0])
    non_slack = [i for i in range(nbus) if i != slack]

    return {
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "bus_id_to_idx": bus_id_to_idx,
        "nbus": nbus,
        "slack": slack,
        "pv": list(pv),
        "pq": list(pq),
        "non_slack": non_slack,
    }


def make_ybus(case):
    nbus = case["nbus"]
    base_mva = case["base_mva"]
    bus = case["bus"]
    branch = case["branch"]
    bus_id_to_idx = case["bus_id_to_idx"]

    Ybus = np.zeros((nbus, nbus), dtype=np.complex128)

    for br in branch:
        if int(br[BR_STATUS]) == 0:
            continue

        f_ext = int(br[F_BUS])
        t_ext = int(br[T_BUS])

        f = bus_id_to_idx[f_ext]
        t = bus_id_to_idx[t_ext]

        r = br[BR_R]
        x = br[BR_X]
        b = br[BR_B]

        z = r + 1j * x
        y = 1.0 / z

        tap_mag = br[TAP]
        if abs(tap_mag) < 1e-14:
            tap_mag = 1.0

        shift_rad = np.deg2rad(br[SHIFT])
        tap = tap_mag * np.exp(1j * shift_rad)

        y_shunt = 1j * b / 2.0

        Yff = (y + y_shunt) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        Ytf = -y / tap
        Ytt = y + y_shunt

        Ybus[f, f] += Yff
        Ybus[f, t] += Yft
        Ybus[t, f] += Ytf
        Ybus[t, t] += Ytt

    for i in range(nbus):
        gs = bus[i, GS] / base_mva
        bs = bus[i, BS] / base_mva
        Ybus[i, i] += gs + 1j * bs

    return Ybus


def make_specified_power(case):
    base_mva = case["base_mva"]
    bus = case["bus"]
    gen = case["gen"]
    bus_id_to_idx = case["bus_id_to_idx"]

    P_spec = -bus[:, PD] / base_mva
    Q_spec = -bus[:, QD] / base_mva

    for g in gen:
        if int(g[GEN_STATUS]) == 0:
            continue

        bus_ext = int(g[GEN_BUS])
        idx = bus_id_to_idx[bus_ext]

        P_spec[idx] += g[PG] / base_mva
        Q_spec[idx] += g[QG] / base_mva

    return P_spec, Q_spec


def initial_voltage(case):
    bus = case["bus"]

    vm = bus[:, VM].astype(float).copy()
    va = np.deg2rad(bus[:, VA].astype(float).copy())

    return vm, va


def calculated_power(Ybus, vm, va):
    V = vm * np.exp(1j * va)
    S = V * np.conj(Ybus @ V)

    return np.real(S), np.imag(S)


def make_fdls_matrices(Ybus, case):
    """
    FDLS approximation:

        B = -Im(Ybus)
        B'  = B without slack bus
        B'' = B restricted to PQ buses
    """

    B = -np.imag(Ybus)

    non_slack = case["non_slack"]
    pq = case["pq"]

    Bprime = B[np.ix_(non_slack, non_slack)]
    Bdouble = B[np.ix_(pq, pq)]

    # Quantum solvers require Hermitian / symmetric matrix.
    Bprime = 0.5 * (Bprime + Bprime.T)
    Bdouble = 0.5 * (Bdouble + Bdouble.T)

    return Bprime, Bdouble