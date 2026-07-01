import numpy as np

from config import POWER_CASE

from pypower.case6ww import case6ww
from pypower.case9 import case9
from pypower.case14 import case14


# ============================================================
# MATPOWER / PYPOWER column indices
# ============================================================

# ----- Bus matrix columns -----
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 7
VA = 8

# ----- Bus types -----
PQ = 1
PV = 2
REF = 3

# ----- Generator matrix columns -----
GEN_BUS = 0
PG = 1
QG = 2
VG = 5
GEN_STATUS = 7

PMAX = 8
PMIN = 9

# ----- Branch matrix columns -----
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
TAP = 8
SHIFT = 9
BR_STATUS = 10

# ----- Generator cost matrix columns -----
GENCOST_MODEL = 0
GENCOST_NCOST = 3
GENCOST_C2 = 4
GENCOST_C1 = 5
GENCOST_C0 = 6


# ============================================================
# Case loader
# ============================================================

def load_power_case(case_name=None):
    """
    Load a PYPOWER/MATPOWER case.

    Supported cases:
        - "case6ww" : 6-bus case, good for fast OPF testing
        - "case9"   : 9-bus case
        - "case14"  : IEEE 14-bus case

    This function returns a dictionary in the format used by the
    rest of your FDLS / OPF code.
    """

    if case_name is None:
        case_name = POWER_CASE

    case_name = str(case_name).lower().strip()

    if case_name == "case6ww":
        mpc = case6ww()
    elif case_name == "case9":
        mpc = case9()
    elif case_name in ["case14", "ieee14", "ieee14_case"]:
        mpc = case14()
        case_name = "case14"
    else:
        raise ValueError(
            f"Unknown POWER_CASE = {case_name}. "
            "Please use 'case6ww', 'case9', or 'case14'."
        )

    base_mva = float(mpc["baseMVA"])
    bus = np.asarray(mpc["bus"], dtype=float)
    gen = np.asarray(mpc["gen"], dtype=float)
    branch = np.asarray(mpc["branch"], dtype=float)

    if "gencost" not in mpc:
        raise ValueError(
            f"{case_name} does not contain gencost data. "
            "OPF needs generator cost data."
        )

    gencost = np.asarray(mpc["gencost"], dtype=float)

    bus_numbers = bus[:, BUS_I].astype(int)
    bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(bus_numbers)}

    nbus = bus.shape[0]

    slack = np.where(bus[:, BUS_TYPE].astype(int) == REF)[0]
    pv = np.where(bus[:, BUS_TYPE].astype(int) == PV)[0]
    pq = np.where(bus[:, BUS_TYPE].astype(int) == PQ)[0]

    if len(slack) != 1:
        raise ValueError(
            f"This script expects exactly one slack bus, "
            f"but {case_name} has {len(slack)} slack buses."
        )

    slack = int(slack[0])
    non_slack = [i for i in range(nbus) if i != slack]

    return {
        "case_name": case_name,
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
        "bus_id_to_idx": bus_id_to_idx,
        "bus_numbers": bus_numbers,
        "nbus": nbus,
        "slack": slack,
        "pv": list(pv),
        "pq": list(pq),
        "non_slack": non_slack,
    }


def load_ieee14_case():
    """
    Compatibility wrapper.

    Your old main.py calls load_ieee14_case().
    We keep the same function name so you do not need to rewrite main.py.

    The real selected case is controlled by POWER_CASE in config.py.
    """
    return load_power_case(POWER_CASE)


# ============================================================
# Ybus construction
# ============================================================

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

        if abs(r) < 1e-14 and abs(x) < 1e-14:
            raise ValueError(
                f"Branch {f_ext} -> {t_ext} has zero impedance."
            )

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


# ============================================================
# Power specification and voltage initialization
# ============================================================

def make_specified_power(case):
    """
    Compute specified net power injection at each bus.

    P_spec = P_generation - P_demand
    Q_spec = Q_generation - Q_demand

    Values are returned in per-unit.
    """

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
    """
    Return initial voltage magnitude and angle.

    vm: voltage magnitude in pu
    va: voltage angle in radians
    """

    bus = case["bus"]

    vm = bus[:, VM].astype(float).copy()
    va = np.deg2rad(bus[:, VA].astype(float).copy())

    return vm, va


def calculated_power(Ybus, vm, va):
    """
    Calculate injected active and reactive power from Ybus and voltage.

    S = V * conj(Ybus @ V)

    Returns:
        P_calc, Q_calc
    """

    V = vm * np.exp(1j * va)
    S = V * np.conj(Ybus @ V)

    return np.real(S), np.imag(S)


# ============================================================
# FDLS matrices
# ============================================================

def make_fdls_matrices(Ybus, case):
    """
    Fast Decoupled Load Flow approximation.

    B = -Im(Ybus)

    B'  : remove slack bus rows/columns
    B'' : keep only PQ bus rows/columns

    B' solves:
        B' Δθ = ΔP / |V|

    B'' solves:
        B'' ΔV = ΔQ / |V|
    """

    B = -np.imag(Ybus)

    non_slack = case["non_slack"]
    pq = case["pq"]

    Bprime = B[np.ix_(non_slack, non_slack)]
    Bdouble = B[np.ix_(pq, pq)]

    # Quantum solvers usually require symmetric / Hermitian matrices.
    Bprime = 0.5 * (Bprime + Bprime.T)
    Bdouble = 0.5 * (Bdouble + Bdouble.T)

    return Bprime, Bdouble