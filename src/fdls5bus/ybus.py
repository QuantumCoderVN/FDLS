"""Ybus builders for AC power-flow studies."""

from __future__ import annotations

import numpy as np

from .data import PowerSystemCase


def make_ybus(
    case: PowerSystemCase,
    *,
    series_x_only: bool = False,
    include_line_charging: bool = True,
    include_bus_shunts: bool = True,
) -> np.ndarray:
    """Build the complex nodal admittance matrix.

    Parameters
    ----------
    case:
        MATPOWER-like case data.
    series_x_only:
        If True, every branch series admittance is approximated by 1/(j*x),
        i.e. resistance and line charging are ignored. This is used for B'.
    include_line_charging:
        Include branch total charging susceptance b/2 at both ends.
    include_bus_shunts:
        Include bus shunts Gs+jBs converted to per unit on case.base_mva.
    """

    bus = case.bus
    branch = case.branch
    nb = bus.shape[0]
    ybus = np.zeros((nb, nb), dtype=complex)

    for row in branch:
        fbus = int(row[0]) - 1
        tbus = int(row[1]) - 1
        r = float(row[2])
        x = float(row[3])
        b_total = float(row[4]) if include_line_charging else 0.0
        ratio = float(row[8])
        angle_deg = float(row[9])
        status = int(row[10])

        if status == 0:
            continue
        if x == 0:
            raise ValueError("Branch reactance x must be non-zero.")

        if series_x_only:
            y_series = 1.0 / (1j * x)
            b_total = 0.0
        else:
            z_series = r + 1j * x
            if z_series == 0:
                raise ValueError("Branch impedance r+jx must be non-zero.")
            y_series = 1.0 / z_series

        tap_mag = ratio if ratio != 0 else 1.0
        tap = tap_mag * np.exp(1j * np.deg2rad(angle_deg))
        y_shunt = 1j * b_total / 2.0

        ybus[fbus, fbus] += (y_series + y_shunt) / (tap * np.conj(tap))
        ybus[tbus, tbus] += y_series + y_shunt
        ybus[fbus, tbus] += -y_series / np.conj(tap)
        ybus[tbus, fbus] += -y_series / tap

    if include_bus_shunts:
        # MATPOWER stores Gs and Bs as MW/MVAr consumed/injected at V=1.0;
        # convert to per-unit admittance on system baseMVA.
        for i, row in enumerate(bus):
            gs = float(row[4]) / case.base_mva
            bs = float(row[5]) / case.base_mva
            ybus[i, i] += gs + 1j * bs

    return ybus
