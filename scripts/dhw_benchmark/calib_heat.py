"""The heat input model the tank calibration fits (benchmark ticket 03, plan revisions 8-11).

Heat enters the tanks only while the heat pump makes hot water (modes hot
water and legionella):

- the compressor drives a closed coil circuit through one coil low in each
  tank, assumed in series (the order is a variant). Its output
  ``Q = p x COP`` is fixed for the minute, from the state at the minute's
  start; the split over the coils follows a heat-exchanger law (coil
  effectiveness ``eps_1, eps_2``, circuit heat-capacity flow ``K``) and is
  integrated exactly by the plant;
- the element heats tank A's top node with its electric power.

Compressor electricity is the measured total minus the element minus the
auxiliary power (pumps and controls, estimated from quiet idle minutes).
Minutes whose heat cannot be determined are ``INVALID``: missing power or
element readings, an unknown compressor state while making hot water, a
missing source temperature while the coil heats, or a residual below
-0.1 kW.

The COP is ``cop_ref + c1 (source - 10 C) + c2 (circuit - 45 C)``, solved
together with the circuit supply temperature and capped to 1-8 (plan
amendment AL). Downstream, inputs outside the calibrated domain are clamped
inside the solve so the balance stays consistent (plan amendment AG).

This module evaluates the model; fitting is done in ``calib_fit``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Final, Literal

from dhw_benchmark.inputs import TankRow
from dhw_benchmark.plant import CoilSystem, HeatInput

CoilOrder = Literal["A", "B"]

SOURCE_REF_C = 10.0
CIRCUIT_REF_C = 45.0
COP_MIN = 1.0
COP_MAX = 8.0
"""Every COP used lies in this range (plan amendments AK and AL: the installation reaches COP 8)."""
A_LOW, A_HIGH, B_LOW, B_HIGH = 0, 1, 2, 3
"""Node indices of the two-tank plant (spec §18)."""

_AUX_MIN_MINUTES = 60
_NEGATIVE_LIMIT_KW = -0.1
_ROUNDING_KW = 1e-9  # a residual exactly at the limit is allowed despite rounding
_HEATING_MODES = ("dhw", "legionella")


class _Invalid:
    """Marker for a minute whose heat input cannot be determined."""

    def __repr__(self) -> str:
        return "INVALID"


INVALID: Final = _Invalid()
"""Returned when a minute's heat input cannot be determined; it splits a window."""


def estimate_aux_kw(rows: Sequence[TankRow]) -> tuple[float, bool]:
    """Auxiliary electric power from idle minutes with compressor and element off.

    Args:
        rows: Minute rows (the training data of one fold).

    Returns:
        ``(aux_kw, flagged)``: the median power of those minutes, or 0 and
        ``flagged`` when there are fewer than 60 of them.
    """
    quiet = [
        r.hp_electric_kw
        for r in rows
        if r.mode == "idle" and r.compressor_on is False and r.element_kw == 0.0 and r.hp_electric_kw is not None
    ]
    if len(quiet) < _AUX_MIN_MINUTES:
        return 0.0, True
    return float(median(quiet)), False


def _coil_active(row: TankRow) -> bool:
    """Whether the heat pump heats the tanks in this minute."""
    return bool(row.compressor_on) and row.mode in _HEATING_MODES


def compressor_electric_kw(row: TankRow, aux_kw: float) -> float | _Invalid:
    """Compressor electricity heating the tanks in one minute, in kW.

    Args:
        row: The minute.
        aux_kw: Auxiliary power to subtract.

    Returns:
        The power; 0 outside coil-active minutes; ``INVALID`` when, in a hot
        water or legionella minute, the compressor state is unknown, or power
        is missing or the residual is below -0.1 kW.
    """
    if row.mode in _HEATING_MODES and row.compressor_on is None:
        return INVALID  # coil activity unknown
    if not _coil_active(row):
        return 0.0
    if row.hp_electric_kw is None or row.element_kw is None:
        return INVALID
    residual = row.hp_electric_kw - row.element_kw - aux_kw
    if residual < _NEGATIVE_LIMIT_KW - _ROUNDING_KW:
        return INVALID
    return max(0.0, residual)


@dataclass(frozen=True)
class CopModel:
    """Effective COP ``cop_ref + c1 (source - 10 C) + c2 (circuit - 45 C)``.

    Attributes:
        cop_ref: COP at 10 C source and 45 C circuit supply.
        c1: Change per kelvin of source temperature.
        c2: Change per kelvin of circuit supply temperature.
    """

    cop_ref: float
    c1: float
    c2: float

    def cop(self, source_c: float, circuit_c: float) -> float:
        """The COP at a source and circuit supply temperature.

        Args:
            source_c: Source temperature.
            circuit_c: Circuit supply temperature.

        Returns:
            The effective COP.
        """
        return self.cop_ref + self.c1 * (source_c - SOURCE_REF_C) + self.c2 * (circuit_c - CIRCUIT_REF_C)


@dataclass(frozen=True)
class CoilLaw:
    """The coil circuit, assumed to run through both coils in series.

    Attributes:
        eps_1: Effectiveness of the coil the circuit reaches first.
        eps_2: Effectiveness of the second coil.
        flow_kw_per_k: Heat-capacity flow of the circuit, in kW/K.
        first: Which tank's coil comes first (``A`` or ``B``).
    """

    eps_1: float
    eps_2: float
    flow_kw_per_k: float
    first: CoilOrder


@dataclass(frozen=True)
class CopDomain:
    """Calibrated range of source and circuit supply temperature."""

    source_min: float
    source_max: float
    circuit_min: float
    circuit_max: float


@dataclass(frozen=True)
class CoilSolution:
    """The coil circuit for one minute.

    Attributes:
        coil: The coil system the plant integrates.
        t_in_c: Circuit supply temperature at the minute's start.
        cop: The COP used for the minute, within 1-8.
        clamped: Whether downstream domain clamping changed the solve.
        capped: Whether the COP was capped to 1-8.
        uncapped_cop: The COP before the cap (after domain clamping), for
            reporting the excess.
    """

    coil: CoilSystem
    t_in_c: float
    cop: float
    clamped: bool
    capped: bool
    uncapped_cop: float


def solve_coils(
    power_kw: float,
    source_c: float,
    t_first_c: float,
    t_second_c: float,
    cop: CopModel,
    law: CoilLaw,
    domain: CopDomain | None = None,
) -> CoilSolution:
    """Solve the circuit supply temperature and the COP together for one minute.

    With ``D = eps_1 + eps_2 - eps_1 eps_2`` and
    ``a = cop_ref + c1 (source - 10) - 45 c2``, the coil balance
    ``Q = eps_1 K (T_in - T_1) + eps_2 K ((1 - eps_1) T_in + eps_1 T_1 - T_2)``
    and ``Q = p (a + c2 T_in)`` give
    ``T_in = (p a / K + eps_1 (1 - eps_2) T_1 + eps_2 T_2) / (D - p c2 / K)``
    (the denominator is positive because ``c2 <= 0``).

    The order (plan amendment AL): with a domain (downstream use) the source
    is clamped to its range first; the joint solve follows; a COP outside 1-8
    is fixed at the violated limit; with a domain, a ``T_in`` outside the
    circuit range fixes the COP at the edge value, capped to 1-8 again.
    ``T_in`` is then recomputed from the coil balance at ``Q = p x COP``, so
    the balance stays consistent and every COP used lies in 1-8.

    Args:
        power_kw: Compressor electricity, positive.
        source_c: Source temperature.
        t_first_c: Temperature of the node at the first coil.
        t_second_c: Temperature of the node at the second coil.
        cop: The COP model.
        law: The coil circuit.
        domain: Calibrated domain for downstream clamping, or None.

    Returns:
        The coil solution.
    """
    e1, e2, k = law.eps_1, law.eps_2, law.flow_kw_per_k
    d = e1 + e2 - e1 * e2
    base = e1 * (1.0 - e2) * t_first_c + e2 * t_second_c
    clamped = False
    source = source_c
    if domain is not None:
        source = min(domain.source_max, max(domain.source_min, source_c))
        clamped = source != source_c
    a = cop.cop_ref + cop.c1 * (source - SOURCE_REF_C) - CIRCUIT_REF_C * cop.c2
    t_in = (power_kw * a / k + base) / (d - power_kw * cop.c2 / k)
    value = a + cop.c2 * t_in
    uncapped = value
    if not COP_MIN <= value <= COP_MAX:
        value = min(COP_MAX, max(COP_MIN, value))
        t_in = (power_kw * value / k + base) / d
    if domain is not None and not domain.circuit_min <= t_in <= domain.circuit_max:
        edge = min(domain.circuit_max, max(domain.circuit_min, t_in))
        uncapped = cop.cop(source, edge)
        value = min(COP_MAX, max(COP_MIN, uncapped))
        t_in = (power_kw * value / k + base) / d
        clamped = True
    first, second = (A_LOW, B_LOW) if law.first == "A" else (B_LOW, A_LOW)
    coil = CoilSystem(
        first_node=first,
        second_node=second,
        q_kw=power_kw * value,
        alpha=e1 / d,
        beta_kw_per_k=k * e1 * e2 / d,
    )
    return CoilSolution(
        coil=coil, t_in_c=t_in, cop=value, clamped=clamped, capped=value != uncapped, uncapped_cop=uncapped
    )


def heat_input_valid(row: TankRow, aux_kw: float) -> bool:
    """Whether a minute's heat input can be determined (the checks of ``heat_input_detail``).

    Args:
        row: The minute.
        aux_kw: Auxiliary power.

    Returns:
        False when, in a hot water or legionella minute, the element reading
        is missing, the compressor electricity is invalid, or the compressor
        heats without a source temperature.
    """
    if row.mode not in _HEATING_MODES:
        return True
    if row.element_kw is None:
        return False
    electric = compressor_electric_kw(row, aux_kw)
    if isinstance(electric, _Invalid):
        return False
    return electric <= 0.0 or row.brine_in_c is not None


def heat_input(
    row: TankRow,
    aux_kw: float,
    temps_c: Sequence[float],
    cop: CopModel,
    law: CoilLaw,
    domain: CopDomain | None = None,
) -> HeatInput | _Invalid:
    """The heat entering the two-tank plant in one minute (see ``heat_input_detail``).

    Args:
        row: The minute.
        aux_kw: Auxiliary power.
        temps_c: Node temperatures at the minute's start.
        cop: The COP model.
        law: The coil circuit.
        domain: Calibrated domain for downstream clamping, or None.

    Returns:
        The heat input, or ``INVALID``.
    """
    detail = heat_input_detail(row, aux_kw, temps_c, cop, law, domain)
    return detail if isinstance(detail, _Invalid) else detail[0]


def heat_input_detail(
    row: TankRow,
    aux_kw: float,
    temps_c: Sequence[float],
    cop: CopModel,
    law: CoilLaw,
    domain: CopDomain | None = None,
) -> tuple[HeatInput, CoilSolution | None] | _Invalid:
    """The heat entering the two-tank plant in one minute, with the coil solution.

    Args:
        row: The minute.
        aux_kw: Auxiliary power.
        temps_c: Node temperatures at the minute's start (A_low, A_high,
            B_low, B_high).
        cop: The COP model.
        law: The coil circuit.
        domain: Calibrated domain for downstream clamping, or None.

    Returns:
        ``(heat input, coil solution or None)``, or ``INVALID`` when the
        heat cannot be determined.
    """
    if row.mode not in _HEATING_MODES:
        return HeatInput(direct_kw=[0.0] * 4), None
    if not heat_input_valid(row, aux_kw):
        return INVALID
    electric = compressor_electric_kw(row, aux_kw)
    assert row.element_kw is not None and not isinstance(electric, _Invalid)  # checked above
    direct = [0.0, row.element_kw, 0.0, 0.0]
    if electric <= 0.0 or row.brine_in_c is None:
        return HeatInput(direct_kw=direct), None
    t_first, t_second = (temps_c[A_LOW], temps_c[B_LOW]) if law.first == "A" else (temps_c[B_LOW], temps_c[A_LOW])
    solved = solve_coils(electric, row.brine_in_c, t_first, t_second, cop, law, domain)
    return HeatInput(direct_kw=direct, coil=solved.coil), solved
