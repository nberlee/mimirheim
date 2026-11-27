"""The benchmark's plant: a stratified hot water installation at minute resolution.

The plant is a chain of nodes in flow order, inlet first and outlet last;
the installation of spec §18 has two tanks of two nodes each (tank A bottom
and top, tank B bottom and top). Water moves between nodes only with a draw
(plug flow along the whole chain). Heat moves between the two nodes of the
same tank, never between tanks. Each minute, in this order:

1. heat input: an optional coil system (the heat pump's output, split over
   two coil nodes by a heat-exchanger law and integrated exactly) plus
   direct heat per node (the element);
2. standby loss of each node to the room, from the heated temperatures;
3. exchange within each tank, by the exact two-node solution, followed by
   instant mixing of a tank whose lower node is warmer than its upper node
   (hot water rises; an adopted limiting assumption);
4. the draw: the outflow leaves at the outlet node, every node takes water
   from the node below and the inlet node takes cold water, in shifts no
   larger than the smallest node.

A draw is a heat demand expressed as hot water at 40 C (spec §14.2): above
40 C at the outlet the household mixes down and gets all of it; between
the cold-water temperature and 40 C it takes the volume it would at 40 C
and gets less heat; at or below the cold-water temperature it gets none.

``reconstruct_draws`` infers the draw heat per quarter from two probe
readings at each quarter's end (spec §14.1), with heat computed along each
candidate's own trajectory by a caller-supplied heat provider.

This module does not fit parameters, compute COPs or decide when to heat.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KWH_PER_LITRE_KELVIN = 4.186 / 3600.0
"""Heat to warm one litre of water by one kelvin, in kWh."""

COMFORT_TEMP_C = 40.0
"""Hot water temperature a draw is expressed at (spec §14.2)."""

_MINUTE_H = 1.0 / 60.0
_COLD_MIN_C, _COLD_MAX_C = 5.0, 20.0
_TIE = 1e-9  # objective values closer than this (K^2) are treated as equal


class TankPair(BaseModel):
    """The two nodes of one tank and the heat exchange between them.

    Attributes:
        lower: Index of the tank's lower node.
        upper: Index of its upper node (the next node in flow order).
        k_kw_per_k: Heat exchange between them per kelvin of difference.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    lower: int = Field(ge=0)
    upper: int = Field(ge=1)
    k_kw_per_k: float = Field(ge=0.0)


class PlantParams(BaseModel):
    """Installation parameters.

    Attributes:
        volumes_l: Node volumes in litres, inlet first, outlet last.
        tank_pairs: The tanks, each a pair of neighbouring nodes; pairs do
            not share nodes. Heat moves only within a pair.
        loss_kw_per_k: Standby loss of each node to the room, in kW/K.
        room_temp_c: Room temperature around the tanks.
        cold_temp_c: Cold-water temperature, clipped to 5-20 C (spec §15.2)
            so that a 40 C draw always has a positive volume.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    volumes_l: list[float] = Field(min_length=2)
    tank_pairs: list[TankPair]
    loss_kw_per_k: list[float]
    room_temp_c: float
    cold_temp_c: float

    @field_validator("volumes_l")
    @classmethod
    def _positive_volumes(cls, volumes: list[float]) -> list[float]:
        """Every node must hold water."""
        if any(v <= 0.0 for v in volumes):
            raise ValueError("node volumes must be positive")
        return volumes

    @field_validator("cold_temp_c")
    @classmethod
    def _clip_cold(cls, value: float) -> float:
        """Keep the cold-water temperature in the admissible range."""
        return min(_COLD_MAX_C, max(_COLD_MIN_C, value))

    @model_validator(mode="after")
    def _consistent(self) -> PlantParams:
        """Per-node lists match; tank pairs are neighbouring, valid and disjoint."""
        n = len(self.volumes_l)
        if len(self.loss_kw_per_k) != n:
            raise ValueError("loss_kw_per_k needs one value per node")
        if any(u < 0.0 for u in self.loss_kw_per_k):
            raise ValueError("loss_kw_per_k must be non-negative")
        used: set[int] = set()
        for pair in self.tank_pairs:
            if pair.upper != pair.lower + 1 or pair.upper >= n:
                raise ValueError("a tank pair must be two neighbouring nodes")
            if {pair.lower, pair.upper} & used:
                raise ValueError("tank pairs must not share nodes")
            used |= {pair.lower, pair.upper}
        return self


@dataclass(frozen=True)
class PlantState:
    """Node temperatures in degrees Celsius, inlet first."""

    temps_c: list[float]


@dataclass(frozen=True)
class CoilSystem:
    """The heat pump's output over two coil nodes for one minute.

    The output ``q_kw`` is constant over the minute. The first coil takes
    ``alpha x q + beta x (T_second - T_first)`` and the second the rest, so
    the split shifts towards the colder coil as the minute goes on. With
    series coils of effectiveness ``eps_1, eps_2`` and circuit heat-capacity
    flow ``K``: ``alpha = eps_1 / D`` and ``beta = K eps_1 eps_2 / D`` with
    ``D = eps_1 + eps_2 - eps_1 eps_2``.

    Attributes:
        first_node: Node the circuit reaches first.
        second_node: Node it reaches second.
        q_kw: Heat pump output for the minute, in kW.
        alpha: Share of the output to the first coil at equal temperatures.
        beta_kw_per_k: Exchange between the coils per kelvin.
    """

    first_node: int
    second_node: int
    q_kw: float
    alpha: float
    beta_kw_per_k: float


@dataclass(frozen=True)
class HeatInput:
    """Heat entering the plant in one minute.

    Attributes:
        direct_kw: Heat per node that does not depend on the state (the
            element), in kW.
        coil: The coil system, or None when the heat pump does not heat.
    """

    direct_kw: list[float]
    coil: CoilSystem | None = None


@dataclass(frozen=True)
class DrawResult:
    """Heat accounting of one minute's draw, in kWh."""

    requested_kwh: float
    delivered_kwh: float
    unmet_kwh: float


def _coil_minute(temps: list[float], capacity: list[float], coil: CoilSystem) -> None:
    """Integrate a coil system exactly over one minute, in place.

    The two coil nodes gain ``q`` together, so their capacity-weighted sum
    rises linearly. Their difference ``d = T_first - T_second`` obeys
    ``dd/dt = alpha q / C_1 - (1 - alpha) q / C_2 - beta (1/C_1 + 1/C_2) d``,
    which relaxes exponentially to its forced equilibrium: no overshoot for
    any parameters.
    """
    i, j = coil.first_node, coil.second_node
    c1, c2 = capacity[i], capacity[j]
    energy = c1 * temps[i] + c2 * temps[j] + coil.q_kw * _MINUTE_H
    drive = coil.alpha * coil.q_kw / c1 - (1.0 - coil.alpha) * coil.q_kw / c2
    rate = coil.beta_kw_per_k * (1.0 / c1 + 1.0 / c2)
    d0 = temps[i] - temps[j]
    if rate > 0.0:
        d_inf = drive / rate
        d = d_inf + (d0 - d_inf) * math.exp(-rate * _MINUTE_H)
    else:
        d = d0 + drive * _MINUTE_H
    # Solve c1 T1 + c2 T2 = energy and T1 - T2 = d.
    temps[j] = (energy - c1 * d) / (c1 + c2)
    temps[i] = temps[j] + d


def _exchange_and_mix(temps: list[float], capacity: list[float], pairs: Sequence[TankPair]) -> None:
    """Exact exchange within each tank, then instant mixing of an inverted tank, in place."""
    for pair in pairs:
        lo, hi = pair.lower, pair.upper
        mean = (capacity[lo] * temps[lo] + capacity[hi] * temps[hi]) / (capacity[lo] + capacity[hi])
        if temps[lo] > temps[hi]:
            temps[lo] = temps[hi] = mean  # hot water rises: the tank mixes
            continue
        if pair.k_kw_per_k > 0.0:
            decay = math.exp(-pair.k_kw_per_k * (1.0 / capacity[lo] + 1.0 / capacity[hi]) * _MINUTE_H)
            temps[lo] = mean + (temps[lo] - mean) * decay
            temps[hi] = mean + (temps[hi] - mean) * decay


def step_minute(
    params: PlantParams,
    state: PlantState,
    heat: HeatInput,
    draw_kwh: float,
) -> tuple[PlantState, DrawResult]:
    """Advance the plant by one minute.

    Args:
        params: Installation parameters.
        state: Node temperatures at the start of the minute.
        heat: Heat entering in this minute.
        draw_kwh: Hot water heat demanded in this minute, at 40 C.

    Returns:
        The state at the end of the minute and the draw accounting.
    """
    temps = list(state.temps_c)
    n = len(temps)
    capacity = [v * KWH_PER_LITRE_KELVIN for v in params.volumes_l]  # kWh/K per node

    # 1. Heat input: the coil system exactly, direct heat per node.
    if heat.coil is not None:
        _coil_minute(temps, capacity, heat.coil)
    for i in range(n):
        temps[i] += heat.direct_kw[i] * _MINUTE_H / capacity[i]
    # 2. Standby loss, from the temperatures after heating.
    for i in range(n):
        temps[i] -= params.loss_kw_per_k[i] * (temps[i] - params.room_temp_c) * _MINUTE_H / capacity[i]
    # 3. Exchange within each tank and instant mixing of an inverted tank.
    _exchange_and_mix(temps, capacity, params.tank_pairs)
    # 4. Draw by plug flow.
    temps, draw = _draw(params, temps, max(0.0, draw_kwh))
    return PlantState(temps_c=temps), draw


def _draw(params: PlantParams, temps: list[float], requested: float) -> tuple[list[float], DrawResult]:
    """Remove a 40 C heat demand at the outlet by plug flow; see the module docstring."""
    cold = params.cold_temp_c
    max_shift = min(params.volumes_l)
    total_volume = sum(params.volumes_l)
    remaining = requested
    delivered = 0.0
    moved = 0.0
    while remaining > 1e-12 and moved < total_volume - 1e-9:
        outlet = temps[-1]
        if outlet <= cold + 1e-9:
            break
        # Per litre taken: how much of the demand it covers, and how much heat it gives.
        covers = KWH_PER_LITRE_KELVIN * (max(outlet, COMFORT_TEMP_C) - cold)
        gives = KWH_PER_LITRE_KELVIN * (outlet - cold)
        litres = min(max_shift, remaining / covers, total_volume - moved)
        # Shift every node up by `litres`, using the temperatures before the shift.
        below = [cold] + temps[:-1]
        temps = [
            (t * (v - litres) + b * litres) / v for t, b, v in zip(temps, below, params.volumes_l, strict=True)
        ]
        remaining -= litres * covers
        delivered += litres * gives
        moved += litres
    return temps, DrawResult(requested_kwh=requested, delivered_kwh=delivered, unmet_kwh=requested - delivered)


HeatProvider = Callable[[int, PlantState], HeatInput]
"""Heat for a minute (index from the start of the reconstruction) given the current state."""


@dataclass(frozen=True)
class Reconstruction:
    """Draws inferred per quarter.

    Attributes:
        draws_kwh: Draw heat per quarter (kWh at 40 C): fitted where both
            probes are known, the caller's fallback where one is missing.
        fitted: Whether each quarter's draw was fitted to the probes.
        residual_k: Root-mean-square probe error at each quarter's end, or
            None where a probe is missing.
        states: Model state at each quarter's end.
    """

    draws_kwh: list[float]
    fitted: list[bool]
    residual_k: list[float | None]
    states: list[PlantState] = field(default_factory=list)


def _run_quarter(
    params: PlantParams, state: PlantState, provider: HeatProvider, first_minute: int, draw_kwh: float
) -> PlantState:
    """Run the 15 minutes of one quarter with the draw spread evenly, heat from the provider."""
    for m in range(15):
        state, _ = step_minute(params, state, provider(first_minute + m, state), draw_kwh / 15)
    return state


def reconstruct_draws(
    params: PlantParams,
    initial: PlantState,
    provider: HeatProvider,
    probes: Sequence[tuple[float | None, float | None]],
    lower_node: int,
    top_node: int,
    fallback_draws_kwh: Sequence[float | None] | None = None,
) -> Reconstruction:
    """Infer the draw heat per quarter from the probes at each quarter's end.

    For each quarter the draw heat ``q >= 0`` (spread evenly over its minutes)
    is chosen that minimises the squared error of the observed nodes against
    the lower and top probes at the quarter's end (spec §14.1). Every
    candidate runs its own trajectory from the quarter's start state, with
    heat from ``provider`` along that trajectory; only the chosen draw
    advances the state. The search covers fifteen per-minute volume caps on
    a geometric grid and resolves ties to the smaller draw.

    Args:
        params: Installation parameters.
        initial: State at the start of the first quarter.
        provider: Heat per minute given the current state.
        probes: ``(lower, top)`` probe readings at each quarter's end.
        lower_node: Node the lower probe observes.
        top_node: Node the top probe observes.
        fallback_draws_kwh: Draw heat for a quarter whose probes are missing
            (a bridged gap, spec §14.3). A missing probe without a fallback
            is an error: the caller must end the segment there.

    Returns:
        The reconstruction.

    Raises:
        ValueError: If a probe node does not exist, the fallback list has the
            wrong length, or a missing probe has no fallback.
    """
    n = len(params.volumes_l)
    if not (0 <= lower_node < n and 0 <= top_node < n):
        raise ValueError("probe nodes must index plant nodes")
    if fallback_draws_kwh is not None and len(fallback_draws_kwh) != len(probes):
        raise ValueError("fallback_draws_kwh needs one value per probe reading")
    state = initial
    draws: list[float] = []
    fitted: list[bool] = []
    residuals: list[float | None] = []
    states: list[PlantState] = []
    for q, (lower, top) in enumerate(probes):
        first = 15 * q
        if lower is None or top is None:
            fallback = None if fallback_draws_kwh is None else fallback_draws_kwh[q]
            if fallback is None:
                raise ValueError(f"probe missing at quarter {q} and no fallback draw: end the segment here")
            state = _run_quarter(params, state, provider, first, max(0.0, fallback))
            draws.append(max(0.0, fallback))
            fitted.append(False)
            residuals.append(None)
            states.append(state)
            continue

        def error(draw: float, start: PlantState = state, first: int = first, lower: float = lower, top: float = top) -> float:
            end = _run_quarter(params, start, provider, first, draw)
            return (end.temps_c[lower_node] - lower) ** 2 + (end.temps_c[top_node] - top) ** 2

        # The largest demand that can still change the plant: in every minute
        # the whole volume at the 40 C-or-hotter rate, with a temperature
        # ceiling well above anything reachable (the volume cap is per minute,
        # and requested heat can exceed the heat delivered when the outlet is
        # lukewarm, spec §14.2). Beyond it the error no longer changes.
        ceiling = max([*state.temps_c, COMFORT_TEMP_C]) + 60.0
        per_minute = sum(params.volumes_l) * KWH_PER_LITRE_KELVIN * (ceiling - params.cold_temp_c)
        best = _minimise(error, 0.0, 15 * per_minute + 0.1)
        state = _run_quarter(params, state, provider, first, best)
        draws.append(best)
        fitted.append(True)
        residuals.append((((state.temps_c[lower_node] - lower) ** 2 + (state.temps_c[top_node] - top) ** 2) / 2.0) ** 0.5)
        states.append(state)
    return Reconstruction(draws_kwh=draws, fitted=fitted, residual_k=residuals, states=states)


def _minimise(
    fn: Callable[[float], float], lo: float, hi: float, grid: int = 48, iterations: int = 40, smallest: float = 0.01
) -> float:
    """Minimise a function of one variable on ``[lo, hi]``.

    The objective need not be unimodal, so a single golden-section search is
    not enough. The search evaluates a grid of ``lo`` plus geometrically
    spaced points from ``lo + smallest`` to ``hi`` (fine near ``lo``, where
    small draws lie), refines every local minimum of the grid by golden
    section between its neighbours, and returns the best point found. Ties
    go to the smaller argument (the smallest draw that explains the probes,
    for instance on the plateau beyond the volume cap).

    Args:
        fn: The objective.
        lo: Lower bound.
        hi: Upper bound.
        grid: Number of geometric grid points.
        iterations: Golden-section iterations per local minimum.
        smallest: First grid step above ``lo``.

    Returns:
        The argument with the lowest objective value found.
    """
    if hi <= lo + smallest:
        points = [lo, hi]
    else:
        ratio = (hi - lo) / smallest
        points = [lo] + [lo + smallest * ratio ** (k / (grid - 1)) for k in range(grid)]
    values = [fn(x) for x in points]
    best_x, best_v = points[0], values[0]
    golden = (5 ** 0.5 - 1) / 2
    for i, v in enumerate(values):
        left = values[i - 1] if i > 0 else float("inf")
        right = values[i + 1] if i + 1 < len(values) else float("inf")
        if v > left or v > right:
            continue  # not a local minimum of the grid
        a = points[max(0, i - 1)]
        b = points[min(len(points) - 1, i + 1)]
        c, d = b - golden * (b - a), a + golden * (b - a)
        fc, fd = fn(c), fn(d)
        for _ in range(iterations):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - golden * (b - a)
                fc = fn(c)
            else:
                a, c, fc = c, d, fd
                d = a + golden * (b - a)
                fd = fn(d)
        for x, fx in ((points[i], v), ((a + b) / 2, fn((a + b) / 2))):
            # Values within _TIE of each other count as equal (squared probe
            # error in K^2), and then the smaller draw wins.
            if fx < best_v - _TIE or (abs(fx - best_v) <= _TIE and x < best_x):
                best_x, best_v = x, fx
    return best_x
