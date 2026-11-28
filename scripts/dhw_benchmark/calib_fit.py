"""Fit the four-node tank model to draw-free windows (benchmark ticket 03, revisions 8-13).

The plant has two tanks in flow order: A (180 L, nodes A_low and A_high) and
B (200 L, nodes B_low and B_high). Nine parameters are fitted together by
bounded least squares on both probes per minute: each window starts from the
probes (hidden nodes per plan amendment AA) and runs forward through the
plant (``plant.step_minute``) with the heat input model (``calib_heat``) and
no draws. Five fixed starting points are tried; the lowest error wins.

The COP of every minute is capped to 1-8 inside the heat input model (plan
amendment AL), so every parameter set is admissible. How often the cap
binds, and by how much, is reported; a share above 10 % of coil-active
minutes marks the COP law as strained (a diagnostic, not a label change).

Identification is a local Jacobian diagnostic (plan amendments K, M and AM):
with columns scaled to unit length, the Jacobian must have full rank and a
condition number below 1e6; parameters loading above 0.3 on a singular
direction with a singular value below 1e-6 of the largest are named weak.
Parameter uncertainty is not assessed (no profiles, amendment AM).

This module fits one set of windows; cross-validation, labels and the
thermostat check live elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from dhw_benchmark.calib_heat import (
    A_HIGH,
    A_LOW,
    B_HIGH,
    B_LOW,
    COP_MAX,
    COP_MIN,
    CoilLaw,
    CoilOrder,
    CopModel,
    heat_input_detail,
)
from dhw_benchmark.inputs import TankRow
from dhw_benchmark.plant import PlantParams, PlantState, TankPair, step_minute

TANK_A_L = 180.0
TANK_B_L = 200.0
"""Tank volumes (spec §16, §18)."""

PARAMETER_NAMES = ("eps_1", "eps_2", "flow_kw_per_k", "k_a", "k_b", "loss_kw_per_k", "cop_ref", "c1", "c2")
BOUNDS: dict[str, tuple[float, float]] = {
    "eps_1": (0.01, 1.0),
    "eps_2": (0.01, 1.0),
    "flow_kw_per_k": (0.05, 10.0),
    "k_a": (0.0, 0.5),
    "k_b": (0.0, 0.5),
    "loss_kw_per_k": (0.0005, 0.02),
    "cop_ref": (1.0, 8.0),
    "c1": (0.0, 0.5),
    "c2": (-0.5, 0.0),
}
"""Fit ranges (plan amendments Q and AK), recorded as assumptions."""

STRAINED_SHARE = 0.10
"""Share of capped coil-active minutes above which the COP law is strained (AL)."""

_LOWER_OFFSET_K = 2.0  # alternative placement: A_low starts this far below the lower probe (AA)
_CONDITION_LIMIT = 1e6
_WEAK_SINGULAR = 1e-6
_WEAK_LOADING = 0.3


@dataclass(frozen=True)
class FitSettings:
    """Fixed values and variant choices for a fit.

    Attributes:
        room_temp_c: Room temperature around the tanks.
        cold_temp_c: Cold-water temperature (does not affect draw-free fits,
            except as the floor of the alternative initial state).
        aux_kw: Auxiliary power subtracted from the electric power.
        coil_order: Which tank's coil the circuit reaches first.
        lower_node: Node the lower probe observes: ``A_LOW`` (central) or
            ``A_HIGH`` (sensitivity scenario).
        split_a: Share of tank A's volume in its lower node.
        split_b: Share of tank B's volume in its lower node.
    """

    room_temp_c: float
    cold_temp_c: float
    aux_kw: float
    coil_order: CoilOrder
    lower_node: int
    split_a: float = 0.5
    split_b: float = 0.5

    def __post_init__(self) -> None:
        """Reject a probe placement or split the model does not define."""
        if self.lower_node not in (A_LOW, A_HIGH):
            raise ValueError("the lower probe observes A_low or A_high")
        if not (0.0 < self.split_a < 1.0 and 0.0 < self.split_b < 1.0):
            raise ValueError("node splits must lie strictly between 0 and 1")


@dataclass(frozen=True)
class FourNodeParams:
    """The fitted parameters of the four-node model.

    Attributes:
        eps_1: Effectiveness of the coil the circuit reaches first.
        eps_2: Effectiveness of the second coil.
        flow_kw_per_k: Heat-capacity flow of the coil circuit, in kW/K.
        k_a: Exchange between tank A's nodes per kelvin.
        k_b: Exchange between tank B's nodes per kelvin.
        loss_kw_per_k: Total standby loss, split over the nodes by volume.
        cop_ref: COP at 10 C source and 45 C circuit supply.
        c1: COP change per kelvin of source temperature.
        c2: COP change per kelvin of circuit supply temperature.
    """

    eps_1: float
    eps_2: float
    flow_kw_per_k: float
    k_a: float
    k_b: float
    loss_kw_per_k: float
    cop_ref: float
    c1: float
    c2: float

    def vector(self) -> list[float]:
        """The parameters in ``PARAMETER_NAMES`` order."""
        return [getattr(self, name) for name in PARAMETER_NAMES]

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> FourNodeParams:
        """Parameters from a vector in ``PARAMETER_NAMES`` order.

        Args:
            values: Nine numbers.

        Returns:
            The parameters.
        """
        return cls(**{name: float(v) for name, v in zip(PARAMETER_NAMES, values, strict=True)})

    def plant(self, settings: FitSettings) -> PlantParams:
        """The plant these parameters describe.

        Args:
            settings: Room and cold-water temperatures and the node splits.

        Returns:
            Plant parameters with nodes A_low, A_high, B_low, B_high.
        """
        volumes = [
            TANK_A_L * settings.split_a,
            TANK_A_L * (1.0 - settings.split_a),
            TANK_B_L * settings.split_b,
            TANK_B_L * (1.0 - settings.split_b),
        ]
        total = TANK_A_L + TANK_B_L
        return PlantParams(
            volumes_l=volumes,
            tank_pairs=[
                TankPair(lower=A_LOW, upper=A_HIGH, k_kw_per_k=self.k_a),
                TankPair(lower=B_LOW, upper=B_HIGH, k_kw_per_k=self.k_b),
            ],
            loss_kw_per_k=[self.loss_kw_per_k * v / total for v in volumes],
            room_temp_c=settings.room_temp_c,
            cold_temp_c=settings.cold_temp_c,
        )

    def cop(self) -> CopModel:
        """The COP model of these parameters."""
        return CopModel(self.cop_ref, self.c1, self.c2)

    def law(self, settings: FitSettings) -> CoilLaw:
        """The coil circuit of these parameters.

        Args:
            settings: The coil order.

        Returns:
            The coil law.
        """
        return CoilLaw(self.eps_1, self.eps_2, self.flow_kw_per_k, settings.coil_order)


STARTS: tuple[FourNodeParams, ...] = (
    FourNodeParams(0.5, 0.5, 1.0, 0.05, 0.05, 0.004, 4.0, 0.05, -0.1),
    FourNodeParams(0.2, 0.8, 0.3, 0.01, 0.2, 0.002, 3.0, 0.0, -0.05),
    FourNodeParams(0.8, 0.2, 3.0, 0.2, 0.01, 0.008, 5.0, 0.1, -0.2),
    FourNodeParams(0.9, 0.9, 0.1, 0.1, 0.1, 0.012, 6.0, 0.2, -0.3),
    FourNodeParams(0.1, 0.4, 8.0, 0.3, 0.3, 0.001, 2.5, 0.02, -0.02),
)
"""The five fixed starting points of the multi-start fit (plan amendment Y)."""


def initial_state(row: TankRow, settings: FitSettings) -> PlantState:
    """The node temperatures at a window's first minute (plan amendment AA).

    Central placement (lower probe in A_low): A_high and B_low lie one and
    two thirds of the way from the lower to the top probe. Alternative (lower
    probe in A_high): A_low is 2 K below the lower probe, but not below the
    cold-water temperature, and B_low lies halfway between the probes.

    Args:
        row: The first minute.
        settings: The probe placement.

    Returns:
        The state.

    Raises:
        ValueError: If a probe reading is missing.
    """
    lower, top = row.t_lower_c, row.t_top_c
    if lower is None or top is None:
        raise ValueError(f"window starts without probe readings at {row.ts.isoformat()}")
    if settings.lower_node == A_LOW:
        span = top - lower
        return PlantState(temps_c=[lower, lower + span / 3.0, lower + 2.0 * span / 3.0, top])
    bottom = max(settings.cold_temp_c, lower - _LOWER_OFFSET_K)
    return PlantState(temps_c=[bottom, lower, (lower + top) / 2.0, top])


@dataclass
class WindowSimulation:
    """One window run forward with a parameter set.

    Attributes:
        residuals: Predicted minus measured probes after every minute,
            ``[lower_1, top_1, lower_2, top_2, ...]`` in kelvin.
        uncapped_cops: COP before the 1-8 cap, per coil-active minute.
        sources_c: Source temperature of those minutes.
        t_in_c: Start-of-minute circuit supply temperature of those minutes.
        t_in_shift_k: Change of the supply temperature over each of those
            minutes, from the end-of-minute coil nodes at the same output
            (plan amendment AJ: the size of the per-minute COP assumption).
    """

    residuals: list[float] = field(default_factory=list)
    uncapped_cops: list[float] = field(default_factory=list)
    sources_c: list[float] = field(default_factory=list)
    t_in_c: list[float] = field(default_factory=list)
    t_in_shift_k: list[float] = field(default_factory=list)


def simulate_window(rows: Sequence[TankRow], params: FourNodeParams, settings: FitSettings) -> WindowSimulation:
    """Run one draw-free window forward and compare with both probes.

    Args:
        rows: Consecutive minutes of one draw-free window.
        params: Model parameters.
        settings: Fixed values and variants.

    Returns:
        The simulation.

    Raises:
        ValueError: If a minute's heat input is invalid or a probe is
            missing (windows must be selected so that neither happens).
    """
    plant = params.plant(settings)
    cop = params.cop()
    law = params.law(settings)
    d = law.eps_1 + law.eps_2 - law.eps_1 * law.eps_2
    first_node, second_node = (A_LOW, B_LOW) if law.first == "A" else (B_LOW, A_LOW)
    state = initial_state(rows[0], settings)
    out = WindowSimulation()
    for row, nxt in zip(rows, rows[1:], strict=False):
        detail = heat_input_detail(row, settings.aux_kw, state.temps_c, cop, law)
        if not isinstance(detail, tuple):
            raise ValueError(f"invalid heat input at {row.ts.isoformat()}")
        heat, solved = detail
        state, _ = step_minute(plant, state, heat, 0.0)
        if solved is not None and row.brine_in_c is not None:
            out.uncapped_cops.append(solved.uncapped_cop)
            out.sources_c.append(row.brine_in_c)
            out.t_in_c.append(solved.t_in_c)
            t1, t2 = state.temps_c[first_node], state.temps_c[second_node]
            end = (solved.coil.q_kw / law.flow_kw_per_k + law.eps_1 * (1.0 - law.eps_2) * t1 + law.eps_2 * t2) / d
            out.t_in_shift_k.append(end - solved.t_in_c)
        if nxt.t_lower_c is None or nxt.t_top_c is None:
            raise ValueError(f"missing probe at {nxt.ts.isoformat()}")
        out.residuals.extend(
            (state.temps_c[settings.lower_node] - nxt.t_lower_c, state.temps_c[B_HIGH] - nxt.t_top_c)
        )
    return out


@dataclass(frozen=True)
class CapStats:
    """How often and how far the 1-8 COP cap binds (plan amendment AL).

    Attributes:
        minutes: Coil-active minutes.
        share_high: Share capped at 8.
        share_low: Share capped at 1.
        excess_high_mean: Mean of ``COP - 8`` over minutes capped at 8.
        excess_high_max: Maximum of the same.
        excess_low_mean: Mean of ``1 - COP`` over minutes capped at 1.
        excess_low_max: Maximum of the same.
    """

    minutes: int
    share_high: float
    share_low: float
    excess_high_mean: float
    excess_high_max: float
    excess_low_mean: float
    excess_low_max: float

    @property
    def strained(self) -> bool:
        """More than 10 % of coil-active minutes hit the cap."""
        return self.share_high + self.share_low > STRAINED_SHARE


def cap_stats(sims: Sequence[WindowSimulation]) -> CapStats:
    """Cap statistics over simulations.

    Args:
        sims: Simulations of a set of windows (training or held out).

    Returns:
        The statistics; zero shares without coil-active minutes.
    """
    cops = [c for sim in sims for c in sim.uncapped_cops]
    high = [c - COP_MAX for c in cops if c > COP_MAX]
    low = [COP_MIN - c for c in cops if c < COP_MIN]
    n = len(cops)
    return CapStats(
        minutes=n,
        share_high=len(high) / n if n else 0.0,
        share_low=len(low) / n if n else 0.0,
        excess_high_mean=float(np.mean(high)) if high else 0.0,
        excess_high_max=max(high, default=0.0),
        excess_low_mean=float(np.mean(low)) if low else 0.0,
        excess_low_max=max(low, default=0.0),
    )


@dataclass
class FitResult:
    """Outcome of a fit.

    Attributes:
        params: Fitted parameters.
        rmse_k: Root-mean-square probe error over all windows.
        jacobian: Jacobian of the probe residuals at the solution, columns
            for the free parameters only.
        free: Names of the free parameters, in Jacobian column order.
        cost: Half the sum of squared probe residuals.
        caps: How often the COP cap binds in the training windows.
        source_range_c: Observed source range in coil-active minutes.
        t_in_range_c: Start-of-minute supply temperature range.
        t_in_shift_k: Within-minute supply temperature changes.
    """

    params: FourNodeParams
    rmse_k: float
    jacobian: np.ndarray
    free: list[str]
    cost: float
    caps: CapStats
    source_range_c: tuple[float, float] | None
    t_in_range_c: tuple[float, float] | None
    t_in_shift_k: list[float]


def _probe_residuals(
    params: FourNodeParams, windows: Sequence[Sequence[TankRow]], settings: FitSettings
) -> tuple[np.ndarray, list[WindowSimulation]]:
    """All probe residuals and the simulations."""
    sims = [simulate_window(w, params, settings) for w in windows]
    return np.asarray([r for sim in sims for r in sim.residuals]), sims


def _fit_from(
    start: FourNodeParams,
    windows: Sequence[Sequence[TankRow]],
    settings: FitSettings,
    fixed: Mapping[str, float],
) -> FitResult:
    """One bounded least-squares run from one starting point."""
    free = [n for n in PARAMETER_NAMES if n not in fixed]
    lower = [BOUNDS[n][0] for n in free]
    upper = [BOUNDS[n][1] for n in free]
    x0 = [min(max(getattr(start, n), BOUNDS[n][0]), BOUNDS[n][1]) for n in free]

    def params_of(x: Sequence[float]) -> FourNodeParams:
        values = dict(fixed)
        values.update(zip(free, (float(v) for v in x), strict=True))
        return FourNodeParams(**values)

    if free:
        solution = least_squares(
            lambda x: _probe_residuals(params_of(x), windows, settings)[0],
            x0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
        )
        params = params_of(solution.x)
    else:
        params = params_of([])  # nothing to optimise: evaluate the fixed set
    probe, sims = _probe_residuals(params, windows, settings)
    sources = [s for sim in sims for s in sim.sources_c]
    t_ins = [t for sim in sims for t in sim.t_in_c]
    return FitResult(
        params=params,
        rmse_k=float(np.sqrt(np.mean(probe**2))) if probe.size else 0.0,
        jacobian=np.asarray(solution.jac) if free else np.zeros((probe.size, 0)),
        free=free,
        cost=float(0.5 * np.sum(probe**2)),
        caps=cap_stats(sims),
        source_range_c=(min(sources), max(sources)) if sources else None,
        t_in_range_c=(min(t_ins), max(t_ins)) if t_ins else None,
        t_in_shift_k=[s for sim in sims for s in sim.t_in_shift_k],
    )


def fit_four_node(
    windows: Sequence[Sequence[TankRow]],
    settings: FitSettings,
    starts: Sequence[FourNodeParams] = STARTS,
    fixed: Mapping[str, float] | None = None,
) -> FitResult:
    """Fit the parameters (or those not fixed) to the windows from several starts.

    Args:
        windows: Draw-free cooling and heating windows.
        settings: Fixed values and variants.
        starts: Starting points, each clipped into the bounds. Folds use the
            same fixed starts as the full-data fit (fold isolation, AM).
        fixed: Parameters held at a value (used by the degeneracy checks).

    Returns:
        The fit with the lowest cost.
    """
    fixed = dict(fixed or {})
    return min((_fit_from(start, windows, settings, fixed) for start in starts), key=lambda r: r.cost)


@dataclass
class Diagnosis:
    """Local Jacobian identification diagnostic of a fit (plan amendment AM).

    Attributes:
        names: The parameters judged (the free ones), in Jacobian order.
        full_rank: The scaled Jacobian has full rank.
        condition_number: Its condition number.
        weak: Parameters loading on near-singular directions, or with a
            zero column.
    """

    names: list[str]
    full_rank: bool
    condition_number: float
    weak: list[str]

    @property
    def identified(self) -> bool:
        """Full rank, a condition number below 1e6, and no weak parameter."""
        return self.full_rank and self.condition_number < _CONDITION_LIMIT and not self.weak


def identification(result: FitResult) -> Diagnosis:
    """Judge locally whether the fitted parameters are pinned down by the data.

    Args:
        result: The fit to judge.

    Returns:
        The diagnosis. It says nothing about how far a parameter could move
        (parameter uncertainty is not assessed, amendment AM).
    """
    names = list(result.free)
    jac = result.jacobian
    norms = np.linalg.norm(jac, axis=0)
    scaled = jac / np.where(norms > 0.0, norms, 1.0)
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    largest = singular[0] if singular.size and singular[0] > 0.0 else 1.0
    rank = int(np.sum(singular > largest * _WEAK_SINGULAR)) if np.all(norms > 0.0) else 0
    smallest = singular[-1] if singular.size else 0.0
    condition = float(largest / smallest) if smallest > 0.0 else float("inf")
    weak: set[str] = {names[i] for i, n in enumerate(norms) if n == 0.0}
    for value, direction in zip(singular, vt, strict=True):
        if value < largest * _WEAK_SINGULAR or condition >= _CONDITION_LIMIT and value == smallest:
            weak.update(names[i] for i, loading in enumerate(direction) if abs(loading) > _WEAK_LOADING)
    return Diagnosis(
        names=names,
        full_rank=rank == len(names),
        condition_number=condition,
        weak=[n for n in names if n in weak],
    )
