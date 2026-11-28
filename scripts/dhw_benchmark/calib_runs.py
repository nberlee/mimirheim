"""Heating runs: the appliance thermostat check and appliance limits (benchmark ticket 03).

A run is a maximal stretch of consecutive minutes with the compressor on in
one hot water mode (``dhw`` or ``legionella``). It is complete when the
minutes just before and after it are observed, one minute away, and are in
idle or space heating mode (compressor on or off), or in the run's own mode
with the compressor off (plan amendment AO: the appliance switches modes
without stopping the compressor); otherwise it is censored, with the
reason.

Thermostat check (plan amendments A, T, AB, AH and AN): the evaluation set
is complete dhw runs that lie entirely inside an accepted draw-free heating
window (``draw_free``). The check simulates with zero draws; a real draw
would lengthen the recorded run while the simulated rule fired too early,
so runs with possible draws are excluded rather than corrected. A run is
evaluated only if both probes exist at its first minute and every minute
has a valid heat input and (for band rules) a high target; otherwise no
rule evaluates it. Each run is simulated from its first minute (initial
state per amendment AA) with its recorded heat input, up to the recorded
stop and no further. A stop rule fires at the first minute its quantity
reaches the band's high target: the capacity-weighted mean of tank A
(``tank_a_mean``), A_high (``a_high``) or A_low (``a_low``). Legionella
runs use the fixed rule A_low reaching 65 C. Coverage is the share of
evaluated runs where the rule fires at or before the recorded stop; the
error is the recorded stop minus the firing minute. A rule that has not
fired reports its shortfall in kelvin at the stop.

Selection: among eligible rules (coverage at least 50 % and at least 10
evaluated runs), the highest coverage, then the lowest median absolute
error; rules within 2 minutes of the best error tie, and a tie goes to the
mean rule when it is one of them, otherwise to the lower error and then the
higher coverage. Without an eligible rule the mean rule is used, flagged
inconclusive. Each cross-validation fold selects on runs that, with both
neighbour minutes, touch neither its held-out week nor its held-out
windows, and that qualify as draw-free through a heating window touching
neither; it reports the selected rule on the held-out week's runs.

The start rule (the appliance starts when the tank A mean falls below the
low target) is a temperature-consistency diagnostic only: at each recorded
start of a complete run, the estimated tank A mean minus ``band_low_c``.
The start timing comparison is deferred (amendment AN).

Appliance limits (plan item 10, amendments F and AP), from every complete
dhw run, labelled "under commanded targets" (the appliance has no default
band; the user's automation sets the targets, so a commanded change can
shorten a run): the shortest run, the steady compressor power (per run of
at least 8 minutes the median from minute 5 to two minutes before the end;
the lowest over the runs), and the configured minimum off time of 5
minutes (user).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Literal

from dhw_benchmark.calib_fit import FitSettings, FourNodeParams, initial_state
from dhw_benchmark.calib_heat import A_HIGH, A_LOW, compressor_electric_kw, heat_input, heat_input_valid
from dhw_benchmark.inputs import TankRow
from dhw_benchmark.plant import HeatInput, step_minute

StopRule = Literal["tank_a_mean", "a_high", "a_low"]
StopKind = Literal["switched", "unknown target", "commanded", "natural"]
STOP_RULES: tuple[StopRule, ...] = ("tank_a_mean", "a_high", "a_low")
Week = tuple[int, int]

LEGIONELLA_STOP_C = 65.0
MIN_OFF_MINUTES = 5
"""Configured minimum off time between runs (user), in minutes."""

_MINUTE = timedelta(minutes=1)
_NORMAL_NEIGHBOUR_MODES = ("idle", "space_heating")
_MIN_RUNS = 10
_MIN_COVERAGE = 0.5
_TIE_MIN = 2.0
_STEADY_FROM = 5
_STEADY_BEFORE_END = 2
_STEADY_MIN_MINUTES = 8


@dataclass(frozen=True)
class Run:
    """One heating run.

    Attributes:
        rows: The run's minutes (compressor on, one hot water mode).
        mode: ``dhw`` or ``legionella``.
        complete: Bordered by normal observed minutes on both sides.
        reason: Why it is censored, or None.
        before: The minute before the run, if any.
        after: The minute after the run, if any.
    """

    rows: list[TankRow]
    mode: str
    complete: bool
    reason: str | None
    before: TankRow | None = None
    after: TankRow | None = None

    @property
    def natural_start(self) -> bool:
        """The run starts from idle or its own mode with the compressor off (AQ)."""
        return self.complete and self.before is not None and self.before.mode != "space_heating"

    @property
    def stop_target_known(self) -> bool:
        """The high target is known at the run's last minute and at the stop minute."""
        return self.after is not None and self.after.band_high_c is not None and self.rows[-1].band_high_c is not None

    @property
    def commanded_stop(self) -> bool:
        """The high target changed at the stop minute: the automation ended the run (AR)."""
        return self.stop_target_known and self.after is not None and self.after.band_high_c != self.rows[-1].band_high_c

    @property
    def stop_kind(self) -> StopKind | None:
        """The exclusive stop kind of a complete run, or None for a censored run (AR).

        Precedence: switched (space heating after), unknown target, commanded,
        natural.
        """
        if not self.complete or self.after is None:
            return None
        if self.after.mode == "space_heating":
            return "switched"
        if not self.stop_target_known:
            return "unknown target"
        return "commanded" if self.commanded_stop else "natural"

    @property
    def natural_stop(self) -> bool:
        """The run stops into idle or its own mode with the compressor off, at a known and
        unchanged high target (AQ, AR)."""
        return self.stop_kind == "natural"

    @property
    def week(self) -> Week:
        """ISO week of the run's first minute."""
        return _iso(self.rows[0].ts)


def _iso(ts: datetime) -> Week:
    """ISO year and week of a timestamp."""
    year, week, _ = ts.isocalendar()
    return (year, week)


def _neighbour_reason(row: TankRow | None, expected: datetime, side: str, mode: str) -> str | None:
    """Why a neighbouring minute makes a run censored, or None when it is normal (amendment AO).

    Idle or space heating is normal with the compressor on or off (the
    appliance switches modes without stopping the compressor and runs it
    down after a run); the run's own mode is normal with the compressor off.
    """
    if row is None or row.ts != expected:
        return f"gap {side}"
    if row.mode in _NORMAL_NEIGHBOUR_MODES or (row.mode == mode and row.compressor_on is False):
        return None
    return f"{(row.mode or 'unknown').replace('_', ' ')} {side}"


def find_runs(rows: Sequence[TankRow]) -> list[Run]:
    """Split minute rows into heating runs.

    Args:
        rows: Minute rows in time order.

    Returns:
        Every run, complete or censored.
    """
    runs: list[Run] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if not (row.compressor_on and row.mode in ("dhw", "legionella")):
            i += 1
            continue
        j = i + 1
        while (
            j < len(rows)
            and rows[j].compressor_on
            and rows[j].mode == row.mode
            and rows[j].ts == rows[j - 1].ts + _MINUTE
        ):
            j += 1
        before = rows[i - 1] if i > 0 else None
        after = rows[j] if j < len(rows) else None
        mode = row.mode or ""
        reason = _neighbour_reason(before, row.ts - _MINUTE, "before", mode) or _neighbour_reason(
            after, rows[j - 1].ts + _MINUTE, "after", mode
        )
        runs.append(
            Run(
                rows=list(rows[i:j]),
                mode=row.mode or "",
                complete=reason is None,
                reason=reason,
                before=before,
                after=after,
            )
        )
        i = j
    return runs


def draw_free(runs: Sequence[Run], heating_windows: Sequence[Sequence[TankRow]]) -> list[Run]:
    """Complete runs that lie entirely inside one accepted draw-free heating window.

    Args:
        runs: Runs.
        heating_windows: Accepted heating windows (``calib_windows``).

    Returns:
        The runs usable for the thermostat check (amendment AN).
    """
    spans = [{r.ts for r in w} for w in heating_windows]
    return [run for run in runs if run.complete and any(all(r.ts in span for r in run.rows) for span in spans)]


def tank_a_mean(temps_c: Sequence[float], settings: FitSettings) -> float:
    """Capacity-weighted mean temperature of tank A.

    Args:
        temps_c: Node temperatures (A_low, A_high, B_low, B_high).
        settings: The node split.

    Returns:
        The mean in C.
    """
    return settings.split_a * temps_c[A_LOW] + (1.0 - settings.split_a) * temps_c[A_HIGH]


def _quantity(rule: StopRule, temps_c: Sequence[float], settings: FitSettings) -> float:
    """The temperature a stop rule watches."""
    if rule == "tank_a_mean":
        return tank_a_mean(temps_c, settings)
    return temps_c[A_HIGH] if rule == "a_high" else temps_c[A_LOW]


@dataclass(frozen=True)
class RunCheck:
    """A stop rule checked against one run.

    Attributes:
        start: The run's first minute.
        fired_at: Minutes from the start at which the rule fired, or None.
        error_min: Recorded stop minus the firing minute, or None.
        shortfall_k: Threshold minus the watched temperature at the
            recorded stop when the rule has not fired, or None.
    """

    start: datetime
    fired_at: int | None
    error_min: int | None
    shortfall_k: float | None


def check_run(
    run: Run,
    rule: StopRule,
    params: FourNodeParams,
    settings: FitSettings,
    threshold_c: float | None = None,
) -> RunCheck | None:
    """Simulate a run with its recorded heat input and check a stop rule.

    Args:
        run: A complete run.
        rule: The stop rule.
        params: Calibrated parameters.
        settings: Fixed values (auxiliary power included).
        threshold_c: Fixed threshold (legionella: 65 C); otherwise each
            minute's ``band_high_c``.

    Returns:
        The check, or None when the run is not evaluable (a missing first
        probe, an invalid heat input or a missing threshold in any minute);
        this is decided before any rule is evaluated.
    """
    first = run.rows[0]
    if first.t_lower_c is None or first.t_top_c is None:
        return None
    if any(not heat_input_valid(r, settings.aux_kw) for r in run.rows):
        return None
    if threshold_c is None and any(r.band_high_c is None for r in run.rows):
        return None
    plant = params.plant(settings)
    cop, law = params.cop(), params.law(settings)
    state = initial_state(first, settings)
    n = len(run.rows)
    for k, row in enumerate(run.rows, start=1):
        heat = heat_input(row, settings.aux_kw, state.temps_c, cop, law)
        assert isinstance(heat, HeatInput)  # validated above
        state, _ = step_minute(plant, state, heat, 0.0)
        threshold = threshold_c if threshold_c is not None else row.band_high_c
        assert threshold is not None  # validated above
        value = _quantity(rule, state.temps_c, settings)
        if value >= threshold:
            return RunCheck(start=run.rows[0].ts, fired_at=k, error_min=n - k, shortfall_k=None)
        if k == n:
            return RunCheck(start=run.rows[0].ts, fired_at=None, error_min=None, shortfall_k=threshold - value)
    return None


@dataclass(frozen=True)
class RuleStats:
    """One stop rule over a set of runs.

    Attributes:
        rule: The rule.
        evaluated: Runs evaluated.
        fired: Runs where it fired by the recorded stop.
        median_abs_error_min: Median absolute error over those, or None.
    """

    rule: StopRule
    evaluated: int
    fired: int
    median_abs_error_min: float | None

    @property
    def coverage(self) -> float:
        """Share of evaluated runs where the rule fired; 0 without runs."""
        return self.fired / self.evaluated if self.evaluated else 0.0


def rule_stats(checks: Sequence[RunCheck | None], rule: StopRule) -> RuleStats:
    """Coverage and median error of one rule.

    Args:
        checks: Per-run checks; None entries were not evaluated.
        rule: The rule's name.

    Returns:
        The statistics.
    """
    done = [c for c in checks if c is not None]
    errors = [abs(c.error_min) for c in done if c.error_min is not None]
    return RuleStats(
        rule=rule,
        evaluated=len(done),
        fired=len(errors),
        median_abs_error_min=float(median(errors)) if errors else None,
    )


def select_rule(
    stats: Mapping[str, RuleStats], min_runs: int = _MIN_RUNS, min_coverage: float = _MIN_COVERAGE
) -> tuple[StopRule, bool]:
    """Select the stop rule (plan amendments AB and AH).

    Args:
        stats: Statistics per rule.
        min_runs: Evaluated runs a rule needs.
        min_coverage: Coverage a rule needs.

    Returns:
        ``(rule, inconclusive)``; the mean rule, flagged, without an
        eligible rule.
    """
    eligible = [
        s for s in stats.values()
        if s.evaluated >= min_runs and s.coverage >= min_coverage and s.median_abs_error_min is not None
    ]
    if not eligible:
        return "tank_a_mean", True
    top = max(s.coverage for s in eligible)
    leaders = [s for s in eligible if s.coverage == top]
    best = min(s.median_abs_error_min for s in leaders if s.median_abs_error_min is not None)
    tied = [s for s in leaders if s.median_abs_error_min is not None and s.median_abs_error_min - best <= _TIE_MIN]
    if any(s.rule == "tank_a_mean" for s in tied):
        return "tank_a_mean", False
    chosen = min(tied, key=lambda s: (s.median_abs_error_min, -s.coverage))
    return chosen.rule, False


@dataclass(frozen=True)
class FoldThermostat:
    """The stop rule of one cross-validation fold.

    Attributes:
        week: The held-out week.
        selected: Rule selected on the training weeks.
        inconclusive: No rule was eligible.
        evaluated_training: Training runs evaluated (selected rule).
        held_out_evaluated: Held-out runs evaluated.
        held_out_coverage: Selected rule's coverage on the held-out runs.
        held_out_median_abs_error_min: Its median absolute error there.
        held_out_shortfalls_k: Shortfalls of the held-out runs where it did
            not fire by the recorded stop (censored checks, amendments A, T).
    """

    week: Week
    selected: StopRule
    inconclusive: bool
    evaluated_training: int
    held_out_evaluated: int
    held_out_coverage: float
    held_out_median_abs_error_min: float | None
    held_out_shortfalls_k: list[float]


def thermostat_folds(
    runs: Sequence[Run],
    heating_windows: Sequence[Sequence[TankRow]],
    fold_models: Mapping[Week, tuple[FourNodeParams, FitSettings]],
    held_minutes: Mapping[Week, frozenset[datetime]] | None = None,
    min_runs: int = _MIN_RUNS,
) -> list[FoldThermostat]:
    """Select the stop rule per fold on training runs and score it on held-out runs.

    Args:
        runs: All runs; complete dhw runs with a natural stop inside an
            accepted heating window are used (AQ).
        heating_windows: Accepted draw-free heating windows.
        fold_models: Per held-out week, the fold's parameters and settings
            (with the fold's auxiliary power).
        held_minutes: Per held-out week, the minutes of the calibration
            fold's held-out windows (spillover into the next week included).
        min_runs: Evaluated runs a rule needs.

    Returns:
        One result per fold, in week order.
    """
    dhw = [r for r in runs if r.complete and r.mode == "dhw"]
    out = []
    for week in sorted(fold_models):
        params, settings = fold_models[week]
        held = (held_minutes or {}).get(week, frozenset())

        def touches(rows: Sequence[TankRow | None], week: Week = week, held: frozenset[datetime] = held) -> bool:
            return any(x is not None and (_iso(x.ts) == week or x.ts in held) for x in rows)

        clean_windows = [w for w in heating_windows if not touches(w)]
        train = [
            r for r in draw_free(dhw, clean_windows) if r.natural_stop and not touches([r.before, *r.rows, r.after])
        ]
        test = [r for r in draw_free(dhw, heating_windows) if r.week == week and r.natural_stop]
        stats = {rule: rule_stats([check_run(r, rule, params, settings) for r in train], rule) for rule in STOP_RULES}
        selected, inconclusive = select_rule(stats, min_runs=min_runs)
        held_checks = [check_run(r, selected, params, settings) for r in test]
        held = rule_stats(held_checks, selected)
        out.append(
            FoldThermostat(
                week=week,
                selected=selected,
                inconclusive=inconclusive,
                evaluated_training=stats[selected].evaluated,
                held_out_evaluated=held.evaluated,
                held_out_coverage=held.coverage,
                held_out_median_abs_error_min=held.median_abs_error_min,
                held_out_shortfalls_k=sorted(
                    c.shortfall_k for c in held_checks if c is not None and c.shortfall_k is not None
                ),
            )
        )
    return out


def start_margins(runs: Sequence[Run], settings: FitSettings) -> list[float]:
    """Estimated tank A mean minus ``band_low_c`` at each recorded start.

    Args:
        runs: Runs; complete dhw runs are used.
        settings: Probe placement and node split.

    Returns:
        One margin per run with a known low target, in K (negative: below
        the target, as the start rule expects).
    """
    margins = []
    for run in runs:
        first = run.rows[0]
        if not (run.complete and run.mode == "dhw"):
            continue
        if first.band_low_c is None or first.t_lower_c is None or first.t_top_c is None:
            continue
        state = initial_state(first, settings)
        margins.append(tank_a_mean(state.temps_c, settings) - first.band_low_c)
    return margins


@dataclass(frozen=True)
class OffGap:
    """The observed time between two consecutive dhw runs.

    Attributes:
        minutes: Minutes from the end of one run to the start of the next.
        outside_dhw_minutes: Of those, minutes in another mode than dhw or
            idle (space heating, legionella).
    """

    minutes: int
    outside_dhw_minutes: int


def off_gaps(rows: Sequence[TankRow], runs: Sequence[Run]) -> list[OffGap]:
    """Observed gaps between consecutive dhw runs, a cross-check of the minimum off time.

    Args:
        rows: Minute rows in time order.
        runs: Runs found in those rows.

    Returns:
        One gap per pair of consecutive complete dhw runs whose
        minutes in between are all observed with a known compressor state.
    """
    index = {r.ts: k for k, r in enumerate(rows)}
    dhw = [r for r in runs if r.mode == "dhw"]
    gaps = []
    for prev, nxt in zip(dhw, dhw[1:], strict=False):
        a, b = index[prev.rows[-1].ts], index[nxt.rows[0].ts]
        between = rows[a + 1 : b]
        expected = int((nxt.rows[0].ts - prev.rows[-1].ts) / _MINUTE) - 1
        if len(between) != expected or any(r.compressor_on is None for r in between):
            continue  # a gap in the data or an unknown compressor state: not an observed shutdown
        if not (prev.complete and nxt.complete):
            continue
        gaps.append(
            OffGap(
                minutes=expected,
                outside_dhw_minutes=sum(1 for r in between if r.mode not in ("dhw", "idle")),
            )
        )
    return gaps


@dataclass(frozen=True)
class ApplianceLimits:
    """Appliance limits estimated from complete dhw runs (estimates under commanded targets).

    Attributes:
        shortest_complete_run_min: Shortest complete run with a natural start
            and stop (AQ), or None.
        steady_power_kw: Lowest per-run steady compressor power, or None.
        steady_runs: Runs long enough for a steady power.
        min_off_minutes: Configured minimum off time (user).
    """

    shortest_complete_run_min: int | None
    steady_power_kw: float | None
    steady_runs: int
    min_off_minutes: int = MIN_OFF_MINUTES


def appliance_limits(runs: Sequence[Run], aux_kw: float) -> ApplianceLimits:
    """Appliance limits from complete dhw runs.

    Args:
        runs: Runs; only complete dhw runs count.
        aux_kw: Auxiliary power subtracted from the electric power.

    Returns:
        The limits.
    """
    complete = [r for r in runs if r.complete and r.mode == "dhw"]
    steady = []
    for run in complete:
        window = run.rows[_STEADY_FROM : len(run.rows) - _STEADY_BEFORE_END]
        powers = [p for p in (compressor_electric_kw(r, aux_kw) for r in window) if isinstance(p, float)]
        if len(run.rows) >= _STEADY_MIN_MINUTES and window and len(powers) == len(window):
            steady.append(median(powers))
    natural = [r for r in complete if r.natural_start and r.natural_stop]
    return ApplianceLimits(
        shortest_complete_run_min=min((len(r.rows) for r in natural), default=None),
        steady_power_kw=min(steady) if steady else None,
        steady_runs=len(steady),
    )
