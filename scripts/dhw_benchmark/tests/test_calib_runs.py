"""Unit tests for scripts/dhw_benchmark/calib_runs.py: runs, the thermostat check and appliance limits."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from dhw_benchmark.calib_fit import FitSettings, FourNodeParams, initial_state
from dhw_benchmark.calib_heat import heat_input
from dhw_benchmark.calib_runs import (
    RuleStats,
    appliance_limits,
    check_run,
    find_runs,
    rule_stats,
    select_rule,
    start_margins,
    tank_a_mean,
    thermostat_folds,
)
from dhw_benchmark.inputs import TankRow
from dhw_benchmark.plant import step_minute

MONDAY = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
M = timedelta(minutes=1)
TRUE = FourNodeParams(
    eps_1=0.5, eps_2=0.6, flow_kw_per_k=1.2, k_a=0.01, k_b=0.02, loss_kw_per_k=0.005, cop_ref=4.2, c1=0.08, c2=-0.12
)
SETTINGS = FitSettings(room_temp_c=18.0, cold_temp_c=12.0, aux_kw=0.05, coil_order="A", lower_node=0)


def _row(ts: datetime, **kw: object) -> TankRow:
    data: dict = dict(
        ts=ts, t_top_c=45.0, t_lower_c=30.0, hp_electric_kw=0.05, mode="idle", compressor_on=False,
        element_kw=0.0, brine_in_c=None, band_low_c=36.0, band_high_c=46.0,
    )
    data.update(kw)
    return TankRow(**data)


def _thermostat_run(start: datetime, lo: float, top: float, band_high: float, rule: str = "tank_a_mean",
                    pre: int = 5, post: int = 5, power: float = 1.65,
                    recorded_band: float | None = None) -> list[TankRow]:
    """Idle minutes, a dhw run the plant stops by the given rule at band_high, idle minutes.

    The probes are read from the simulated state, so the check reproduces the stop exactly.
    """
    plant = TRUE.plant(SETTINGS)
    rows: list[TankRow] = []
    state = initial_state(_row(start, t_top_c=top, t_lower_c=lo), SETTINGS)
    t = start
    heating = False
    done = False
    i = 0
    while True:
        if i == pre:
            heating = True
            # The check starts from the interpolated probes: make the truth match at the run start.
            state = initial_state(_row(t, t_top_c=state.temps_c[3], t_lower_c=state.temps_c[0]), SETTINGS)
        if heating and not done:
            value = {"tank_a_mean": tank_a_mean(state.temps_c, SETTINGS), "a_high": state.temps_c[1],
                     "a_low": state.temps_c[0]}[rule]
            if value >= band_high and i > pre:
                heating, done = False, True
                stop = i
        if done and i >= stop + post:
            break
        on = heating and not done
        row = _row(t, t_top_c=state.temps_c[3], t_lower_c=state.temps_c[0], hp_electric_kw=power if on else 0.05,
                   mode="dhw" if on else "idle", compressor_on=on, brine_in_c=10.0 if on else None,
                   band_high_c=band_high if recorded_band is None else recorded_band)
        rows.append(row)
        heat = heat_input(row, SETTINGS.aux_kw, state.temps_c, TRUE.cop(), TRUE.law(SETTINGS))
        state, _ = step_minute(plant, state, heat, draw_kwh=0.0)
        t += M
        i += 1
    return rows


def test_runs_are_complete_between_normal_mode_switches() -> None:
    """AO: idle or space heating on either side is normal, compressor on or off (the
    appliance switches modes and runs the compressor down); the run's own mode with the
    compressor off is normal; legionella beside a dhw run, a gap or the data edge censor."""
    def on(i: int, mode: str = "dhw") -> TankRow:
        return _row(MONDAY + i * M, mode=mode, compressor_on=True, hp_electric_kw=1.6, brine_in_c=10.0)
    rows = [_row(MONDAY + 0 * M, mode="space_heating", compressor_on=True)]
    rows += [on(i) for i in range(1, 6)]
    rows += [_row(MONDAY + 6 * M, compressor_on=True)]                       # idle, compressor running down
    rows += [_row(MONDAY + 7 * M, mode="dhw", compressor_on=False)]          # dhw mode before the compressor
    rows += [on(i) for i in range(8, 12)]
    rows += [_row(MONDAY + 12 * M, mode="space_heating", compressor_on=True)]
    rows += [on(i) for i in range(13, 16)]
    rows += [on(i, "legionella") for i in range(16, 18)]
    rows += [on(i) for i in range(30, 33)]                                   # after a data gap
    runs = find_runs(rows)
    assert [(r.mode, len(r.rows), r.complete) for r in runs] == [
        ("dhw", 5, True), ("dhw", 4, True), ("dhw", 3, False), ("legionella", 2, False), ("dhw", 3, False),
    ]
    assert runs[2].reason == "legionella after" and runs[3].reason == "dhw before"
    assert runs[4].reason == "gap before"


def test_the_matching_stop_rule_fires_exactly_at_the_recorded_stop() -> None:
    """A run stopped by the tank-A-mean rule: that rule fires at the stop with zero error."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    check = check_run(run, "tank_a_mean", TRUE, SETTINGS)
    assert check is not None and check.fired_at == len(run.rows) and check.error_min == 0


def test_a_rule_that_has_not_fired_by_the_stop_reports_the_shortfall() -> None:
    """No firing after the recorded stop and no heat invented (A, AH): the shortfall is reported."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    check = check_run(run, "tank_a_mean", TRUE, SETTINGS, threshold_c=60.0)
    assert check is not None and check.fired_at is None and check.error_min is None
    assert check.shortfall_k is not None and 10.0 < check.shortfall_k < 15.0


def _stats(rule: str, evaluated: int, fired: int, median: float) -> RuleStats:
    return RuleStats(rule=rule, evaluated=evaluated, fired=fired, median_abs_error_min=median)


def test_rule_selection_prefers_coverage_then_error_with_a_tie_to_the_mean() -> None:
    """Highest coverage, then lowest median error; within 2 minutes the mean rule wins only
    when it is among the tied eligible rules (AB, AH)."""
    stats = {"tank_a_mean": _stats("tank_a_mean", 20, 18, 3.0), "a_high": _stats("a_high", 20, 18, 1.5),
             "a_low": _stats("a_low", 20, 10, 0.0)}
    assert select_rule(stats) == ("tank_a_mean", False)
    stats["a_high"] = _stats("a_high", 20, 18, 0.5)
    assert select_rule(stats) == ("a_high", False)
    stats["a_high"] = _stats("a_high", 20, 19, 3.0)
    assert select_rule(stats) == ("a_high", False)
    no_mean = {"tank_a_mean": _stats("tank_a_mean", 20, 5, 1.0), "a_high": _stats("a_high", 20, 18, 2.0),
               "a_low": _stats("a_low", 20, 18, 1.0)}
    assert select_rule(no_mean) == ("a_low", False)


def test_without_enough_evidence_the_mean_rule_is_used_and_flagged() -> None:
    """Coverage below 50 % or fewer than 10 runs: inconclusive, mean rule, flagged."""
    few = {r: _stats(r, 9, 9, 1.0) for r in ("tank_a_mean", "a_high", "a_low")}
    assert select_rule(few) == ("tank_a_mean", True)
    low = {r: _stats(r, 20, 9, 1.0) for r in ("tank_a_mean", "a_high", "a_low")}
    assert select_rule(low) == ("tank_a_mean", True)


def test_rule_statistics_count_coverage_and_median_error() -> None:
    """Coverage over evaluated runs; median absolute error over the runs where it fired."""
    rows = []
    for k in range(3):
        rows += _thermostat_run(MONDAY + timedelta(hours=3 * k), 30.0 + k, 44.0, 46.0)
    runs = [r for r in find_runs(rows) if r.complete]
    stats = rule_stats([check_run(r, "tank_a_mean", TRUE, SETTINGS) for r in runs], "tank_a_mean")
    assert (stats.evaluated, stats.fired, stats.median_abs_error_min) == (3, 3, 0.0)
    assert stats.coverage == 1.0


def test_the_stop_rule_is_selected_on_training_weeks_only() -> None:
    """Each fold selects from runs outside its held-out week and scores the held-out runs."""
    blocks = []
    for w in range(3):
        for k in range(4):
            # Week 0 records a high target the plant never reaches: no rule fires there.
            blocks.append(_thermostat_run(MONDAY + timedelta(weeks=w, hours=3 * k), 30.0 + k, 44.0, 46.0,
                                          recorded_band=60.0 if w == 0 else None))
    runs = [r for r in find_runs([r for b in blocks for r in b]) if r.complete]
    weeks = [(2026, 45), (2026, 46), (2026, 47)]
    folds = thermostat_folds(runs, blocks, {w: (TRUE, SETTINGS) for w in weeks}, min_runs=4)
    assert [f.held_out_coverage for f in folds] == [0.0, 1.0, 1.0]
    assert all(f.evaluated_training == 8 and f.held_out_evaluated == 4 for f in folds)
    assert all(f.selected == "tank_a_mean" and not f.inconclusive for f in folds)
    assert [f.held_out_median_abs_error_min for f in folds] == [None, 0.0, 0.0]
    assert len(folds[0].held_out_shortfalls_k) == 4 and all(v > 0.0 for v in folds[0].held_out_shortfalls_k)
    assert folds[1].held_out_shortfalls_k == []


def test_start_margins_compare_the_tank_a_mean_with_the_low_band() -> None:
    """At each recorded start: estimated tank A mean minus band_low."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    margins = start_margins([run], SETTINGS)
    state = initial_state(run.rows[0], SETTINGS)
    assert margins == [pytest.approx(tank_a_mean(state.temps_c, SETTINGS) - 36.0)]


def test_appliance_limits_from_complete_runs() -> None:
    """Steady power: per run of at least 8 minutes the median compressor power from minute 5
    to two before the end, the lowest reported (F); every complete run counts, whatever the
    commanded targets (AP)."""
    def run(start: int, n: int, power: float) -> list[TankRow]:
        out = [_row(MONDAY + (start - 1) * M)]
        out += [_row(MONDAY + (start + i) * M, mode="dhw", compressor_on=True, hp_electric_kw=power + 0.01 * i,
                     brine_in_c=10.0) for i in range(n)]
        out.append(_row(MONDAY + (start + n) * M))
        return out
    rows = run(1, 20, 1.5) + run(40, 30, 1.3) + run(100, 9, 1.0) + run(200, 7, 0.9)
    off_band = [r.model_copy(update={"band_high_c": 50.0}) for r in run(300, 12, 0.5)]
    limits = appliance_limits(find_runs(rows + off_band), aux_kw=0.05)
    assert limits.steady_runs == 4
    assert limits.steady_power_kw == pytest.approx(0.5 + 0.01 * 7 - 0.05)  # targets do not filter (AP)
    assert limits.shortest_complete_run_min == 7
    assert limits.min_off_minutes == 5


def test_only_runs_inside_a_draw_free_heating_window_are_checked() -> None:
    """A run outside every accepted heating window may hide a draw: excluded (AN)."""
    from dhw_benchmark.calib_runs import draw_free

    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0) + _thermostat_run(MONDAY + timedelta(hours=3), 31.0, 44.0, 46.0)
    runs = [r for r in find_runs(rows) if r.complete]
    first_window = [r for r in rows if r.ts < MONDAY + timedelta(hours=3)]
    assert draw_free(runs, [first_window]) == [runs[0]]


def test_a_run_is_validated_before_any_rule_is_evaluated() -> None:
    """A missing first probe or a late missing threshold excludes the run for every rule,
    even one that would have fired before the damaged minute (AN)."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    no_probe = replace(run, rows=[run.rows[0].model_copy(update={"t_lower_c": None}), *run.rows[1:]])
    assert check_run(no_probe, "tank_a_mean", TRUE, SETTINGS) is None
    late = replace(run, rows=[*run.rows[:-1], run.rows[-1].model_copy(update={"band_high_c": None})])
    early = check_run(late, "tank_a_mean", TRUE, SETTINGS, threshold_c=35.0)
    assert early is not None and early.fired_at is not None  # a fixed threshold needs no band
    assert check_run(late, "tank_a_mean", TRUE, SETTINGS) is None
    bad_heat = replace(run, rows=[*run.rows[:-1], run.rows[-1].model_copy(update={"brine_in_c": None})])
    assert check_run(bad_heat, "tank_a_mean", TRUE, SETTINGS, threshold_c=35.0) is None


def _ending_at(end: datetime) -> list[TankRow]:
    """A thermostat block shifted so that its run's last compressor minute is ``end``."""
    block = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    last = max(r.ts for r in block if r.compressor_on)
    return [r.model_copy(update={"ts": r.ts + (end - last)}) for r in block]


def _two_weeks() -> list[list[TankRow]]:
    return [_thermostat_run(MONDAY + timedelta(weeks=w, hours=3 * k), 30.0 + k, 44.0, 46.0)
            for w in range(2) for k in range(3)]


def test_a_run_touching_the_held_out_week_or_windows_is_not_training_data() -> None:
    """A Sunday-night run into Monday, and a run inside a held-out window's minutes, are
    excluded from that fold's selection (AN)."""
    blocks = _two_weeks()
    straddle = _thermostat_run(MONDAY + timedelta(days=6, hours=23, minutes=50), 30.0, 44.0, 46.0)
    runs = [r for r in find_runs([r for b in [*blocks, straddle] for r in b]) if r.complete]
    weeks = [(2026, 45), (2026, 46)]
    models = {w: (TRUE, SETTINGS) for w in weeks}
    by_week = {f.week: f for f in thermostat_folds(runs, [*blocks, straddle], models, min_runs=1)}
    assert by_week[(2026, 46)].evaluated_training == 3  # the straddling run touches week 46
    assert by_week[(2026, 45)].evaluated_training == 3  # the three week-46 runs
    week46_run = next(r for r in runs if r.week == (2026, 46))
    held = {(2026, 45): frozenset(r.ts for r in week46_run.rows), (2026, 46): frozenset()}
    folds = thermostat_folds(runs, [*blocks, straddle], models, held_minutes=held, min_runs=1)
    assert {f.week: f for f in folds}[(2026, 45)].evaluated_training == 2


def test_a_run_whose_stop_minute_is_held_out_is_not_training_data() -> None:
    """The run ends Sunday 23:59; its observed stop is Monday 00:00 (AN)."""
    blocks = _two_weeks()
    sunday = _ending_at(MONDAY + timedelta(days=6, hours=23, minutes=59))
    window = [r for r in sunday if r.ts < MONDAY + timedelta(weeks=1)]  # only the neighbour reaches Monday
    runs = [r for r in find_runs([r for b in [*blocks, sunday] for r in b]) if r.complete]
    folds = thermostat_folds(runs, [*blocks, window], {(2026, 46): (TRUE, SETTINGS)}, min_runs=1)
    assert folds[0].evaluated_training == 3


def test_a_run_qualifying_only_through_a_held_out_window_is_not_training_data() -> None:
    """The run and its neighbours lie in Sunday, but its heating window's padding reaches
    into Monday: held-out probes would decide eligibility (AN)."""
    blocks = _two_weeks()
    sunday = _ending_at(MONDAY + timedelta(days=6, hours=23, minutes=57))
    run = next(r for r in find_runs(sunday) if r.complete)
    assert run.after is not None and run.after.ts.isocalendar()[1] == 45
    runs = [r for r in find_runs([r for b in [*blocks, sunday] for r in b]) if r.complete]
    folds = thermostat_folds(runs, [*blocks, sunday], {(2026, 46): (TRUE, SETTINGS)}, min_runs=1)
    assert folds[0].evaluated_training == 3


def test_observed_off_gaps_cross_check_the_minimum_off_time() -> None:
    """Gaps between consecutive dhw runs, with minutes outside dhw mode; unobserved gaps skipped."""
    from dhw_benchmark.calib_runs import off_gaps

    def on(i: int) -> TankRow:
        return _row(MONDAY + i * M, mode="dhw", compressor_on=True, hp_electric_kw=1.6, brine_in_c=10.0)
    rows = [_row(MONDAY)] + [on(i) for i in range(1, 5)] + [_row(MONDAY + 5 * M)]
    rows += [_row(MONDAY + 6 * M, mode="space_heating", compressor_on=True)] + [_row(MONDAY + 7 * M)]
    rows += [on(i) for i in range(8, 12)] + [_row(MONDAY + 12 * M)]
    rows += [on(i) for i in range(20, 24)] + [_row(MONDAY + 24 * M)]  # after a data gap
    gaps = off_gaps(rows, find_runs(rows))
    assert [(g.minutes, g.outside_dhw_minutes) for g in gaps] == [(3, 1)]
    unknown = [r.model_copy(update={"compressor_on": None}) if r.ts == MONDAY + 7 * M else r for r in rows]
    assert off_gaps(unknown, find_runs(unknown)) == []  # no observed shutdown


def test_runs_cut_short_by_space_heating_do_not_select_the_stop_rule() -> None:
    """AQ: a switched stop may come before the thermostat would have stopped; such runs are
    left out of rule selection and of the shortest-run estimate."""
    blocks = [_thermostat_run(MONDAY + timedelta(weeks=w, hours=3 * k), 30.0 + k, 44.0, 46.0)
              for w in range(2) for k in range(3)]
    # Cut one week-46 run short: switch to space heating after 4 compressor minutes.
    cut = blocks[3]
    first_on = next(i for i, r in enumerate(cut) if r.compressor_on)
    cut = cut[: first_on + 4] + [cut[first_on + 4].model_copy(update={"mode": "space_heating", "compressor_on": True})]
    blocks[3] = cut
    runs = find_runs([r for b in blocks for r in b])
    switched = [r for r in runs if r.complete and not r.natural_stop]
    assert len(switched) == 1 and switched[0].natural_start
    folds = thermostat_folds(runs, blocks, {(2026, 45): (TRUE, SETTINGS), (2026, 46): (TRUE, SETTINGS)}, min_runs=1)
    by_week = {f.week: f for f in folds}
    assert by_week[(2026, 45)].evaluated_training == 2  # the cut week-46 run is not evidence
    assert by_week[(2026, 46)].held_out_evaluated == 2
    limits = appliance_limits(runs, aux_kw=0.05)
    assert limits.shortest_complete_run_min is not None and limits.shortest_complete_run_min > 4


def test_a_target_change_at_the_stop_is_a_commanded_stop() -> None:
    """AR: the automation lowers the high target at the stop minute; the run is not
    natural evidence for the thermostat."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    assert run.natural_stop and not run.commanded_stop
    stop = run.after
    assert stop is not None
    changed = [r.model_copy(update={"band_high_c": 40.0}) if r.ts == stop.ts else r for r in rows]
    run = next(r for r in find_runs(changed) if r.complete)
    assert run.commanded_stop and not run.natural_stop



def test_an_unknown_stop_target_is_neither_natural_nor_commanded() -> None:
    """A missing target at the last minute or the stop minute proves nothing either way."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    stop = run.after
    assert stop is not None
    for ts in (stop.ts, run.rows[-1].ts):
        missing = [r.model_copy(update={"band_high_c": None}) if r.ts == ts else r for r in rows]
        run_m = next(r for r in find_runs(missing) if r.complete)
        assert not run_m.stop_target_known and not run_m.commanded_stop and not run_m.natural_stop



def test_stop_kinds_are_exclusive_and_cover_every_complete_run() -> None:
    """Precedence switched > unknown target > commanded > natural; counts sum up (AR)."""
    rows = _thermostat_run(MONDAY, 30.0, 44.0, 46.0)
    run = next(r for r in find_runs(rows) if r.complete)
    stop = run.after
    assert stop is not None and run.stop_kind == "natural"

    def variant(**update: object) -> str | None:
        changed = [r.model_copy(update=update) if r.ts == stop.ts else r for r in rows]
        return next(r for r in find_runs(changed) if r.complete).stop_kind

    assert variant(mode="space_heating", compressor_on=True, band_high_c=40.0) == "switched"
    assert variant(mode="space_heating", compressor_on=True, band_high_c=None) == "switched"
    assert variant(band_high_c=None) == "unknown target"
    assert variant(band_high_c=40.0) == "commanded"
    both = [r.model_copy(update={"band_high_c": None}) if r.ts in (stop.ts, run.rows[-1].ts) else r for r in rows]
    assert next(r for r in find_runs(both) if r.complete).stop_kind == "unknown target"
