"""Unit tests for scripts/dhw_benchmark/calib_windows.py: calibration window selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dhw_benchmark.calib_windows import (
    DrawRule,
    cooling_windows,
    heating_windows,
    legionella_excluded,
    segments,
)
from dhw_benchmark.inputs import TankRow

T0 = datetime(2026, 11, 2, 1, 0, tzinfo=UTC)
M = timedelta(minutes=1)


def _row(i: int, **kw: object) -> TankRow:
    """An idle minute with a slowly cooling tank; overrides per field."""
    data: dict = dict(
        ts=T0 + i * M, t_top_c=50.0 - 0.002 * i, t_lower_c=30.0 - 0.002 * i, hp_electric_kw=0.05,
        mode="idle", compressor_on=False, element_kw=0.0, brine_in_c=None,
    )
    data.update(kw)
    return TankRow(**data)


def _idle(n: int, start: int = 0) -> list[TankRow]:
    """n idle minutes starting at minute index ``start``."""
    return [_row(start + i) for i in range(n)]


def _run_with_padding(top_drop: float = 0.0, missing_brine_at: int | None = None) -> list[TankRow]:
    """30 quiet minutes, a 40-minute dhw run, 30 quiet minutes, with continuous probes."""
    pre = [_row(i, t_top_c=44.06 - 0.002 * i, t_lower_c=30.06 - 0.002 * i) for i in range(30)]
    run = []
    for i in range(40):
        top = 44.0 + 0.1 * i - (top_drop if i >= 20 else 0.0)
        run.append(_row(30 + i, mode="dhw", compressor_on=True, hp_electric_kw=1.6,
                        brine_in_c=None if i == missing_brine_at else 10.5, t_top_c=top, t_lower_c=30.0 + 0.2 * i))
    end_top, end_lower = run[-1].t_top_c, run[-1].t_lower_c
    post = [_row(70 + i, t_top_c=end_top - 0.002 * i, t_lower_c=end_lower - 0.002 * i) for i in range(30)]
    return pre + run + post


def test_segments_split_on_timestamp_gaps_and_missing_probes() -> None:
    """A missing minute or a missing probe reading ends a segment."""
    rows = _idle(10) + _idle(10, start=11)  # minute 10 is absent
    rows[15] = _row(16, t_top_c=None)
    lengths = [len(s) for s in segments(rows)]
    assert lengths == [10, 5, 4]


def test_legionella_and_six_hours_after_are_excluded() -> None:
    """A legionella minute and the six hours after it are flagged."""
    rows = _idle(5) + [_row(5, mode="legionella")] + _idle(400, start=6)
    flags = legionella_excluded(rows)
    assert flags[:5] == [False] * 5
    assert all(flags[5 : 5 + 361])
    assert flags[5 + 361] is False


def test_a_quiet_idle_stretch_is_a_cooling_window() -> None:
    """Two hours or more of idle minutes without a draw are one cooling window."""
    report = cooling_windows(_idle(150), DrawRule())
    assert [len(w) for w in report.accepted] == [150]
    assert report.rejected == {}


def test_a_draw_or_space_heating_breaks_a_cooling_window() -> None:
    """A lower-probe fall of more than 0.5 K in 15 minutes is a draw; space heating
    minutes are not part of a cooling window; short pieces are dropped."""
    rows = _idle(300)
    for i in range(140, 150):
        rows[i] = _row(i, t_lower_c=29.7 - 0.1 * (i - 140))  # falls 1.0 K in 10 minutes
    rows[250] = _row(250, mode="space_heating")
    report = cooling_windows(rows, DrawRule())
    assert all(len(w) >= 120 for w in report.accepted)
    assert report.rejected.get("draw", 0) >= 1
    assert all(r.mode == "idle" for w in report.accepted for r in w)


def test_a_clean_dhw_run_becomes_a_heating_window_with_padding() -> None:
    """A dhw run with 30 quiet minutes before and after is one heating window."""
    report = heating_windows(_run_with_padding(), DrawRule())
    assert [len(w) for w in report.accepted] == [100]


def test_a_heating_run_with_a_top_probe_drop_is_rejected() -> None:
    """A top-probe fall of more than 0.3 K within the run means a draw during heating."""
    # A 2 K drop outweighs the 1.5 K the run adds in 15 minutes.
    report = heating_windows(_run_with_padding(top_drop=2.0), DrawRule())
    assert report.accepted == []
    assert report.rejected == {"draw": 1}


def test_a_heating_run_needs_brine_while_the_coil_heats() -> None:
    """Brine is required in coil-active minutes only."""
    report = heating_windows(_run_with_padding(missing_brine_at=5), DrawRule())
    assert report.rejected == {"missing_brine": 1}


def test_a_run_without_full_padding_is_incomplete() -> None:
    """A run at the edge of the data has no 30 minutes before it: not used."""
    report = heating_windows(_run_with_padding()[10:], DrawRule())
    assert report.rejected == {"incomplete": 1}


def test_draw_thresholds_can_be_scaled_for_sensitivity() -> None:
    """Doubling the thresholds accepts a stretch that the default rule rejects."""
    rows = _idle(200)
    for i in range(100, 115):
        rows[i] = _row(i, t_lower_c=29.8 - 0.05 * (i - 100))  # 0.7 K in 15 minutes
    assert cooling_windows(rows, DrawRule()).rejected.get("draw", 0) >= 1
    assert [len(w) for w in cooling_windows(rows, DrawRule().scaled(2.0)).accepted] == [200]


def test_space_heating_in_the_padding_is_allowed_but_legionella_is_not() -> None:
    """The heat pump switches straight from space heating to hot water: space heating
    in the padding keeps the window; legionella in it rejects the window."""
    rows = _run_with_padding()
    rows[10] = rows[10].model_copy(update={"mode": "space_heating"})
    assert len(heating_windows(rows, DrawRule()).accepted) == 1
    rows[80] = rows[80].model_copy(update={"mode": "legionella"})
    assert heating_windows(rows, DrawRule()).accepted == []


def test_a_draw_smaller_than_the_heating_rise_stays_hidden() -> None:
    """The spec's hidden-draw assumption: a 1 K draw while the run adds 1.5 K per
    15 minutes leaves a net rise, so the window is accepted."""
    assert len(heating_windows(_run_with_padding(top_drop=1.0), DrawRule()).accepted) == 1


def test_differences_never_span_the_padding_and_run_boundary() -> None:
    """The top probe stepping down from padding to run is not read as a draw:
    each region is checked on its own."""
    rows = _run_with_padding()
    rows = [r.model_copy(update={"t_top_c": 47.0}) if i < 30 else r for i, r in enumerate(rows)]
    assert len(heating_windows(rows, DrawRule()).accepted) == 1


def test_an_unknown_mode_in_either_padding_rejects_the_window() -> None:
    """A minute whose mode is unknown has unknown tank heat."""
    for index in (10, 80):
        rows = _run_with_padding()
        rows[index] = rows[index].model_copy(update={"mode": None})
        assert heating_windows(rows, DrawRule()).rejected == {"missing_mode": 1}


def test_brine_is_required_for_heating_inside_the_padding_too() -> None:
    """A second run reaching into the padding needs its source temperature as well."""
    rows = _run_with_padding()
    rows[85] = rows[85].model_copy(update={"mode": "dhw", "compressor_on": True, "brine_in_c": None})
    report = heating_windows(rows, DrawRule())
    assert report.accepted == [] and sum(report.rejected.values()) >= 1


def test_a_fall_exactly_at_the_threshold_is_allowed() -> None:
    """50.1 to 49.8 is 0.3 K up to rounding: allowed; 0.31 K is not, also when scaled."""
    def stretch(fall: float) -> list[TankRow]:
        # Flat at 50.1 C, then one step down that holds: the only fall is the step.
        return [_row(i, t_top_c=50.1 if i < 115 else 50.1 - fall, t_lower_c=30.0) for i in range(200)]
    assert cooling_windows(stretch(0.3), DrawRule()).rejected.get("draw", 0) == 0
    assert cooling_windows(stretch(0.31), DrawRule()).rejected.get("draw", 0) >= 1
    assert cooling_windows(stretch(0.6), DrawRule().scaled(2.0)).rejected.get("draw", 0) == 0
