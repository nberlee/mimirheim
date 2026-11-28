"""Unit tests for scripts/dhw_benchmark/calibrate.py: the calibration run and calibration.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from dhw_benchmark.calib_cv import Acceptance
from dhw_benchmark.calib_fit import FitSettings, FourNodeParams
from dhw_benchmark.calib_heat import heat_input
from dhw_benchmark.calibrate import CalibrationConfig, brine_medians, calibrate, cold_water
from dhw_benchmark.inputs import TankRow
from dhw_benchmark.plant import PlantState, step_minute

MONDAY = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
M = timedelta(minutes=1)
TRUE = FourNodeParams(
    eps_1=0.5, eps_2=0.6, flow_kw_per_k=1.2, k_a=0.01, k_b=0.02, loss_kw_per_k=0.005, cop_ref=4.2, c1=0.08, c2=-0.12
)
GEN = FitSettings(room_temp_c=18.0, cold_temp_c=12.0, aux_kw=0.05, coil_order="A", lower_node=0)


def _day(start: datetime, lo: float, top: float, runs: list[tuple[int, int]], brine: float,
         minutes: int = 600) -> list[TankRow]:
    """One day of minutes from the plant with TRUE parameters: idle, with dhw runs at
    (start minute, length); no draws."""
    plant = TRUE.plant(GEN)
    state = PlantState(temps_c=[lo, lo + (top - lo) / 3, lo + 2 * (top - lo) / 3, top])
    on_minutes = {m for s, n in runs for m in range(s, s + n)}
    rows = []
    for i in range(minutes):
        on = i in on_minutes
        row = TankRow(
            ts=start + i * M, t_top_c=state.temps_c[3], t_lower_c=state.temps_c[0],
            hp_electric_kw=1.65 if on else 0.05, mode="dhw" if on else "idle", compressor_on=on,
            element_kw=0.0, brine_in_c=brine if on else None, band_low_c=36.0, band_high_c=46.0,
        )
        rows.append(row)
        heat = heat_input(row, GEN.aux_kw, state.temps_c, TRUE.cop(), TRUE.law(GEN))
        state, _ = step_minute(plant, state, heat, draw_kwh=0.0)
    return rows


def _installation(weeks: int = 3) -> list[TankRow]:
    """Two six-hour days per week, each one dhw run between idle stretches of over two hours."""
    rows: list[TankRow] = []
    for w in range(weeks):
        for d, brine in ((0, 8.0 + w), (2, 12.0 - w)):
            rows += _day(MONDAY + timedelta(weeks=w, days=d), 16.0 + d, 26.0 + d, [(150, 25)], brine, minutes=360)
    return rows


@pytest.fixture(scope="module")
def result() -> dict:
    """One end-to-end calibration of the synthetic installation (slow, shared)."""
    config = CalibrationConfig(acceptance=Acceptance(min_heating=2, min_cooling=1))
    return calibrate(_installation(), config, starts=[TRUE])


def test_cold_water_is_the_raw_minimum_outside_legionella_or_twelve_flagged() -> None:
    """The lowest lower-probe reading outside legionella windows, if within 5-20 C (§15.2)."""
    rows = _day(MONDAY, 15.0, 44.0, [], 10.0, minutes=120)
    value, flagged, raw = cold_water(rows)
    assert (value, flagged) == (pytest.approx(raw), False) and 5.0 <= raw <= 15.0
    warm = _day(MONDAY, 28.0, 44.0, [], 10.0, minutes=60)
    assert cold_water(warm)[:2] == (12.0, True)
    cold = [r.model_copy(update={"t_lower_c": 3.0}) if i == 10 else r for i, r in enumerate(rows)]
    assert cold_water(cold)[:2] == (12.0, True)
    legionella = [r.model_copy(update={"t_lower_c": 3.0, "mode": "legionella"}) if i == 10 else r
                  for i, r in enumerate(rows)]
    assert cold_water(legionella)[1] is False


def test_brine_medians_per_week_and_overall() -> None:
    """Median source temperature of compressor dhw minutes, per ISO week and overall (I)."""
    rows = []
    for w in range(2):
        for d, brine in ((0, 8.0 + w), (2, 12.0 - w)):
            rows += _day(MONDAY + timedelta(weeks=w, days=d), 16.0, 44.0, [(10, 20)], brine, minutes=40)
    weekly, overall = brine_medians(rows)
    assert weekly == {"2026-W45": pytest.approx(10.0), "2026-W46": pytest.approx(10.0)}
    assert overall == pytest.approx(10.0)


def test_calibration_json_holds_the_label_parameters_and_unassessed_items(result: dict) -> None:
    """End to end on a synthetic installation: both coil orders, a label, the domain for
    downstream clamping, and the reduced-scope items marked not_assessed."""
    json.dumps(result, allow_nan=False)  # strict JSON: raises on NaN or infinity
    assert result["label"] == "conditional on the listed assumptions"
    assert result["central_coil_order"] in ("A", "B")
    central = result["models"][result["central_coil_order"]]
    assert central["cross_validation"]["usable_folds"] == 3
    assert central["cross_validation"]["heating_top_mae_k"] < 0.2
    assert set(central["parameters"]) == set(TRUE.__dict__)
    assert result["domain"]["source_c"] == [pytest.approx(8.0), pytest.approx(12.0)]
    assert result["cold_water"]["flagged"] is False
    assert "no default band" in result["targets"]
    for item in ("parameter_profiles", "parameter_uncertainty", "weak_parameter_scenarios",
                 "lower_probe_placement", "room_15_21_c", "draw_thresholds", "node_splits", "start_timing"):
        assert result["not_assessed"][item]["value"] == "not_assessed"
    assert result["thermostat"]["stop_rule"]["selected"] in ("tank_a_mean", "a_high", "a_low")
    assert result["limits"]["min_off_minutes"] == 5
    assert result["limits"]["observed_off_gaps"]["count"] == 0  # one run per day, days not contiguous
    for order in ("A", "B"):
        model = result["models"][order]
        assert model["domain"]["source_c"] == [pytest.approx(8.0), pytest.approx(12.0)]
        assert model["domain"]["supply_c"][0] < model["domain"]["supply_c"][1]
        cv = model["cross_validation"]
        assert cv["heating_minutes"] > 0 and cv["cooling_minutes"] > 0
        assert cv["heating_lower_mae_k"] is not None
    sens = result["sensitivities"]
    assert sens["source_offsets_k"] == [-2.0, 2.0]
    assert sens["coil_order"] != result["central_coil_order"]
    low, high = sens["cold_water_c"]
    assert 5.0 <= low <= high <= 20.0
    rules = result["thermostat"]["stop_rule"]["rules"]
    assert all(set(r["shortfalls_k"]) == {"count", "median", "max", "values"} for r in rules.values())
    assert result["windows"]["invalid_minutes"] == 0
    kinds = result["thermostat"]["stop_kinds"]
    assert sum(kinds[k] for k in ("switched", "unknown target", "commanded", "natural")) == kinds["complete_dhw_runs"]


def test_cold_water_scenarios_are_clipped() -> None:
    """Central 19 C gives scenarios 16 and 20 C (item 8)."""
    from dhw_benchmark.calibrate import _clip_cold

    assert (_clip_cold(19.0 - 3.0), _clip_cold(19.0 + 3.0)) == (16.0, 20.0)


def test_configuration_numbers_are_validated_upfront() -> None:
    """Non-finite or non-positive thresholds are rejected before any fitting."""
    from dhw_benchmark.calib_windows import DrawRule

    for bad in (
        dict(room_temp_c=float("nan")),
        dict(acceptance=Acceptance(heating_mae_k=float("inf"))),
        dict(acceptance=Acceptance(min_folds=0)),
        dict(draw_rule=DrawRule(top_fall_k=-0.1)),
        dict(acceptance=Acceptance(min_folds=float("nan"))),
        dict(acceptance=Acceptance(min_heating=2.5)),
        dict(draw_rule=DrawRule(window_minutes=float("inf"))),
    ):
        with pytest.raises(ValueError):
            CalibrationConfig(**bad)
