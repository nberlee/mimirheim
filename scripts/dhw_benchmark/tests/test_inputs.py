"""Unit tests for scripts/dhw_benchmark/inputs.py: the generic input set."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from dhw_benchmark.inputs import (
    ActualsRow,
    PriceRow,
    TankRow,
    WeatherRow,
    counter_deltas,
    held_minutes,
    hourly_energy_to_quarters,
    quarter_means,
    read_rows,
    write_rows,
)

T0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
Q = timedelta(minutes=15)


def test_counter_deltas_differences_and_drops_resets() -> None:
    """A cumulative counter becomes per-interval energy; a reset leaves a gap."""
    samples = [(T0, 100.0), (T0 + timedelta(hours=1), 101.5), (T0 + timedelta(hours=2), 0.5), (T0 + timedelta(hours=3), 1.0)]
    assert counter_deltas(samples) == {T0: 1.5, T0 + timedelta(hours=1): None, T0 + timedelta(hours=2): 0.5}


def test_hourly_energy_spreads_evenly_and_keeps_missing() -> None:
    """1.2 kWh in an hour is 1.2 kW in each of its four quarters; energy is preserved."""
    quarters = hourly_energy_to_quarters({T0: 1.2, T0 + timedelta(hours=1): None})
    assert [quarters[T0 + i * Q] for i in range(4)] == pytest.approx([1.2] * 4)
    assert sum(quarters[T0 + i * Q] * 0.25 for i in range(4)) == pytest.approx(1.2)
    assert all(quarters[T0 + timedelta(hours=1) + i * Q] is None for i in range(4))


def test_quarter_means_are_time_weighted_with_held_values() -> None:
    """A value holds until the next sample: 40 C for 5 min then 46 C for 10 min averages 44 C."""
    samples = [(T0 - timedelta(minutes=3), 40.0), (T0 + timedelta(minutes=5), 46.0)]
    means = quarter_means(samples, T0, T0 + Q, max_age=timedelta(hours=1))
    assert means[T0] == pytest.approx(44.0)


def test_quarter_means_mark_missing_without_a_recent_sample() -> None:
    """No sample before the quarter, or only one older than max_age, means missing."""
    old = [(T0 - timedelta(hours=2), 40.0)]
    assert quarter_means([], T0, T0 + Q, max_age=timedelta(hours=1))[T0] is None
    assert quarter_means(old, T0, T0 + Q, max_age=timedelta(hours=1))[T0] is None


def test_rows_reject_unaligned_or_naive_timestamps() -> None:
    """Price rows need an aware UTC timestamp on a quarter-hour and no unknown fields."""
    with pytest.raises(ValidationError):
        PriceRow(ts=T0 + timedelta(minutes=5), spot_eur_per_kwh=0.1)
    with pytest.raises(ValidationError):
        PriceRow(ts=datetime(2026, 10, 20), spot_eur_per_kwh=0.1)
    with pytest.raises(ValidationError):
        PriceRow(ts=T0, spot_eur_per_kwh=0.1, extra=1)


def _tank(ts: datetime, **kw: object) -> TankRow:
    """A valid tank row with overrides."""
    data: dict = dict(ts=ts, t_top_c=50.0, t_lower_c=30.0, hp_electric_kw=1.5, mode="dhw",
                      compressor_on=True, element_kw=0.0, brine_in_c=10.0)
    data.update(kw)
    return TankRow(**data)


def test_tank_rows_are_on_a_minute_grid_with_a_generic_mode() -> None:
    """Tank rows are minute readings: any whole minute is valid, seconds are not."""
    _tank(T0 + timedelta(minutes=7))
    with pytest.raises(ValidationError):
        _tank(T0 + timedelta(minutes=7, seconds=30))
    with pytest.raises(ValidationError):
        _tank(T0, mode="3")


def test_held_minutes_take_the_reading_in_force_at_each_minute() -> None:
    """A probe value holds until it changes: the boundary reading is exact, not a mean."""
    samples = [(T0 - timedelta(seconds=30), 40.0), (T0 + timedelta(minutes=2, seconds=10), 46.0)]
    minutes = held_minutes(samples, T0, T0 + timedelta(minutes=4), max_age=timedelta(hours=1))
    assert [minutes[T0 + timedelta(minutes=i)] for i in range(4)] == [40.0, 40.0, 40.0, 46.0]


def test_held_minutes_mark_stale_or_unknown_readings_missing() -> None:
    """A held reading older than max_age, or none at all, is missing."""
    old = [(T0 - timedelta(hours=2), 40.0)]
    assert held_minutes(old, T0, T0 + timedelta(minutes=1), max_age=timedelta(hours=1))[T0] is None
    assert held_minutes([], T0, T0 + timedelta(minutes=1), max_age=timedelta(hours=1))[T0] is None


def test_csv_round_trip_keeps_missing_values(tmp_path: Path) -> None:
    """Missing values are written as empty fields and read back as None."""
    rows = [
        ActualsRow(ts=T0, base_load_kw=0.4, pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=8.5),
        ActualsRow(ts=T0 + Q, base_load_kw=None, pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=None),
    ]
    path = tmp_path / "actuals.csv"
    write_rows(path, rows)
    assert read_rows(path, ActualsRow) == rows


def test_read_rejects_unsorted_or_duplicate_rows(tmp_path: Path) -> None:
    """Rows must be strictly increasing in time."""
    path = tmp_path / "prices.csv"
    write_rows(path, [PriceRow(ts=T0 + Q, spot_eur_per_kwh=0.1), PriceRow(ts=T0, spot_eur_per_kwh=0.2)])
    with pytest.raises(ValueError, match="sorted"):
        read_rows(path, PriceRow)


def test_tank_csv_round_trip_keeps_booleans_and_missing(tmp_path: Path) -> None:
    """Booleans and empty fields survive a write and read."""
    rows = [_tank(T0), _tank(T0 + timedelta(minutes=1), compressor_on=False, mode=None, t_top_c=None)]
    path = tmp_path / "tank.csv"
    write_rows(path, rows)
    assert read_rows(path, TankRow) == rows


def test_actuals_rows_reject_invalid_input() -> None:
    """Actuals rows need a quarter-hour timestamp, known fields and numbers."""
    with pytest.raises(ValidationError):
        ActualsRow(ts=T0 + timedelta(minutes=1), base_load_kw=0.4, pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=8.5)
    with pytest.raises(ValidationError):
        ActualsRow(ts=T0, base_load_kw="a lot", pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=8.5)
    with pytest.raises(ValidationError):
        ActualsRow(ts=T0, base_load_kw=0.4, pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=8.5, heat_pump_kw=1.0)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", float("nan"), float("inf")])
def test_non_finite_values_are_rejected_in_every_row_model(value: object) -> None:
    """NaN or infinity never pass as data; missing data is None only."""
    with pytest.raises(ValidationError):
        PriceRow(ts=T0, spot_eur_per_kwh=value)
    with pytest.raises(ValidationError):
        ActualsRow(ts=T0, base_load_kw=value, pv_kw=0.0, ev_kw=0.0, outdoor_temp_c=8.5)
    with pytest.raises(ValidationError):
        _tank(T0, t_top_c=value)


def test_held_readings_exactly_max_age_old_are_missing() -> None:
    """The age limit is left-open: exactly max_age old is too old, just under is fine."""
    limit = timedelta(hours=1)
    at_limit = [(T0 - limit, 40.0)]
    just_under = [(T0 - limit + timedelta(seconds=1), 40.0)]
    assert held_minutes(at_limit, T0, T0 + timedelta(minutes=1), max_age=limit)[T0] is None
    assert held_minutes(just_under, T0, T0 + timedelta(minutes=1), max_age=limit)[T0] == 40.0
    assert quarter_means(at_limit, T0, T0 + Q, max_age=limit)[T0] is None


def test_weather_rows_are_on_the_quarter_grid_and_reject_invalid_input() -> None:
    """Weather rows follow the quarter grid; irradiance cannot be negative."""
    WeatherRow(ts=T0, temperature_c=5.0, gti_w_m2=0.0)
    with pytest.raises(ValidationError):
        WeatherRow(ts=T0 + timedelta(minutes=5), temperature_c=5.0, gti_w_m2=0.0)
    with pytest.raises(ValidationError):
        WeatherRow(ts=T0, temperature_c=5.0, gti_w_m2=-1.0)


def test_tank_band_columns_are_optional_and_round_trip(tmp_path: Path) -> None:
    """The thermostat band is optional (not every installation logs it) and survives CSV."""
    assert _tank(T0).band_low_c is None and _tank(T0).band_high_c is None
    rows = [_tank(T0, band_low_c=45.0, band_high_c=50.0), _tank(T0 + timedelta(minutes=1))]
    path = tmp_path / "tank.csv"
    write_rows(path, rows)
    assert read_rows(path, TankRow) == rows


def test_tank_csv_without_band_columns_still_reads(tmp_path: Path) -> None:
    """A tank file exported before the band columns existed loads with both as None."""
    path = tmp_path / "tank.csv"
    path.write_text(
        "ts,t_top_c,t_lower_c,hp_electric_kw,mode,compressor_on,element_kw,brine_in_c\n"
        f"{T0.isoformat()},50.0,30.0,1.5,dhw,True,0.0,10.0\n"
    )
    row = read_rows(path, TankRow)[0]
    assert (row.band_low_c, row.band_high_c) == (None, None)
    assert row.t_top_c == 50.0
