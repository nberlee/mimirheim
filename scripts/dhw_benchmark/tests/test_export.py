"""Unit tests for the pure parts of scripts/dhw_benchmark/export.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from dhw_benchmark.export import (
    ExportConfig,
    build_actuals,
    build_tank_rows,
    signal_lookback,
    hourly_sum,
    sqlite_recorder_path,
    recorder_engine,
)
from dhw_benchmark.inputs import ActualsRow

T0 = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
H = timedelta(hours=1)
M = timedelta(minutes=1)


def _config(**overrides: object) -> dict:
    """A minimal valid private export config."""
    data: dict = {
        "metrics": {
            "base_url": "http://metrics",
            "tank": {
                "t_top": '{entity_id="top"}', "t_lower": '{entity_id="low"}',
                "hp_power": '{entity_id="p"}', "hp_power_kw_per_unit": 0.001,
                "mode": '{entity_id="mode"}', "mode_map": {"3": "dhw", "7": "legionella", "4": "space_heating", "99": "idle"},
                "compressor_speed": '{entity_id="speed"}', "element_step": '{entity_id="step"}',
                "element_kw_per_step": 3.0, "source_temp": '{entity_id="brine"}',
            },
        },
        "recorder": {
            "db_url": "sqlite:///:memory:",
            "grid_import": ["sensor.imp1", "sensor.imp2"], "grid_export": ["sensor.exp"],
            "pv": ["sensor.pv"], "ev": ["sensor.ev"], "heat_pump": ["sensor.hp"],
            "battery_charge": ["sensor.bc"], "battery_discharge": ["sensor.bd"],
            "battery_installed": "2026-11-16T19:00:00Z",
            "outdoor_temperature": "sensor.out",
        },
        "prices": {"hourly_base_url": "http://prices", "quarter_area": "NL"},
        "weather": {"base_url": "http://weather", "latitude": 52.0, "longitude": 5.0, "tilt": 35.0, "azimuth": 0.0},
    }
    data.update(overrides)
    return data


def test_config_validates_and_rejects_unknown_fields_and_modes() -> None:
    """The private config is strict: unknown keys and non-generic modes are rejected."""
    ExportConfig.model_validate(_config())
    with pytest.raises(ValidationError):
        ExportConfig.model_validate(_config(extra_field=1))
    bad = _config()
    bad["metrics"]["tank"]["mode_map"] = {"3": "hot_water"}
    with pytest.raises(ValidationError):
        ExportConfig.model_validate(bad)


def test_hourly_sum_adds_counters_and_propagates_gaps() -> None:
    """Hours where any listed counter is missing are missing in the sum."""
    a = {T0: 1.0, T0 + H: 2.0, T0 + 2 * H: None}
    b = {T0: 0.5, T0 + H: None, T0 + 2 * H: 1.0}
    assert hourly_sum([a, b], [T0, T0 + H, T0 + 2 * H]) == {T0: 1.5, T0 + H: None, T0 + 2 * H: None}


def test_hourly_sum_is_zero_before_installation_and_keeps_gaps_after_it() -> None:
    """Before the installation time the counters add nothing; after it, a missing
    hour (including a leading gap or a whole missing series) stays missing."""
    battery = {T0 + 3 * H: 0.4}
    hours = [T0 + i * H for i in range(5)]
    installed = T0 + 2 * H
    assert hourly_sum([battery], hours, zero_before=installed) == {
        T0: 0.0, T0 + H: 0.0, T0 + 2 * H: None, T0 + 3 * H: 0.4, T0 + 4 * H: None,
    }
    assert hourly_sum([{}], hours, zero_before=installed)[T0 + 3 * H] is None


def test_battery_counters_need_an_installation_time() -> None:
    """Listing battery counters without battery_installed is rejected."""
    bad = _config()
    del bad["recorder"]["battery_installed"]
    with pytest.raises(ValidationError, match="battery_installed"):
        ExportConfig.model_validate(bad)


@pytest.mark.parametrize("name", ["recorder.db", "rec#backup.db", "rec%20x.db", "rec 100%.db"])
def test_sqlite_recorder_opens_exactly_that_file_read_only(tmp_path: Path, name: str) -> None:
    """Special characters in the path neither change the file opened nor drop
    read-only mode: the table written beforehand is found, and writes fail."""
    import sqlite3

    import sqlalchemy as sa

    db = tmp_path / name
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE marker (x INTEGER)")
    # A database URL is URL-escaped: a literal "%" in the file name is "%25".
    url = f"sqlite:///{quote(str(db))}"
    with recorder_engine(url) as engine, engine.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM marker")).scalar() == 0
        with pytest.raises(sa.exc.OperationalError):
            conn.execute(sa.text("CREATE TABLE t (x INTEGER)"))


def test_sqlite_recorder_does_not_create_a_missing_file(tmp_path: Path) -> None:
    """A missing database file is an error, not a new empty database."""
    import sqlalchemy as sa

    missing = tmp_path / "missing.db"
    with recorder_engine(f"sqlite:///{missing}") as engine, pytest.raises(sa.exc.OperationalError):
        engine.connect()
    assert not missing.exists()


@pytest.mark.parametrize(
    "url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///file:/tmp/r.db?mode=ro&uri=true",
        "sqlite:///file::memory:?uri=true",
        "sqlite:///r.db?mode=memory&uri=true",
        "sqlite:///r.db?mode=",
        "sqlite:///r.db?uri",
    ],
)
def test_sqlite_urls_that_are_not_plain_paths_are_rejected(url: str) -> None:
    """In-memory, path-less, URI-form and option-carrying SQLite URLs are rejected."""
    with pytest.raises(ValueError, match="plain file path"):
        sqlite_recorder_path(url)


def test_other_drivers_connect_as_given() -> None:
    """Non-SQLite URLs pass through; their account is expected to be read-only."""
    assert sqlite_recorder_path("mysql+pymysql://u:p@h/db") is None


def test_recorder_engine_is_disposed_also_after_an_error() -> None:
    """The engine's pool is released when the block ends, with or without an error."""
    from unittest.mock import patch

    disposed: list[bool] = []
    real = __import__("sqlalchemy").engine.Engine.dispose

    def spy(self: object, *a: object, **k: object) -> object:
        """Record the call and dispose as usual."""
        disposed.append(True)
        return real(self, *a, **k)

    with patch("sqlalchemy.engine.Engine.dispose", spy):
        with pytest.raises(RuntimeError), recorder_engine("mysql+pymysql://u:p@h/db"):
            raise RuntimeError("boom")
    assert disposed == [True]


def test_actuals_use_the_uncontrolled_load_formula() -> None:
    """base load = import - export + pv - ev - heat pump + battery discharge - battery charge,
    spread over the hour's quarters as mean kW."""
    hour = {
        "grid_import": {T0: 2.0}, "grid_export": {T0: 0.5}, "pv": {T0: 1.0}, "ev": {T0: 0.2},
        "heat_pump": {T0: 0.8}, "battery_charge": {T0: 0.3}, "battery_discharge": {T0: 0.1},
    }
    rows = build_actuals(hour, outdoor={T0: 7.0}, start=T0, end=T0 + H)
    assert len(rows) == 4
    assert rows[0].base_load_kw == pytest.approx(1.3)
    assert rows[0].model_copy(update={"base_load_kw": 1.3}) == ActualsRow(ts=T0, base_load_kw=1.3, pv_kw=1.0, ev_kw=0.2, outdoor_temp_c=7.0)


def test_actuals_are_missing_when_a_component_is() -> None:
    """One missing term makes the base load missing; other columns keep their own gaps."""
    hour = {k: {T0: 1.0} for k in ("grid_import", "grid_export", "pv", "ev", "heat_pump", "battery_charge", "battery_discharge")}
    hour["pv"] = {T0: None}
    rows = build_actuals(hour, outdoor={}, start=T0, end=T0 + H)
    assert rows[0].base_load_kw is None and rows[0].pv_kw is None and rows[0].outdoor_temp_c is None


def test_tank_rows_map_modes_compressor_and_element() -> None:
    """Mode codes map to generic modes; speed above zero means running; element power
    is the step times the power per step; power is converted to kW."""
    series = {
        "t_top": {T0: 52.0, T0 + M: 52.1}, "t_lower": {T0: 30.0, T0 + M: 30.0},
        "hp_power": {T0: 1800.0, T0 + M: None}, "mode": {T0: 3.0, T0 + M: 42.0},
        "compressor_speed": {T0: 55.0, T0 + M: 0.0}, "element_step": {T0: 0.0, T0 + M: 2.0},
        "source_temp": {T0: 9.5, T0 + M: 9.5},
    }
    tank = ExportConfig.model_validate(_config()).metrics.tank
    rows = build_tank_rows(series, tank, start=T0, end=T0 + 2 * M)
    assert (rows[0].mode, rows[0].compressor_on, rows[0].hp_electric_kw, rows[0].element_kw) == ("dhw", True, 1.8, 0.0)
    # An unknown code is missing, not guessed.
    assert (rows[1].mode, rows[1].compressor_on, rows[1].hp_electric_kw, rows[1].element_kw) == (None, False, None, 6.0)


def test_state_signals_use_the_longer_lookback() -> None:
    """Mode, compressor speed and element step are logged on change only and can hold
    for days; temperatures and power change constantly and keep the strict lookback."""
    metrics = ExportConfig.model_validate(_config()).metrics
    assert signal_lookback("mode", metrics) == timedelta(hours=metrics.state_lookback_hours)
    assert signal_lookback("element_step", metrics) == timedelta(hours=metrics.state_lookback_hours)
    assert signal_lookback("compressor_speed", metrics) == timedelta(hours=metrics.state_lookback_hours)
    assert signal_lookback("t_top", metrics) == timedelta(hours=metrics.lookback_hours)
    assert metrics.state_lookback_hours > metrics.lookback_hours


def _source_series(pump: dict | None) -> dict:
    """Minute readings for two minutes with source temperatures 11 and 24 C."""
    series = {
        "t_top": {T0: 50.0, T0 + M: 50.0}, "t_lower": {T0: 30.0, T0 + M: 30.0},
        "hp_power": {T0: 0.0, T0 + M: 0.0}, "mode": {T0: 99.0, T0 + M: 99.0},
        "compressor_speed": {T0: 0.0, T0 + M: 0.0}, "element_step": {T0: 0.0, T0 + M: 0.0},
        "source_temp": {T0: 11.0, T0 + M: 24.0},
    }
    if pump is not None:
        series["source_pump_speed"] = pump
    return series


def _tank_with_pump() -> object:
    """Tank signals with a source pump configured."""
    data = _config()
    data["metrics"]["tank"]["source_pump_speed"] = '{entity_id="pump"}'
    return ExportConfig.model_validate(data).metrics.tank


@pytest.mark.parametrize(
    "pump",
    [
        {T0: 40.0, T0 + M: 0.0},  # running, then stopped
        {T0: 40.0, T0 + M: None},  # running, then a missing reading
        {T0: 40.0},  # running, then no reading at all
        {T0: 40.0, T0 + M: float("nan")},  # running, then not a number
    ],
)
def test_source_temperature_is_only_valid_while_its_pump_runs(pump: dict) -> None:
    """A stationary source sensor reads the inside of the heat pump, not the source:
    with a source pump configured, the temperature is missing unless the pump is
    known to run."""
    rows = build_tank_rows(_source_series(pump), _tank_with_pump(), start=T0, end=T0 + 2 * M)
    assert [r.brine_in_c for r in rows] == [11.0, None]


def test_source_temperature_without_a_pump_signal_is_kept() -> None:
    """Without a configured source pump (an air source, for instance) every reading counts."""
    tank = ExportConfig.model_validate(_config()).metrics.tank
    assert tank.source_pump_speed is None
    rows = build_tank_rows(_source_series(None), tank, start=T0, end=T0 + 2 * M)
    assert [r.brine_in_c for r in rows] == [11.0, 24.0]


def test_the_source_pump_is_a_state_signal() -> None:
    """The pump speed is logged on change, so it gets the long lookback."""
    metrics = ExportConfig.model_validate(_config()).metrics
    assert signal_lookback("source_pump_speed", metrics) == timedelta(hours=metrics.state_lookback_hours)


def test_tank_rows_carry_the_thermostat_band_when_configured() -> None:
    """With band selectors configured the band is exported; without them it is missing."""
    data = _config()
    data["metrics"]["tank"]["band_low"] = '{entity_id="b", __name__="low"}'
    data["metrics"]["tank"]["band_high"] = '{entity_id="b", __name__="high"}'
    tank = ExportConfig.model_validate(data).metrics.tank
    series = _source_series(None) | {"band_low": {T0: 45.0, T0 + M: 45.0}, "band_high": {T0: 50.0}}
    rows = build_tank_rows(series, tank, start=T0, end=T0 + 2 * M)
    assert [(r.band_low_c, r.band_high_c) for r in rows] == [(45.0, 50.0), (45.0, None)]
    plain = build_tank_rows(_source_series(None), ExportConfig.model_validate(_config()).metrics.tank, start=T0, end=T0 + M)
    assert (plain[0].band_low_c, plain[0].band_high_c) == (None, None)
    assert signal_lookback("band_low", ExportConfig.model_validate(data).metrics) == timedelta(hours=720)
