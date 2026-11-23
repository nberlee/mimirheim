"""Unit tests for hot_water_draw.config: defaults, conversions and rejections."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hot_water_draw.config import (
    HeatSourceConfig,
    ModelConfig,
    RecorderConfig,
    TankConfig,
    TankTemperatureEntityConfig,
)


def test_capacity_and_datasheet_loss_have_known_values() -> None:
    """200 L of water holds 0.23256 kWh per K; 90 W at 45 K is 2 W/K."""
    tank = TankConfig(volume_liters=200.0, standby_loss_w=90.0, ambient_temp_c=18.0)
    assert tank.capacity_kwh_per_k() == pytest.approx(0.232556, rel=1e-5)
    assert tank.loss_kw_per_k() == pytest.approx(0.002)
    direct = TankConfig(volume_liters=200.0, standby_loss_kw_per_k=0.0031, ambient_temp_c=18.0)
    assert direct.loss_kw_per_k() == 0.0031


@pytest.mark.parametrize(
    "data",
    [
        {"volume_liters": 200.0, "ambient_temp_c": 18.0},  # no loss
        {"volume_liters": 200.0, "ambient_temp_c": 18.0, "standby_loss_w": 90.0, "standby_loss_kw_per_k": 0.002},
        {"volume_liters": 0.0, "ambient_temp_c": 18.0, "standby_loss_w": 90.0},
        {"volume_liters": 200.0, "ambient_temp_c": float("nan"), "standby_loss_w": 90.0},
        {"volume_liters": 200.0, "ambient_temp_c": 18.0, "standby_loss_w": float("inf")},
        {"volume_liters": 200.0, "ambient_temp_c": 18.0, "standby_loss_w": -1.0},
        {"volume_liters": 200.0, "standby_loss_w": 90.0},  # no ambient
        {"volume_liters": 200.0, "ambient_temp_c": 18.0, "standby_loss_w": 90.0, "extra": 1},
    ],
)
def test_invalid_tanks_are_rejected(data: dict) -> None:
    with pytest.raises(ValidationError):
        TankConfig.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"energy_entity": "sensor.e", "cop": 0.0},
        {"energy_entity": "sensor.e", "cop": float("inf")},
        {"energy_entity": "sensor.e", "cop": 2.0, "unit": "W"},
        {"cop": 2.0},
        {"energy_entity": "sensor.e", "cop": 2.0, "outlier_factor": 10.0},
    ],
)
def test_invalid_heat_sources_are_rejected(data: dict) -> None:
    with pytest.raises(ValidationError):
        HeatSourceConfig.model_validate(data)


def test_tank_sensor_units_are_limited() -> None:
    with pytest.raises(ValidationError):
        TankTemperatureEntityConfig(entity_id="sensor.t", unit="K")


@pytest.mark.parametrize("field", ["tank_temperature_entities", "heat_sources"])
def test_recorder_needs_at_least_one_entity(field: str) -> None:
    data = {
        "db_url": "sqlite:////tmp/x.db",
        "tank_temperature_entities": [{"entity_id": "sensor.t"}],
        "heat_sources": [{"energy_entity": "sensor.e", "cop": 1.0}],
    }
    data[field] = []
    with pytest.raises(ValidationError):
        RecorderConfig.model_validate(data)


def test_model_defaults() -> None:
    m = ModelConfig()
    assert (m.lookback_days, m.lookback_decay, m.min_samples_per_hour) == (14, 1.0, 3)
    assert (m.heating_threshold_kwh, m.fallback_daily_draw_kwh, m.horizon_hours) == (0.01, 6.0, 168)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lookback_days", 6),
        ("lookback_days", 91),
        ("lookback_decay", 0.0),
        ("lookback_decay", 1.1),
        ("min_samples_per_hour", 0),
        ("heating_threshold_kwh", -0.1),
        ("fallback_daily_draw_kwh", float("nan")),
        ("horizon_hours", 0),
    ],
)
def test_model_bounds_are_enforced(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({field: value})


@pytest.mark.parametrize("zone", ["Europe/Amsterdam", "UTC", "America/New_York"])
def test_valid_timezones_are_accepted(zone: str) -> None:
    assert ModelConfig(timezone=zone).timezone == zone


@pytest.mark.parametrize("zone", ["Mars/Olympus", "", "/etc/passwd", "../../etc/passwd"])
def test_invalid_timezones_are_rejected(zone: str) -> None:
    with pytest.raises(ValidationError, match="time zone"):
        ModelConfig(timezone=zone)
