"""Wiring of the thermal boiler hot water draw forecast input.

The draw forecast is a timestamped series in the static-load forecast shape
([{ts, kw}], kW thermal). When ``use_draw_forecast`` is set, its topic is
derived, parsed like other power forecasts, required for readiness, limits
the horizon by its coverage, and reaches the bundle both resampled per step
(``draw_kw``) and raw (``draw_forecast``). These tests cover each step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mimirheim.__main__ import _clip_bundle
from mimirheim.config.schema import MimirheimConfig
from mimirheim.core.bundle import PowerForecastStep, PriceStep, SolveBundle, ThermalBoilerInputs
from mimirheim.core.readiness import ReadinessState
from mimirheim.io.input_parser import parse_power_forecast
from mimirheim.io.mqtt_client import MqttClient

_DRAW_TOPIC = "mimir/input/thermal_boiler/dhw/draw_forecast"
_TEMP_TOPIC = "mimir/input/thermal_boiler/dhw/temp_c"
_STANDBY_TANK = {
    "volume_liters": 180.0,
    "elec_power_kw": 2.0,
    "setpoint_c": 55.0,
    "min_temp_c": 40.0,
    "standby_loss_w": 90.0,
    "ambient_temp_c": 18.0,
}


def _config(use_draw_forecast: bool = True, explicit_topic: bool = False) -> MimirheimConfig:
    tank: dict = {**_STANDBY_TANK, "use_draw_forecast": use_draw_forecast}
    if explicit_topic:
        tank["inputs"] = {"topic_draw_forecast": _DRAW_TOPIC}
    return MimirheimConfig.model_validate(
        {
            "mqtt": {"host": "localhost", "client_id": "test"},
            "grid": {"import_limit_kw": 10.0, "export_limit_kw": 5.0},
            "thermal_boilers": {"dhw": tank},
        }
    )


def _hour() -> datetime:
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def _prices(hours: int = 24) -> list[PriceStep]:
    return [
        PriceStep(ts=_hour() + timedelta(hours=i), import_eur_per_kwh=0.2, export_eur_per_kwh=0.05)
        for i in range(hours + 1)
    ]


def _draw(hours: int, kw: float = 0.5) -> list[PowerForecastStep]:
    return [PowerForecastStep(ts=_hour() + timedelta(hours=i), kw=kw) for i in range(hours + 1)]


def _varying_draw(hours: int) -> list[PowerForecastStep]:
    """Hourly draws 0.1, 0.2, ... so a shifted timestamp changes the values."""
    return [PowerForecastStep(ts=_hour() + timedelta(hours=i), kw=0.1 * (i + 1)) for i in range(hours + 1)]


def _state(config: MimirheimConfig) -> ReadinessState:
    state = ReadinessState(config)
    state.update("mimir/input/prices", _prices())
    state.update(_TEMP_TOPIC, 48.0)
    return state


class TestTopic:
    def test_derived_when_the_draw_forecast_is_used(self) -> None:
        inputs = _config().thermal_boilers["dhw"].inputs
        assert inputs is not None and inputs.topic_draw_forecast == _DRAW_TOPIC

    def test_not_derived_otherwise(self) -> None:
        inputs = _config(use_draw_forecast=False).thermal_boilers["dhw"].inputs
        assert inputs is not None and inputs.topic_draw_forecast is None

    def test_an_explicit_topic_is_ignored_while_disabled(self) -> None:
        """Turning the feature off keeps a stored topic harmless: no subscription, no requirement."""
        config = _config(use_draw_forecast=False, explicit_topic=True)
        client = MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("x.yaml"))
        assert _DRAW_TOPIC not in client._topic_handlers
        state = _state(config)
        assert state.is_ready()
        assert state.snapshot().thermal_boiler_inputs["dhw"].draw_kw is None

    def test_parsed_as_a_power_forecast(self) -> None:
        config = _config()
        client = MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("x.yaml"))
        assert client._topic_handlers[_DRAW_TOPIC] is parse_power_forecast


class TestReadiness:
    def test_a_missing_draw_forecast_blocks_the_solve(self) -> None:
        state = _state(_config())
        assert not state.is_ready()
        state.update(_DRAW_TOPIC, _draw(24))
        assert state.is_ready()

    def test_not_required_when_unused(self) -> None:
        assert _state(_config(use_draw_forecast=False)).is_ready()

    def test_coverage_limits_the_horizon(self) -> None:
        state = _state(_config())
        state.update(_DRAW_TOPIC, _draw(6))
        bundle = state.snapshot()
        steps = len(bundle.horizon_prices)
        assert 6 * 4 - 4 <= steps <= 6 * 4

    def test_snapshot_carries_resampled_and_raw_draws(self) -> None:
        """Each 15-minute step takes the value of the hour it lies in."""
        state = _state(_config())
        raw = _varying_draw(24)
        state.update(_DRAW_TOPIC, raw)
        bundle = state.snapshot()
        tb = bundle.thermal_boiler_inputs["dhw"]
        assert tb.draw_kw is not None and len(tb.draw_kw) == len(bundle.horizon_prices)
        for t, value in enumerate(tb.draw_kw):
            step_start = bundle.solve_time_utc + timedelta(minutes=15 * t)
            hour = int((step_start - _hour()).total_seconds() // 3600)
            assert value == pytest.approx(0.1 * (hour + 1))
        assert tb.draw_forecast == raw

    def test_an_infinite_draw_is_rejected_and_the_last_valid_forecast_kept(self) -> None:
        """A non-finite payload never reaches readiness, so solves continue on the previous forecast."""
        config = _config()
        state = _state(config)
        state.update(_DRAW_TOPIC, _draw(24))
        client = MqttClient(config, state, MagicMock(), MagicMock(), Path("x.yaml"))
        bad = ('[{"ts": "%s", "kw": 1e309}]' % _hour().isoformat()).encode()
        client._on_message(client._client, None, MagicMock(topic=_DRAW_TOPIC, payload=bad))
        tb = state.snapshot().thermal_boiler_inputs["dhw"]
        assert tb.draw_kw is not None and all(v == pytest.approx(0.5) for v in tb.draw_kw)

    def test_unused_draw_forecast_leaves_the_fields_empty(self) -> None:
        tb = _state(_config(use_draw_forecast=False)).snapshot().thermal_boiler_inputs["dhw"]
        assert tb.draw_kw is None and tb.draw_forecast is None


_RAW = [
    PowerForecastStep(ts=datetime(2026, 6, 1, 12, tzinfo=UTC) + timedelta(hours=h), kw=0.1 * (h + 1))
    for h in range(6)
]


def _bundle(n: int, draw_kw: list[float] | None) -> SolveBundle:
    return SolveBundle(
        solve_time_utc=datetime(2026, 6, 1, 12, tzinfo=UTC),
        horizon_prices=[0.2] * n,
        horizon_export_prices=[0.05] * n,
        horizon_confidence=[1.0] * n,
        pv_forecast=[0.0] * n,
        base_load_forecast=[0.3] * n,
        thermal_boiler_inputs={
            "dhw": ThermalBoilerInputs(
                current_temp_c=48.0,
                draw_kw=draw_kw,
                draw_forecast=_RAW,
            )
        },
    )


class TestBundle:
    def test_draw_kw_must_match_the_horizon(self) -> None:
        with pytest.raises(ValidationError, match="draw_kw"):
            _bundle(4, [0.5, 0.5])

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_draw_kw_must_be_non_negative_and_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ThermalBoilerInputs(current_temp_c=48.0, draw_kw=[0.5, bad])

    def test_old_bundles_without_draws_validate(self) -> None:
        assert ThermalBoilerInputs.model_validate({"current_temp_c": 48.0}).draw_kw is None

    def test_clip_slices_draw_kw_and_keeps_the_raw_forecast(self) -> None:
        clipped = _clip_bundle(_bundle(8, [0.1 * i for i in range(8)]), 3)
        tb = clipped.thermal_boiler_inputs["dhw"]
        assert tb.draw_kw == pytest.approx([0.0, 0.1, 0.2])
        # The raw forecast reaches beyond the clipped (and the original) horizon and is kept whole.
        assert tb.draw_forecast == _RAW
