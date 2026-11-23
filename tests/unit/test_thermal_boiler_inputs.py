"""Wiring of the optional thermal boiler heat-source inputs.

Covers the parsers (which store an INVALID marker instead of raising, so a bad
reading replaces the previous value), topic derivation, MQTT handler
registration and how readiness turns the stored readings into
``ThermalBoilerInputs``. These inputs are optional: their absence never
blocks a solve.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mimirheim.config.schema import MimirheimConfig
from mimirheim.core.bundle import PriceStep
from mimirheim.core.readiness import ReadinessState
from mimirheim.io.input_parser import (
    INVALID_READING,
    parse_current_run_minutes,
    parse_finite_float,
)
from mimirheim.io.mqtt_client import InputTopicConflictError, MqttClient, build_topic_handlers

_SOURCES_TANK = {
    "volume_liters": 180.0,
    "setpoint_c": 55.0,
    "min_temp_c": 40.0,
    "cooling_rate_k_per_hour": 1.5,
    "compressor": {
        "heat_output_kw": 4.5,
        "elec_kw_at_temp": [{"temp_c": 35.0, "elec_kw": 1.1}, {"temp_c": 58.0, "elec_kw": 1.9}],
        "max_temp_c": 55.0,
        "source_temp_min_c": -5.0,
        "source_temp_max_c": 25.0,
    },
}


_BATTERY = {
    "capacity_kwh": 10.0,
    "charge_segments": [{"power_max_kw": 3.0, "efficiency": 0.95}],
    "discharge_segments": [{"power_max_kw": 3.0, "efficiency": 0.95}],
}


def _config(tank: dict | None = None, battery: bool = False) -> MimirheimConfig:
    raw: dict = {
        "mqtt": {"host": "localhost", "client_id": "test"},
        "grid": {"import_limit_kw": 10.0, "export_limit_kw": 5.0},
        "thermal_boilers": {"dhw": tank or _SOURCES_TANK},
    }
    if battery:
        raw["batteries"] = {"home": _BATTERY}
    return MimirheimConfig.model_validate(raw)


class TestParsers:
    @pytest.mark.parametrize("payload,expected", [(b"12.5", 12.5), (b"-3", -3.0), (b" 7 ", 7.0), (b"8.25\n", 8.25)])
    def test_finite_float(self, payload: bytes, expected: float) -> None:
        assert parse_finite_float(payload) == expected

    @pytest.mark.parametrize("payload", [b"nan", b"NaN", b"inf", b"-inf", b"1e400", b"warm", b"", b"\xff"])
    def test_finite_float_invalid(self, payload: bytes) -> None:
        assert parse_finite_float(payload) is INVALID_READING

    @pytest.mark.parametrize("payload,expected", [(b"0", 0.0), (b"37.5", 37.5)])
    def test_run_minutes(self, payload: bytes, expected: float) -> None:
        assert parse_current_run_minutes(payload) == expected

    @pytest.mark.parametrize("payload", [b"-1", b"nan", b"inf", b"x", b"\xff"])
    def test_run_minutes_invalid(self, payload: bytes) -> None:
        assert parse_current_run_minutes(payload) is INVALID_READING


class TestTopicsAndHandlers:
    def test_topics_derived_with_a_compressor(self) -> None:
        inputs = _config().thermal_boilers["dhw"].inputs
        assert inputs is not None
        assert inputs.topic_source_temp == "mimir/input/thermal_boiler/dhw/source_temp_c"
        assert inputs.topic_current_run_minutes == "mimir/input/thermal_boiler/dhw/current_run_minutes"

    def test_no_topics_without_a_compressor(self) -> None:
        legacy = {k: v for k, v in _SOURCES_TANK.items() if k != "compressor"}
        legacy["elec_power_kw"] = 3.0
        inputs = _config(legacy).thermal_boilers["dhw"].inputs
        assert inputs is not None
        assert inputs.topic_source_temp is None
        assert inputs.topic_current_run_minutes is None

    @pytest.mark.parametrize("attr", ["topic_source_temp", "topic_current_run_minutes"])
    @pytest.mark.parametrize(
        "other",
        [
            "mimir/input/thermal_boiler/dhw/temp_c",  # required tank temperature
            "mimir/input/battery/home/soc",  # required battery SOC
            "mimir/input/strategy",  # implicit subscription
            "mimir/input/trigger",  # handled before the parser lookup
        ],
    )
    def test_optional_topic_shared_with_another_input_is_rejected(self, attr: str, other: str) -> None:
        """An INVALID marker would otherwise land on a different input, or the input's parser would win."""
        config = _config({**_SOURCES_TANK, "inputs": {attr: other}}, battery=True)
        with pytest.raises(InputTopicConflictError, match=f"inputs.{attr}"):
            MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("unused.yaml"))

    def test_source_and_run_topics_may_not_share(self) -> None:
        shared = "sensors/compressor"
        config = _config({**_SOURCES_TANK, "inputs": {"topic_source_temp": shared, "topic_current_run_minutes": shared}})
        with pytest.raises(InputTopicConflictError, match="already used"):
            MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("unused.yaml"))

    def test_tanks_may_share_one_source_sensor(self) -> None:
        tank = {**_SOURCES_TANK, "inputs": {"topic_source_temp": "sensors/brine_in"}}
        config = MimirheimConfig.model_validate(
            {
                "mqtt": {"host": "localhost", "client_id": "test"},
                "grid": {"import_limit_kw": 10.0, "export_limit_kw": 5.0},
                "thermal_boilers": {"upper": tank, "lower": tank},
            }
        )
        client = MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("unused.yaml"))
        assert client._topic_handlers["sensors/brine_in"] is parse_finite_float

    def test_unrelated_config_strings_are_not_collisions(self) -> None:
        """Only subscribed topics count; a topic equal to the client id is fine."""
        config = _config({**_SOURCES_TANK, "inputs": {"topic_source_temp": "test"}})
        client = MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("unused.yaml"))
        assert client._topic_handlers["test"] is parse_finite_float

    def test_malformed_payloads_are_not_reported_as_topic_conflicts(self) -> None:
        """Only registry collisions use InputTopicConflictError; bad data stays a plain ValueError."""
        config = MimirheimConfig.model_validate(
            {
                "mqtt": {"host": "localhost", "client_id": "test"},
                "grid": {"import_limit_kw": 10.0, "export_limit_kw": 5.0},
                "ev_chargers": {
                    "car": {
                        "capacity_kwh": 60.0,
                        "charge_segments": [{"power_max_kw": 8.0, "efficiency": 1.0}],
                        "discharge_segments": [],
                        "inputs": {"soc": {"unit": "kwh"}},
                    }
                },
            }
        )
        plug_topic = config.ev_chargers["car"].inputs.plugged_in_topic
        parse_plug = build_topic_handlers(config)[plug_topic]
        with pytest.raises(ValueError) as exc:
            parse_plug(b"maybe")
        assert not isinstance(exc.value, InputTopicConflictError)

    def test_handlers_registered_with_a_compressor(self) -> None:
        config = _config()
        client = MqttClient(config, MagicMock(spec=ReadinessState), MagicMock(), MagicMock(), Path("unused.yaml"))
        inputs = config.thermal_boilers["dhw"].inputs
        assert client._topic_handlers[inputs.topic_source_temp] is parse_finite_float
        assert client._topic_handlers[inputs.topic_current_run_minutes] is parse_current_run_minutes

    @pytest.mark.parametrize("attr", ["topic_source_temp", "topic_current_run_minutes"])
    def test_invalid_utf8_replaces_the_stored_value(self, attr: str) -> None:
        """A payload that is not UTF-8 is stored as INVALID, not skipped."""
        config = _config()
        readiness = MagicMock(spec=ReadinessState)
        client = MqttClient(config, readiness, MagicMock(), MagicMock(), Path("unused.yaml"))
        topic = getattr(config.thermal_boilers["dhw"].inputs, attr)
        client._on_message(client._client, None, MagicMock(topic=topic, payload=b"\xff"))
        readiness.update.assert_called_once_with(topic, INVALID_READING)


def _ready_state(config: MimirheimConfig) -> ReadinessState:
    state = ReadinessState(config)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    state.update(
        "mimir/input/prices",
        [PriceStep(ts=now + timedelta(hours=i), import_eur_per_kwh=0.2, export_eur_per_kwh=0.05) for i in range(25)],
    )
    state.update("mimir/input/thermal_boiler/dhw/temp_c", 48.0)
    return state


class TestReadiness:
    def test_absent_optional_inputs_do_not_block(self) -> None:
        state = _ready_state(_config())
        assert state.is_ready()
        tb = state.snapshot().thermal_boiler_inputs["dhw"]
        assert tb.source_temp_c is None and tb.source_temp_invalid is False
        assert tb.current_run_minutes is None

    def test_valid_then_invalid_readings_replace_the_old_value(self) -> None:
        state = _ready_state(_config())
        topic = "mimir/input/thermal_boiler/dhw/source_temp_c"
        state.update(topic, parse_finite_float(b"9.5"))
        tb = state.snapshot().thermal_boiler_inputs["dhw"]
        assert (tb.source_temp_c, tb.source_temp_invalid) == (9.5, False)
        for bad in (b"nan", b"inf"):
            state.update(topic, parse_finite_float(bad))
            tb = state.snapshot().thermal_boiler_inputs["dhw"]
            assert (tb.source_temp_c, tb.source_temp_invalid) == (None, True)
        state.update(topic, parse_finite_float(b"30"))  # outside -5..25
        tb = state.snapshot().thermal_boiler_inputs["dhw"]
        assert (tb.source_temp_c, tb.source_temp_invalid) == (None, True)
        state.update(topic, parse_finite_float(b"12"))
        tb = state.snapshot().thermal_boiler_inputs["dhw"]
        assert (tb.source_temp_c, tb.source_temp_invalid) == (12.0, False)

    def test_run_minutes_reading(self) -> None:
        state = _ready_state(_config())
        topic = "mimir/input/thermal_boiler/dhw/current_run_minutes"
        state.update(topic, parse_current_run_minutes(b"20"))
        assert state.snapshot().thermal_boiler_inputs["dhw"].current_run_minutes == 20.0
        state.update(topic, parse_current_run_minutes(b"-4"))
        assert state.snapshot().thermal_boiler_inputs["dhw"].current_run_minutes is None
