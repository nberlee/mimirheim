"""Tests for HotWaterDrawDaemon._run_cycle orchestration.

The recorder is replaced with an in-memory fake; the real estimator runs.
Covers the history window, publication of the forecast and the optional
quality record, signalling mimirheim, and the failure paths (nothing
published, retained forecast left unchanged).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest
from freezegun import freeze_time

from helper_common.publish import PublishError
from hot_water_draw import __main__ as main_mod
from hot_water_draw.__main__ import HotWaterDrawDaemon
from hot_water_draw.config import HotWaterDrawConfig
from hot_water_draw.recorder import RecorderError, RecorderHistory

NOW = datetime(2026, 1, 20, 12, 7, tzinfo=UTC)
HOUR = NOW.replace(minute=0)
H = 3600


def _config(**overrides: Any) -> HotWaterDrawConfig:
    data: dict[str, Any] = {
        "mqtt": {"host": "localhost", "client_id": "test"},
        "homeassistant": {
            "db_url": "sqlite:////config/home-assistant_v2.db",
            "tank_temperature_entities": [{"entity_id": "sensor.tank"}],
            "heat_sources": [{"energy_entity": "sensor.heat", "cop": 2.0}],
        },
        "tank": {"volume_liters": 200.0, "standby_loss_kw_per_k": 0.002, "ambient_temp_c": 18.0},
        "model": {"timezone": "UTC", "lookback_days": 14, "horizon_hours": 24},
    }
    data.update(overrides)
    return HotWaterDrawConfig.model_validate(data)


def _client() -> MagicMock:
    client = MagicMock()
    client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
    return client


DaemonFactory = Callable[[HotWaterDrawConfig], HotWaterDrawDaemon]


@pytest.fixture
def daemon_factory(tmp_path: Path) -> DaemonFactory:
    def make(config: HotWaterDrawConfig) -> HotWaterDrawDaemon:
        with patch("helper_common.daemon.mqtt.Client"):
            return HotWaterDrawDaemon(config, tmp_path / "config.yaml")

    return make


def _fake_history(config: Any, *, start_ts: int, end_ts: int) -> RecorderHistory:
    """A tank that only loses standby heat: no draws."""
    hours = range(start_ts, end_ts, H)
    return RecorderHistory(
        heat_kwh={"sensor.heat": {ts: 0.0 for ts in hours}},
        tank_temp_c={"sensor.tank": {ts: 50.0 for ts in hours}},
    )


@freeze_time(NOW)
def test_cycle_reads_the_window_and_publishes(daemon_factory: DaemonFactory) -> None:
    config = _config()
    daemon = daemon_factory(config)
    client = _client()
    recorder = MagicMock(side_effect=_fake_history)
    with patch.object(main_mod, "fetch_history", recorder):
        result = daemon._run_cycle(client)

    end = int(HOUR.timestamp())
    assert recorder.call_args.kwargs == {"start_ts": end - 14 * 24 * H, "end_ts": end}
    assert result is not None and result.success and result.horizon_hours == 24
    topic, payload = client.publish.call_args_list[0].args[:2]
    assert topic == "mimir/input/thermal_boiler/dhw/draw_forecast"
    steps = json.loads(payload)
    assert len(steps) == 25
    assert steps[0]["ts"] == HOUR.isoformat()
    assert all(step["kw"] >= 0.0 for step in steps)
    assert client.publish.call_count == 1  # no quality topic configured


@freeze_time(NOW)
def test_quality_is_published_to_its_own_topic(daemon_factory: DaemonFactory) -> None:
    daemon = daemon_factory(_config(quality_topic="mimir/status/hot_water_draw/quality"))
    client = _client()
    with patch.object(main_mod, "fetch_history", side_effect=_fake_history):
        daemon._run_cycle(client)
    topics = [c.args[0] for c in client.publish.call_args_list]
    assert topics == ["mimir/input/thermal_boiler/dhw/draw_forecast", "mimir/status/hot_water_draw/quality"]
    quality = json.loads(client.publish.call_args_list[1].args[1])
    assert quality["ts"] == NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert {
        "daily_draw_kwh", "segments", "segment_hours", "samples", "clipped_samples",
        "cells_by_level", "fallback_used",
    } <= set(quality)


@freeze_time(NOW)
def test_signal_mimir_after_the_forecast(daemon_factory: DaemonFactory) -> None:
    daemon = daemon_factory(_config(signal_mimir=True))
    client = _client()
    with patch.object(main_mod, "fetch_history", side_effect=_fake_history):
        daemon._run_cycle(client)
    topics = [c.args[0] for c in client.publish.call_args_list]
    assert topics == ["mimir/input/thermal_boiler/dhw/draw_forecast", "mimir/input/trigger"]


@freeze_time(NOW)
def test_a_recorder_failure_publishes_nothing(daemon_factory: DaemonFactory) -> None:
    daemon = daemon_factory(_config())
    client = _client()
    with patch.object(main_mod, "fetch_history", side_effect=RecorderError("db down")):
        result = daemon._run_cycle(client)
    assert result is not None and result.success is False
    client.publish.assert_not_called()


@freeze_time(NOW)
def test_a_forecast_publish_failure_stops_the_cycle(daemon_factory: DaemonFactory) -> None:
    daemon = daemon_factory(_config(quality_topic="mimir/status/hot_water_draw/quality"))
    client = _client()
    client.publish.return_value.rc = mqtt.MQTT_ERR_QUEUE_SIZE
    with patch.object(main_mod, "fetch_history", side_effect=_fake_history), pytest.raises(PublishError):
        daemon._run_cycle(client)
    assert client.publish.call_count == 1


@freeze_time(NOW)
def test_a_quality_publish_failure_keeps_the_cycle_successful(daemon_factory: DaemonFactory) -> None:
    daemon = daemon_factory(_config(quality_topic="mimir/status/hot_water_draw/quality"))
    client = _client()
    with patch.object(main_mod, "fetch_history", side_effect=_fake_history), patch.object(
        main_mod, "publish_quality", side_effect=PublishError("quality")
    ):
        result = daemon._run_cycle(client)
    assert result is not None and result.success


def test_startup_check_reports_ineligible_entities_before_any_trigger(
    daemon_factory: DaemonFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """A wrong entity is logged at startup; the daemon still runs (Config Service stays reachable)."""
    daemon = daemon_factory(_config())
    with patch.object(main_mod, "check_entities", side_effect=RecorderError("sensor.tank: not found")):
        with caplog.at_level("ERROR", logger="hot_water_draw"):
            assert daemon.check_eligibility() is False
    assert any("sensor.tank" in r.getMessage() for r in caplog.records)
    with patch.object(main_mod, "check_entities", return_value=None):
        assert daemon.check_eligibility() is True


def test_main_checks_eligibility_before_running(tmp_path: Path) -> None:
    calls: list[str] = []
    with patch.object(main_mod, "load_helper_config", return_value=_config()), patch(
        "helper_common.daemon.mqtt.Client"
    ), patch.object(
        main_mod.HotWaterDrawDaemon, "check_eligibility", lambda self: calls.append("check") or True
    ), patch.object(main_mod.HotWaterDrawDaemon, "run", lambda self: calls.append("run")), patch(
        "sys.argv", ["hot_water_draw", "--config", str(tmp_path / "c.yaml")]
    ):
        main_mod.main()
    assert calls == ["check", "run"]
