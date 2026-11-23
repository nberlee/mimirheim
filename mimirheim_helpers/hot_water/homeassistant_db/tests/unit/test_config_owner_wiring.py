"""Tests for HotWaterDrawDaemon's Config Service protocol wiring.

Covers only the wiring between the daemon and ``ConfigOwnerSupport``, whose own
behaviour is unit-tested in ``mimirheim_helpers/common/tests``. These tests
prove the wiring adds to, and does not disturb, ``HelperDaemon``'s trigger
handling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mimirheim_shared.config_service import (
    CLEARING_PAYLOAD,
    descriptor_topic,
    state_topic,
    validate_and_write_request_topic,
)

from hot_water_draw.__main__ import CONFIG_OWNER_ID, HotWaterDrawDaemon
from hot_water_draw.config import HotWaterDrawConfig


def _reason(code: int) -> int:
    class _ReasonCode(int):
        @property
        def is_failure(self) -> bool:
            return int(self) != 0

    return _ReasonCode(code)


def _make_daemon(tmp_path: Path) -> HotWaterDrawDaemon:
    config = HotWaterDrawConfig.model_validate(
        {
            "mqtt": {"host": "localhost", "client_id": "test"},
            "homeassistant": {
                "db_url": "sqlite:////config/home-assistant_v2.db",
                "tank_temperature_entities": [{"entity_id": "sensor.tank_top"}],
                "heat_sources": [{"energy_entity": "sensor.dhw_energy", "cop": 2.5}],
            },
            "tank": {"volume_liters": 180.0, "standby_loss_w": 90.0, "ambient_temp_c": 18.0},
            "model": {"timezone": "Europe/Amsterdam"},
        }
    )
    return HotWaterDrawDaemon(config, tmp_path / "config.yaml")


def test_tool_name_and_owner_id() -> None:
    assert HotWaterDrawDaemon.TOOL_NAME == "hot_water_draw"
    assert CONFIG_OWNER_ID == "hot_water_draw"


class TestLastWill:
    def test_registers_a_last_will_clearing_the_descriptor_on_construction(
        self, tmp_path: Path
    ) -> None:
        with patch("helper_common.daemon.mqtt.Client") as client_cls:
            _make_daemon(tmp_path)

        client_cls.return_value.will_set.assert_called_once_with(
            descriptor_topic(CONFIG_OWNER_ID), payload=CLEARING_PAYLOAD, qos=1, retain=True
        )


class TestOnConnect:
    def test_publishes_the_descriptor_and_subscribes_alongside_the_trigger_topic(
        self, tmp_path: Path
    ) -> None:
        with patch("helper_common.daemon.mqtt.Client"):
            daemon = _make_daemon(tmp_path)
        client = MagicMock()

        daemon._on_connect(client, None, None, _reason(0), None)

        subscribed = [call.args[0] for call in client.subscribe.call_args_list]
        assert validate_and_write_request_topic(CONFIG_OWNER_ID) in subscribed
        assert "mimir/input/tools/hot_water/trigger" in subscribed
        published_topics = [call.args[0] for call in client.publish.call_args_list]
        assert descriptor_topic(CONFIG_OWNER_ID) in published_topics

    def test_a_refused_connection_does_not_publish_or_subscribe_for_config_service(
        self, tmp_path: Path
    ) -> None:
        with patch("helper_common.daemon.mqtt.Client"):
            daemon = _make_daemon(tmp_path)
        client = MagicMock()

        daemon._on_connect(client, None, None, _reason(5), None)

        client.subscribe.assert_not_called()
        client.publish.assert_not_called()


class TestOnMessage:
    def test_config_service_requests_are_handled_and_do_not_reach_the_trigger_logic(
        self, tmp_path: Path
    ) -> None:
        with patch("helper_common.daemon.mqtt.Client"):
            daemon = _make_daemon(tmp_path)
        message = SimpleNamespace(
            topic=validate_and_write_request_topic(CONFIG_OWNER_ID),
            payload=b"not json",
            retain=False,
        )

        with patch.object(daemon, "_run_cycle") as run_cycle:
            daemon._on_message(MagicMock(), None, message)

        run_cycle.assert_not_called()

    def test_other_messages_still_reach_the_normal_dispatch(self, tmp_path: Path) -> None:
        with patch("helper_common.daemon.mqtt.Client"):
            daemon = _make_daemon(tmp_path)
        message = SimpleNamespace(topic="homeassistant/status", payload=b"offline", retain=True)

        daemon._on_message(MagicMock(), None, message)


class TestOnShutdown:
    def test_clears_the_retained_descriptor(self, tmp_path: Path) -> None:
        with patch("helper_common.daemon.mqtt.Client"):
            daemon = _make_daemon(tmp_path)
        client = MagicMock()
        daemon._client = client

        daemon._on_shutdown()

        client.publish.assert_any_call(
            descriptor_topic(CONFIG_OWNER_ID), payload=CLEARING_PAYLOAD, qos=1, retain=True
        )
        client.publish.assert_any_call(
            state_topic(CONFIG_OWNER_ID), payload=CLEARING_PAYLOAD, qos=1, retain=True
        )
