"""Entry point for the hot water draw forecast daemon.

``HotWaterDrawDaemon`` is a ``HelperDaemon`` subclass. On each trigger it
reads the tank temperatures and heat-source energy counters of the last
``lookback_days`` from the Home Assistant recorder, estimates the daily draw
and its hourly profile, and publishes an hourly draw forecast retained for a
mimirheim thermal boiler with ``use_draw_forecast``.

The base class handles the MQTT boilerplate: TLS, authentication, trigger
subscription, HA MQTT discovery, retain guard, debounce, run statistics and
signal handling. The daemon also opts into the Config Service protocol via
``helper_common.config_owner.ConfigOwnerSupport``.

It contains no data access or estimation logic of its own; it delegates to
``recorder``, ``estimator`` and ``publisher``. A recorder failure publishes
nothing, so mimirheim keeps the last retained forecast; the optional quality
record published afterwards is best effort.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

from helper_common.config import load_helper_config
from helper_common.config_owner import ConfigOwnerSupport
from helper_common.cycle import CycleResult
from helper_common.daemon import HelperDaemon
from helper_common.discovery import POWER_NO_CONFIDENCE_FORECAST_ATTRIBUTES_TEMPLATE
from helper_common.publish import PublishError

from hot_water_draw.config import HotWaterDrawConfig
from hot_water_draw.estimator import estimate, forecast
from hot_water_draw.formspec import HOT_WATER_DRAW_CONFIG_FORM_SPEC
from hot_water_draw.publisher import publish_forecast, publish_quality
from hot_water_draw.recorder import RecorderError, check_entities, fetch_history

# Named explicitly: under ``python -m`` this module is ``__main__``.
logger = logging.getLogger("hot_water_draw")

CONFIG_OWNER_ID = "hot_water_draw"
CONFIG_OWNER_DISPLAY_NAME = "Hot water draw"

_HOUR_S = 3600


class HotWaterDrawDaemon(HelperDaemon):
    """Daemon that publishes a hot water draw forecast on demand."""

    TOOL_NAME = "hot_water_draw"
    FORECAST_ATTRIBUTES_TEMPLATE = POWER_NO_CONFIDENCE_FORECAST_ATTRIBUTES_TEMPLATE

    def __init__(self, config: HotWaterDrawConfig, config_path: Path) -> None:
        """Construct the daemon and register it as a Config Service owner.

        Args:
            config: Validated tool configuration.
            config_path: Path to the YAML file this daemon was started with;
                the target of a successful validate_and_write request.
        """
        super().__init__(config)
        self._config_owner = ConfigOwnerSupport(
            owner_id=CONFIG_OWNER_ID,
            display_name=CONFIG_OWNER_DISPLAY_NAME,
            model=HotWaterDrawConfig,
            form_spec=HOT_WATER_DRAW_CONFIG_FORM_SPEC,
            config_path=config_path,
        )
        # Must be registered before self._client.connect() (called by run()).
        self._config_owner.register_last_will(self._client)

    def check_eligibility(self) -> bool:
        """Check the recorder entities once at startup and log any problem.

        A failure does not stop the daemon: it keeps running so the Config
        Service stays reachable to correct the configuration, and every cycle
        reports the same error until it is fixed.

        Returns:
            True when every entity is usable.
        """
        try:
            check_entities(self._config.homeassistant)
        except RecorderError as exc:
            logger.error("Recorder entities are not usable: %s", exc)
            return False
        return True

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any,
    ) -> None:
        super()._on_connect(client, userdata, flags, reason_code, properties)
        if reason_code.is_failure:
            return
        self._config_owner.on_connect(client)

    def _on_message(self, client: mqtt.Client, userdata: Any, message: Any) -> None:
        if self._config_owner.handle_message(client, message):
            return
        super()._on_message(client, userdata, message)

    def _on_shutdown(self) -> None:
        self._config_owner.clear_descriptor(self._client)

    def _run_cycle(self, client: mqtt.Client) -> CycleResult | None:
        """Read history, estimate the draw and publish the forecast.

        A recorder failure is logged and nothing is published, so the retained
        forecast on the output topic stays as it was. A rejected forecast
        publish raises, so the shared daemon records a failed cycle. Once the
        forecast is out, the optional quality record is best effort.

        Args:
            client: Connected paho MQTT client.

        Returns:
            The cycle outcome for the daemon's run statistics.
        """
        config = self._config
        now = datetime.now(UTC)
        now_ts = int(now.timestamp())
        window_end = now_ts // _HOUR_S * _HOUR_S
        window_start = window_end - config.model.lookback_days * 24 * _HOUR_S

        try:
            history = fetch_history(config.homeassistant, start_ts=window_start, end_ts=window_end)
        except RecorderError:
            logger.exception("Recorder read failed; retaining existing payload on %s", config.output_topic)
            return CycleResult(success=False)

        result = estimate(
            history, config.homeassistant.heat_sources, config.tank, config.model, window_end_ts=window_end
        )
        steps = [
            {"ts": datetime.fromtimestamp(ts, UTC).isoformat(), "kw": round(kw, 4)}
            for ts, kw in forecast(result, config.model, now_ts=now_ts)
        ]
        publish_forecast(
            client,
            config.output_topic,
            steps,
            signal_mimir=config.signal_mimir,
            mimir_trigger_topic=config.mimir_trigger_topic,
        )
        if config.quality_topic:
            q = result.quality
            quality = {
                "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "daily_draw_kwh": round(result.daily_draw_kwh, 4),
                "segments": q.segments,
                "segment_hours": q.segment_hours,
                "samples": q.samples,
                "clipped_samples": q.clipped_samples,
                "cells_by_level": q.cells_by_level,
                "fallback_used": q.fallback_used,
                "negative_total_clipped": q.negative_total_clipped,
            }
            # Best effort: the forecast (and any solve trigger) is already out
            # and MQTT publishes cannot be rolled back.
            try:
                publish_quality(client, config.quality_topic, quality)
            except PublishError:
                logger.exception("Quality publish failed on %s", config.quality_topic)
        logger.info(
            "Published %d draw forecast points (%.2f kWh/day, fallback=%s).",
            len(steps),
            result.daily_draw_kwh,
            result.quality.fallback_used,
        )
        return CycleResult(horizon_hours=config.model.horizon_hours)


def main() -> None:
    """Parse arguments, load the configuration and run the daemon."""
    parser = argparse.ArgumentParser(description="Hot water draw forecast tool for mimirheim")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    daemon = HotWaterDrawDaemon(
        load_helper_config(
            args.config,
            HotWaterDrawConfig,
            logger,
            owner_id=CONFIG_OWNER_ID,
            display_name=CONFIG_OWNER_DISPLAY_NAME,
            form_spec=HOT_WATER_DRAW_CONFIG_FORM_SPEC,
        ),
        Path(args.config),
    )
    daemon.check_eligibility()
    daemon.run()


if __name__ == "__main__":
    main()
