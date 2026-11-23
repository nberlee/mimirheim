"""MQTT publishing for the hot water draw helper.

Publishes the retained draw forecast for a mimirheim thermal boiler, the
optional mimirheim solve trigger, and the retained quality record. It does
not estimate anything.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from helper_common.publish import publish_checked

logger = logging.getLogger(__name__)


def publish_forecast(
    client: mqtt.Client,
    output_topic: str,
    steps: list[dict[str, Any]],
    *,
    signal_mimir: bool,
    mimir_trigger_topic: str | None = None,
) -> None:
    """Publish the draw forecast steps retained to ``output_topic``.

    Args:
        client: A connected paho MQTT client instance.
        output_topic: Topic of the JSON forecast array; retained so mimirheim
            has a forecast after a reconnect.
        steps: Timestamped step dicts with "ts" and "kw" (kW thermal).
        signal_mimir: If True, also publish an empty non-retained message to
            ``mimir_trigger_topic`` to trigger a mimirheim solve.
        mimir_trigger_topic: The mimirheim trigger topic. Required when
            ``signal_mimir`` is True.

    Raises:
        ValueError: If ``signal_mimir`` is True but ``mimir_trigger_topic`` is
            not provided.
    """
    if signal_mimir and not mimir_trigger_topic:
        raise ValueError("mimir_trigger_topic must be set when signal_mimir is True")
    publish_checked(
        client, output_topic, json.dumps(steps), qos=1, retain=True, description="hot water draw forecast"
    )
    logger.info("Published %d draw forecast steps to %s", len(steps), output_topic)
    if signal_mimir:
        publish_checked(
            client, mimir_trigger_topic, "", qos=0, retain=False, description="mimirheim solve trigger"
        )
        logger.debug("Signalled mimirheim via %s", mimir_trigger_topic)


def publish_quality(client: mqtt.Client, topic: str, quality: dict[str, Any]) -> None:
    """Publish the estimate's quality record retained to ``topic``.

    Separate from the daemon's run statistics topic, whose fixed payload the
    shared helper daemon publishes after every cycle.

    Args:
        client: A connected paho MQTT client instance.
        topic: Quality topic.
        quality: JSON-serialisable diagnostics of the last estimate.
    """
    publish_checked(client, topic, json.dumps(quality), qos=1, retain=True, description="hot water draw quality")
