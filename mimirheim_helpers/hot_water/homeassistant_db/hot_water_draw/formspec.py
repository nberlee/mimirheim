"""FormSpec authored for HotWaterDrawConfig, this helper's Config Owner presentation layer.

Kept as a separate, colocated artifact from ``hot_water_draw.config`` per
ADR-0002 (``mimirheim_shared/docs/adr``): the config models carry no
presentation metadata of their own. ``hot_water_draw.__main__`` combines this
FormSpec with ``HotWaterDrawConfig`` through
``helper_common.config_owner.ConfigOwnerSupport`` to build the Config Service
Descriptor.

The ``mqtt`` and ``ha_discovery`` fields nest ``helper_common``'s shared
models, so per ADR-0007 this module references ``helper_common.formspec``
instead of re-authoring their labels. Per ADR-0006,
``mimirheim_shared.alignment.assert_form_spec_complete`` checks this FormSpec
recursively at every depth.
"""

from __future__ import annotations

from helper_common.formspec import HOME_ASSISTANT_CONFIG_FORM_SPEC, MQTT_CONFIG_FORM_SPEC
from mimirheim_shared.formspec import FieldSpec, FormSpec, Tier
from mimirheim_shared.visibility import Comparison, ComparisonOperator

HEAT_SOURCE_FORM_SPEC = FormSpec(
    fields={
        "energy_entity": FieldSpec(
            label="Energy entity",
            description="Home Assistant entity ID of this heat source's electric energy counter.",
            tier=Tier.BASIC,
        ),
        "cop": FieldSpec(
            label="COP",
            description="Heat delivered per kWh of electricity: 1.0 for an element, the average DHW COP for a heat pump.",
            tier=Tier.BASIC,
        ),
        "unit": FieldSpec(
            label="Unit override",
            description="Unit override. When unset, this is auto-detected from statistics_meta.",
            tier=Tier.EXPERT,
            option_labels={
                "Wh": "Watt-hours (Wh)",
                "kWh": "Kilowatt-hours (kWh)",
                "MWh": "Megawatt-hours (MWh)",
            },
        ),
    }
)

TANK_TEMPERATURE_ENTITY_FORM_SPEC = FormSpec(
    fields={
        "entity_id": FieldSpec(
            label="Entity ID",
            description="Home Assistant entity ID of a tank temperature sensor with long-term statistics.",
            tier=Tier.BASIC,
        ),
        "unit": FieldSpec(
            label="Unit override",
            description="Unit override. When unset, this is auto-detected from statistics_meta.",
            tier=Tier.EXPERT,
            option_labels={"°C": "Degrees Celsius (°C)", "°F": "Degrees Fahrenheit (°F)"},
        ),
    }
)

RECORDER_FORM_SPEC = FormSpec(
    fields={
        "db_url": FieldSpec(
            label="DB URL",
            description="SQLAlchemy database URL of the Home Assistant recorder database.",
            tier=Tier.BASIC,
        ),
        "tank_temperature_entities": FieldSpec(
            label="Tank temperature entities",
            description="Tank temperature sensors; their hourly means are averaged (for example top and lower).",
            tier=Tier.BASIC,
            nested_form_spec=TANK_TEMPERATURE_ENTITY_FORM_SPEC,
        ),
        "heat_sources": FieldSpec(
            label="Heat sources",
            description="Every heat source of the tank with its energy counter and COP.",
            tier=Tier.BASIC,
            nested_form_spec=HEAT_SOURCE_FORM_SPEC,
        ),
    }
)

TANK_FORM_SPEC = FormSpec(
    fields={
        "volume_liters": FieldSpec(label="Tank volume (L)", description="Water volume of the tank in litres.", tier=Tier.BASIC),
        "standby_loss_w": FieldSpec(
            label="Standby loss (W)",
            description=(
                "Standing loss in W from the tank's datasheet or energy label (ErP 'S', at 45 K). "
                "Set exactly one of this and the kW/K value; clear the other one."
            ),
            tier=Tier.BASIC,
        ),
        "standby_loss_kw_per_k": FieldSpec(
            label="Standby loss (kW/K)",
            description=(
                "Heat loss in kW per K of tank-to-room difference. "
                "Set exactly one of this and the standby loss in W; clear the other one."
            ),
            tier=Tier.EXPERT,
        ),
        "ambient_temp_c": FieldSpec(
            label="Room temperature (°C)",
            description="Temperature of the room around the tank in degrees C.",
            tier=Tier.BASIC,
        ),
    }
)

MODEL_FORM_SPEC = FormSpec(
    fields={
        "timezone": FieldSpec(
            label="Timezone",
            description="IANA timezone of the household, for hour of day and weekday or weekend.",
            tier=Tier.BASIC,
        ),
        "lookback_days": FieldSpec(label="Lookback (days)", description="Days of history used.", tier=Tier.EXPERT),
        "lookback_decay": FieldSpec(
            label="Lookback decay",
            description="Per-day weight factor for the hourly profile; 1.0 weighs every day equally.",
            tier=Tier.EXPERT,
        ),
        "min_samples_per_hour": FieldSpec(
            label="Minimum samples per hour",
            description="Samples an hour of day needs before its own value is used.",
            tier=Tier.EXPERT,
        ),
        "heating_threshold_kwh": FieldSpec(
            label="Heating threshold (kWh)",
            description="Hourly energy above which a heat source counts as heating in that hour.",
            tier=Tier.EXPERT,
        ),
        "fallback_daily_draw_kwh": FieldSpec(
            label="Fallback daily draw (kWh)",
            description="Daily hot water draw in kWh of heat used while no usable history exists.",
            tier=Tier.BASIC,
        ),
        "horizon_hours": FieldSpec(
            label="Horizon (hours)", description="Hours of forecast to publish after the current hour.", tier=Tier.EXPERT
        ),
    }
)

HOT_WATER_DRAW_CONFIG_FORM_SPEC = FormSpec(
    fields={
        "homeassistant": FieldSpec(
            label="Home Assistant DB",
            description="Recorder database connection and entities.",
            tab="Home Assistant DB",
            tier=Tier.BASIC,
            nested_form_spec=RECORDER_FORM_SPEC,
        ),
        "tank": FieldSpec(
            label="Tank",
            description="Physical properties of the tank.",
            tab="Tank",
            tier=Tier.BASIC,
            nested_form_spec=TANK_FORM_SPEC,
        ),
        "model": FieldSpec(
            label="Estimation",
            description="Estimation settings.",
            tab="Estimation",
            tier=Tier.BASIC,
            nested_form_spec=MODEL_FORM_SPEC,
        ),
        "mqtt": FieldSpec(
            label="MQTT",
            description="MQTT broker connection parameters.",
            tab="MQTT",
            tier=Tier.BASIC,
            nested_form_spec=MQTT_CONFIG_FORM_SPEC,
        ),
        "signal_mimir": FieldSpec(
            label="Signal mimirheim",
            description="Publish an empty trigger message to mimirheim after publishing the forecast, so it solves immediately.",
            tab="MQTT",
            tier=Tier.BASIC,
        ),
        "ha_discovery": FieldSpec(
            label="Home Assistant discovery",
            description="Optional Home Assistant MQTT discovery settings for this tool.",
            tab="MQTT",
            tier=Tier.BASIC,
            nested_form_spec=HOME_ASSISTANT_CONFIG_FORM_SPEC,
        ),
        "mimir_topic_prefix": FieldSpec(
            label="mimirheim topic prefix",
            description="The mqtt.topic_prefix configured in mimirheim core. Used to derive the default topics.",
            tab="MQTT",
            tier=Tier.EXPERT,
        ),
        "mimir_thermal_boiler_name": FieldSpec(
            label="mimirheim thermal boiler name",
            description="The thermal_boilers device name in mimirheim this tool publishes to.",
            tab="MQTT",
            tier=Tier.BASIC,
        ),
        "trigger_topic": FieldSpec(
            label="Trigger topic",
            description="MQTT topic that fires one cycle when a message arrives.",
            tab="MQTT",
            tier=Tier.EXPERT,
        ),
        "output_topic": FieldSpec(
            label="Output topic",
            description="MQTT topic the retained draw forecast is published to. Defaults to the mimirheim draw forecast topic.",
            tab="MQTT",
            tier=Tier.EXPERT,
        ),
        "stats_topic": FieldSpec(
            label="Stats topic",
            description="MQTT topic where per-cycle run statistics are published.",
            tab="MQTT",
            tier=Tier.EXPERT,
        ),
        "quality_topic": FieldSpec(
            label="Quality topic",
            description="MQTT topic where estimate diagnostics are published after each successful cycle.",
            tab="MQTT",
            tier=Tier.EXPERT,
        ),
        "mimir_trigger_topic": FieldSpec(
            label="mimirheim trigger topic",
            description="The trigger topic to signal after publishing the forecast.",
            tab="MQTT",
            tier=Tier.EXPERT,
            visible_if=Comparison(field="signal_mimir", operator=ComparisonOperator.EQ, value=True),
        ),
    }
)
