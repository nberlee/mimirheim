"""Configuration models of the hot water draw helper.

This module defines the root ``HotWaterDrawConfig`` (MQTT, topics, optional
Home Assistant discovery) and the recorder, tank and estimation settings the
recorder reader and the estimator use. It does not connect to anything and
does not import from mimirheim core (helpers import only ``helper_common``).
The editor presentation lives in ``hot_water_draw.formspec``.
"""

from __future__ import annotations

from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import helper_common.topics as _topics
from helper_common.config import HomeAssistantConfig, MqttConfig

DEFAULT_CLIENT_ID = "mimir-hot-water-draw"

# Tank-to-room temperature difference at which the ErP / EN 12897 standing
# loss "S" of a hot water tank is measured, in K. Same convention as
# mimirheim's thermal boiler config.
STANDBY_LOSS_REFERENCE_DELTA_K: float = 45.0

# Specific heat capacity of water in kWh per litre per kelvin.
WATER_KWH_PER_LITRE_K: float = 4186 / 3600 / 1000


class HeatSourceConfig(BaseModel):
    """One heat source of the tank, measured by a cumulative energy counter.

    Attributes:
        energy_entity: Home Assistant entity ID of the source's electric
            energy counter (long-term statistics, total or total_increasing).
        cop: Heat delivered per unit of electricity: 1.0 for an element,
            the average DHW COP for a heat pump compressor.
        unit: Unit override; when unset the unit in ``statistics_meta`` is
            used.
    """

    model_config = ConfigDict(extra="forbid")

    energy_entity: str = Field(description="Energy counter entity of this heat source.")
    cop: float = Field(gt=0, allow_inf_nan=False, description="Heat delivered per kWh of electricity.")
    unit: Optional[Literal["Wh", "kWh", "MWh"]] = Field(
        default=None, description="Unit override. When unset, the unit in statistics_meta is used."
    )


class TankTemperatureEntityConfig(BaseModel):
    """A tank temperature sensor with long-term statistics.

    Attributes:
        entity_id: Home Assistant entity ID (``state_class: measurement``).
        unit: Unit override; readings are normalised to degrees Celsius.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Tank temperature sensor entity.")
    unit: Optional[Literal["°C", "°F"]] = Field(
        default=None, description="Unit override. When unset, the unit in statistics_meta is used."
    )


class RecorderConfig(BaseModel):
    """Recorder database and the entities read from it.

    Attributes:
        db_url: SQLAlchemy URL of the Home Assistant recorder database.
            SQLite files are always opened read-only.
        tank_temperature_entities: Tank temperature sensors; their hourly
            means are averaged (for example a top and a lower sensor).
        heat_sources: Every heat source of the tank. An hour's heat input is
            known only when every source has a value for it.
    """

    model_config = ConfigDict(extra="forbid")

    db_url: str = Field(description="SQLAlchemy URL of the Home Assistant recorder database.")
    tank_temperature_entities: list[TankTemperatureEntityConfig] = Field(
        min_length=1, description="Tank temperature sensors, averaged per hour."
    )
    heat_sources: list[HeatSourceConfig] = Field(min_length=1, description="Heat sources of the tank.")


class TankConfig(BaseModel):
    """Physical properties of the tank.

    Attributes:
        volume_liters: Water volume in litres.
        standby_loss_w: Standing loss in W from the datasheet or energy label
            (at a 45 K tank-to-room difference). Exactly one of this and
            ``standby_loss_kw_per_k`` is required.
        standby_loss_kw_per_k: Heat loss in kW per K of tank-to-room
            difference.
        ambient_temp_c: Temperature of the room around the tank in °C.
    """

    model_config = ConfigDict(extra="forbid")

    volume_liters: float = Field(gt=0, allow_inf_nan=False, description="Water volume in litres.")
    standby_loss_w: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Standing loss in W (ErP 'S', at 45 K)."
    )
    standby_loss_kw_per_k: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Heat loss in kW per K. Alternative to standby_loss_w."
    )
    ambient_temp_c: float = Field(allow_inf_nan=False, description="Room temperature around the tank in °C.")

    @model_validator(mode="after")
    def _exactly_one_loss(self) -> "TankConfig":
        """Require exactly one form of the standby loss."""
        if (self.standby_loss_w is None) == (self.standby_loss_kw_per_k is None):
            raise ValueError("Set exactly one of standby_loss_w and standby_loss_kw_per_k.")
        return self

    def loss_kw_per_k(self) -> float:
        """Heat loss in kW per K of tank-to-room difference."""
        if self.standby_loss_kw_per_k is not None:
            return self.standby_loss_kw_per_k
        assert self.standby_loss_w is not None
        return self.standby_loss_w / STANDBY_LOSS_REFERENCE_DELTA_K / 1000.0

    def capacity_kwh_per_k(self) -> float:
        """Thermal capacity of the water in kWh per K."""
        return self.volume_liters * WATER_KWH_PER_LITRE_K


class ModelConfig(BaseModel):
    """Estimation settings.

    Attributes:
        timezone: IANA timezone of the household, for hour-of-day and
            weekday/weekend.
        lookback_days: Days of history used.
        lookback_decay: Per-day weight factor for shape samples; 1.0 weighs
            every day equally, 0.9 halves a sample's weight in about a week.
        min_samples_per_hour: Samples a cell needs before its own mean is used.
        heating_threshold_kwh: Hourly energy above which a source counts as
            heating in that hour.
        fallback_daily_draw_kwh: Daily draw in kWh thermal used, with a flat
            shape, when no usable history exists.
        horizon_hours: Hours of forecast published.
    """

    model_config = ConfigDict(extra="forbid")

    timezone: str = Field(default="Europe/Amsterdam", description="IANA timezone of the household.")
    lookback_days: int = Field(default=14, ge=7, le=90, description="Days of history used.")
    lookback_decay: float = Field(
        default=1.0, gt=0.0, le=1.0, description="Per-day weight factor for shape samples (1.0 = equal)."
    )
    min_samples_per_hour: int = Field(default=3, ge=1, description="Samples a cell needs for its own mean.")
    heating_threshold_kwh: float = Field(
        default=0.01, ge=0.0, allow_inf_nan=False, description="Hourly energy above which a source is heating."
    )
    fallback_daily_draw_kwh: float = Field(
        default=6.0, ge=0.0, allow_inf_nan=False, description="Daily draw (kWh thermal) without usable history."
    )
    horizon_hours: int = Field(default=168, ge=1, le=336, description="Hours of forecast published.")

    @field_validator("timezone")
    @classmethod
    def _timezone_exists(cls, value: str) -> str:
        """Reject names that are not IANA time zones (including empty strings and paths)."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA time zone {value!r}") from exc
        return value


class HotWaterDrawConfig(BaseModel):
    """Root configuration of the hot water draw helper.

    Attributes:
        mqtt: MQTT broker connection settings. ``client_id`` defaults to
            ``mimir-hot-water-draw``.
        mimir_topic_prefix: The ``mqtt.topic_prefix`` configured in mimirheim.
        mimir_thermal_boiler_name: The ``thermal_boilers`` device name in the
            mimirheim config this helper publishes to.
        trigger_topic: A message on this topic fires one cycle. Defaults to
            ``{mimir_topic_prefix}/input/tools/hot_water/trigger``.
        output_topic: Retained draw forecast topic. Defaults to the mimirheim
            draw forecast topic of ``mimir_thermal_boiler_name``.
        signal_mimir: Publish to ``mimir_trigger_topic`` after a forecast.
        mimir_trigger_topic: mimirheim trigger topic. Derived when not set.
        stats_topic: Retained per-cycle run statistics published by the
            shared helper daemon.
        quality_topic: Retained estimate diagnostics published after each
            successful cycle (best effort).
        ha_discovery: Optional Home Assistant MQTT discovery settings.
        homeassistant: Recorder database connection and entities.
        tank: Physical properties of the tank.
        model: Estimation settings.
    """

    model_config = ConfigDict(extra="forbid")

    mqtt: MqttConfig = Field(description="MQTT broker connection settings.")
    mimir_topic_prefix: str = Field(
        default="mimir", description="mimirheim mqtt.topic_prefix. Used to derive default topics."
    )
    mimir_thermal_boiler_name: str = Field(
        default="dhw", description="mimirheim thermal_boilers device name this helper publishes to."
    )
    trigger_topic: str | None = Field(
        default=None,
        description="MQTT topic that triggers a cycle. Defaults to '{mimir_topic_prefix}/input/tools/hot_water/trigger'.",
    )
    output_topic: str | None = Field(
        default=None,
        description=(
            "MQTT topic for the retained draw forecast. "
            "Defaults to '{mimir_topic_prefix}/input/thermal_boiler/{mimir_thermal_boiler_name}/draw_forecast'."
        ),
    )
    signal_mimir: bool = Field(default=False, description="Publish to mimir_trigger_topic after publishing the forecast.")
    mimir_trigger_topic: str | None = Field(
        default=None, description="mimirheim trigger topic. Derives from mimir_topic_prefix when not set."
    )
    stats_topic: str | None = Field(default=None, description="MQTT topic where per-cycle run statistics are published.")
    quality_topic: str | None = Field(default=None, description="MQTT topic where estimate diagnostics are published.")
    ha_discovery: HomeAssistantConfig | None = Field(
        default=None, description="Optional Home Assistant MQTT discovery settings."
    )
    homeassistant: RecorderConfig = Field(description="Recorder database connection and entities.")
    tank: TankConfig = Field(description="Physical properties of the tank.")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Estimation settings.")

    @model_validator(mode="after")
    def _derive_mimir_topics(self) -> "HotWaterDrawConfig":
        """Fill in mimirheim-side topics that were not explicitly set."""
        p = self.mimir_topic_prefix
        if self.trigger_topic is None:
            self.trigger_topic = _topics.helper_trigger_topic(p, "hot_water")
        if self.output_topic is None:
            self.output_topic = _topics.thermal_boiler_draw_forecast_topic(p, self.mimir_thermal_boiler_name)
        if self.mimir_trigger_topic is None:
            self.mimir_trigger_topic = _topics.trigger_topic(p)
        return self

    @model_validator(mode="after")
    def _require_trigger_topic_when_signalling(self) -> "HotWaterDrawConfig":
        """Reject an empty mimirheim trigger topic when signalling is enabled."""
        if self.signal_mimir and not self.mimir_trigger_topic:
            raise ValueError("mimir_trigger_topic must not be empty when signal_mimir is true")
        return self

    @model_validator(mode="after")
    def _set_client_id_default(self) -> "HotWaterDrawConfig":
        """Set the default MQTT client identifier when not explicitly configured."""
        if not self.mqtt.client_id:
            self.mqtt.client_id = DEFAULT_CLIENT_ID
        return self
