"""Export one installation's data into the benchmark's generic input set.

Usage (from the ``scripts`` directory)::

    uv run python -m dhw_benchmark.export --config private.yaml \\
        --start 2026-10-14 --end 2026-11-27 --out data/

The configuration file maps the generic columns to this installation's
metrics selectors and recorder statistic ids. It contains entity names and
private URLs and must not be committed.

The pure parts (``hourly_sum``, ``build_actuals``, ``build_tank_rows``)
assemble rows from already fetched series; ``main`` fetches with the
adapters and writes ``prices.csv``, ``actuals.csv``, ``tank.csv`` and
``weather.csv``. All reads are read-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from dhw_benchmark.inputs import QUARTER, ActualsRow, Mode, TankRow, write_rows

_HOUR = timedelta(hours=1)
_MINUTE = timedelta(minutes=1)

# The counters that make up the uncontrolled load (benchmark spec §3), with
# their sign: base load = import - export + pv - ev - heat_pump
#                         + battery_discharge - battery_charge.
_LOAD_TERMS: dict[str, float] = {
    "grid_import": 1.0,
    "grid_export": -1.0,
    "pv": 1.0,
    "ev": -1.0,
    "heat_pump": -1.0,
    "battery_discharge": 1.0,
    "battery_charge": -1.0,
}
# Battery counters count as zero before the battery was installed (benchmark
# spec §3); the installation time comes from the configuration.
_BATTERY_TERMS = {"battery_charge", "battery_discharge"}


class _Strict(BaseModel):
    """Base for the private configuration models."""

    model_config = ConfigDict(extra="forbid")


class TankSignals(_Strict):
    """Metrics selectors and conversions for the tank and heat pump.

    Attributes:
        t_top: Selector of the top (outlet) tank probe, degrees Celsius.
        t_lower: Selector of the lower (inlet side) tank probe.
        hp_power: Selector of the heat pump's total electric power.
        hp_power_kw_per_unit: Factor from that signal's unit to kW
            (0.001 for watts).
        mode: Selector of the heat pump's mode code.
        mode_map: Mode code (as text) to generic mode; codes not listed are
            missing.
        compressor_speed: Selector of the compressor speed; above zero
            means running.
        element_step: Selector of the electric element step.
        element_kw_per_step: Element power per step in kW.
        source_temp: Selector of the source temperature (brine or air).
        source_pump_speed: Selector of the source circulation pump speed, or
            None. When set, the source temperature counts only while the pump
            runs (speed above zero): a stationary sensor reads the inside of
            the heat pump, not the source.
        band_low: Selector of the hot water thermostat's start temperature,
            or None if not logged.
        band_high: Selector of its stop temperature, or None.
    """

    t_top: str
    t_lower: str
    hp_power: str
    hp_power_kw_per_unit: float
    mode: str
    mode_map: dict[str, Mode]
    compressor_speed: str
    element_step: str
    element_kw_per_step: float
    source_temp: str
    source_pump_speed: str | None = None
    band_low: str | None = None
    band_high: str | None = None


class MetricsSource(_Strict):
    """The metrics query API.

    Attributes:
        base_url: Base URL without ``/api/v1``.
        lookback_hours: How long a held reading of a continuous signal
            (temperatures, power) stays valid.
        state_lookback_hours: The same for state signals (mode, compressor
            speed, element step), which are logged only when they change and
            can hold for days.
        tank: Tank and heat pump signals.
    """

    base_url: str
    lookback_hours: float = 6.0
    state_lookback_hours: float = 720.0
    tank: TankSignals


class RecorderSource(_Strict):
    """Recorder statistic ids per term of the uncontrolled load.

    Attributes:
        db_url: Read-only database URL.
        grid_import: Grid import counters.
        grid_export: Grid export counters.
        pv: PV production counters.
        ev: EV charging counters.
        heat_pump: Heat pump electricity counters.
        battery_charge: Battery charge counters (empty without a battery).
        battery_discharge: Battery discharge counters.
        battery_installed: When the battery was installed (UTC). Before it
            the battery terms are zero; after it a missing hour stays
            missing. Required when battery counters are listed.
        outdoor_temperature: Outdoor temperature statistic.
    """

    db_url: str
    grid_import: list[str]
    grid_export: list[str]
    pv: list[str]
    ev: list[str] = []
    heat_pump: list[str]
    battery_charge: list[str] = []
    battery_discharge: list[str] = []
    battery_installed: datetime | None = None
    outdoor_temperature: str

    @model_validator(mode="after")
    def _installation_known(self) -> RecorderSource:
        """Require the installation time when battery counters are listed."""
        if (self.battery_charge or self.battery_discharge) and self.battery_installed is None:
            raise ValueError("battery_installed is required when battery counters are listed")
        if self.battery_installed is not None and self.battery_installed.tzinfo is None:
            raise ValueError("battery_installed must be timezone-aware")
        return self


class PriceSource(_Strict):
    """Price sources.

    Attributes:
        hourly_base_url: Base URL of the hourly price API.
        quarter_area: Bidding zone for quarter-hour prices, or None to use
            hourly prices only.
    """

    hourly_base_url: str
    quarter_area: str | None = None


class WeatherSource(_Strict):
    """Archived weather forecasts for the site and its PV plane.

    Attributes:
        base_url: Archive base URL.
        latitude: Site latitude in degrees.
        longitude: Site longitude in degrees.
        tilt: PV plane tilt in degrees.
        azimuth: PV plane azimuth in degrees (0 = south).
    """

    base_url: str
    latitude: float
    longitude: float
    tilt: float
    azimuth: float


class ExportConfig(_Strict):
    """The private export configuration.

    Attributes:
        metrics: Tank and heat pump signals.
        recorder: Household counters.
        prices: Price sources.
        weather: Weather archive and site.
    """

    metrics: MetricsSource
    recorder: RecorderSource
    prices: PriceSource
    weather: WeatherSource


def hourly_sum(
    series: Iterable[Mapping[datetime, float | None]],
    hours: Iterable[datetime],
    zero_before: datetime | None = None,
) -> dict[datetime, float | None]:
    """Sum hourly energies of several counters; any missing term makes the hour missing.

    Args:
        series: kWh per hour for each counter.
        hours: The hours to produce.
        zero_before: Hours before this time count as zero for these
            counters (the device did not exist yet). From this time on, a
            missing hour stays missing.

    Returns:
        kWh per hour.
    """
    series = list(series)
    total: dict[datetime, float | None] = {}
    for hour in hours:
        if zero_before is not None and hour < zero_before:
            total[hour] = 0.0
            continue
        value: float | None = 0.0
        for s in series:
            term = s.get(hour)
            if term is None or value is None:
                value = None
                continue
            value += term
        total[hour] = value
    return total


def build_actuals(
    terms: Mapping[str, Mapping[datetime, float | None]],
    outdoor: Mapping[datetime, float | None],
    start: datetime,
    end: datetime,
) -> list[ActualsRow]:
    """Household actuals on the quarter grid from hourly terms.

    Args:
        terms: kWh per hour for each key of the uncontrolled-load formula
            (``grid_import``, ``grid_export``, ``pv``, ``ev``,
            ``heat_pump``, ``battery_charge``, ``battery_discharge``).
        outdoor: Hourly mean outdoor temperature.
        start: First quarter (UTC, whole hour).
        end: End of the range (exclusive, whole hour).

    Returns:
        One row per quarter; an hour's values hold for its four quarters
        as mean kW (kWh in one hour equals mean kW).
    """
    rows: list[ActualsRow] = []
    hour = start
    while hour < end:
        base: float | None = 0.0
        for key, sign in _LOAD_TERMS.items():
            value = terms.get(key, {}).get(hour)
            base = None if base is None or value is None else base + sign * value
        pv = terms.get("pv", {}).get(hour)
        ev = terms.get("ev", {}).get(hour)
        for i in range(4):
            rows.append(
                ActualsRow(
                    ts=hour + i * QUARTER,
                    base_load_kw=base,
                    pv_kw=pv,
                    ev_kw=ev,
                    outdoor_temp_c=outdoor.get(hour),
                )
            )
        hour += _HOUR
    return rows


def _mode(code: float | None, mode_map: Mapping[str, Mode]) -> Mode | None:
    """Map a numeric mode code to a generic mode; unknown codes are missing."""
    if code is None:
        return None
    key = str(int(code)) if float(code).is_integer() else str(code)
    return mode_map.get(key)


def build_tank_rows(
    series: Mapping[str, Mapping[datetime, float | None]],
    tank: TankSignals,
    start: datetime,
    end: datetime,
) -> list[TankRow]:
    """Tank rows on the minute grid from held minute readings.

    Args:
        series: Minute readings per signal name of ``TankSignals``
            (``t_top``, ``t_lower``, ``hp_power``, ``mode``,
            ``compressor_speed``, ``element_step``, ``source_temp``, and
            ``source_pump_speed``, ``band_low`` and ``band_high`` when
            configured).
        tank: Conversions and the mode map.
        start: First minute (UTC).
        end: End of the range (exclusive).

    Returns:
        One row per minute.
    """
    rows: list[TankRow] = []
    t = start
    while t < end:
        power = series["hp_power"].get(t)
        speed = series["compressor_speed"].get(t)
        step = series["element_step"].get(t)
        source = series["source_temp"].get(t)
        if tank.source_pump_speed is not None:
            pump = series["source_pump_speed"].get(t)
            if pump is None or not pump > 0.0:
                source = None  # stationary sensor: not the source temperature
        rows.append(
            TankRow(
                ts=t,
                t_top_c=series["t_top"].get(t),
                t_lower_c=series["t_lower"].get(t),
                hp_electric_kw=None if power is None else power * tank.hp_power_kw_per_unit,
                mode=_mode(series["mode"].get(t), tank.mode_map),
                compressor_on=None if speed is None else speed > 0.0,
                element_kw=None if step is None else step * tank.element_kw_per_step,
                brine_in_c=source,
                band_low_c=series.get("band_low", {}).get(t) if tank.band_low else None,
                band_high_c=series.get("band_high", {}).get(t) if tank.band_high else None,
            )
        )
        t += _MINUTE
    return rows


def sqlite_recorder_path(url: str) -> Path | None:
    """The database file of an SQLite recorder URL, or None for other drivers.

    Only a plain file path is accepted (``sqlite:////path/to/recorder.db`` or a
    relative ``sqlite:///recorder.db``; the path is URL-escaped, so a literal
    ``%`` is written ``%25``): the read-only connection is built
    from that path by ``recorder_engine``. In-memory databases, path-less
    URLs, ``file:`` URIs and any URL with options (``?...``, even with blank
    values) are rejected, because their meaning could bypass read-only mode
    or open a different file.

    Args:
        url: SQLAlchemy database URL.

    Returns:
        The absolute database path for SQLite, None for other drivers.

    Raises:
        ValueError: For an SQLite URL that is not a plain file path.
    """
    parsed = sa.engine.make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return None
    database = parsed.database or ""
    if "?" in url or not database or database == ":memory:" or database.startswith("file:"):
        raise ValueError(
            "an SQLite recorder URL must be a plain file path such as sqlite:////path/to/recorder.db"
        )
    return Path(database).resolve()


@contextmanager
def recorder_engine(url: str) -> Iterator[sa.Engine]:
    """A read-only recorder engine that is disposed when the block ends.

    For SQLite the file is opened through ``sqlite3`` with a ``file:`` URI
    built by ``Path.as_uri()``, which escapes every special character, and
    ``mode=ro``: the file cannot be written, and a missing file is an error
    instead of being created. Other drivers connect with the URL as given;
    use a database account that can only read.

    Args:
        url: SQLAlchemy database URL.

    Yields:
        The engine.
    """
    import sqlite3

    path = sqlite_recorder_path(url)
    if path is None:
        engine = sa.create_engine(url)
    else:
        uri = f"{path.as_uri()}?mode=ro"
        engine = sa.create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True))
    try:
        yield engine
    finally:
        engine.dispose()


_STATE_SIGNALS = {"mode", "compressor_speed", "element_step", "source_pump_speed", "band_low", "band_high"}
_OPTIONAL_SIGNALS = ("source_pump_speed", "band_low", "band_high")


def signal_lookback(name: str, metrics: MetricsSource) -> timedelta:
    """The lookback for one tank signal: longer for state signals.

    Args:
        name: Field name of the signal in ``TankSignals``.
        metrics: The metrics configuration.

    Returns:
        How old a held reading of that signal may be.
    """
    hours = metrics.state_lookback_hours if name in _STATE_SIGNALS else metrics.lookback_hours
    return timedelta(hours=hours)


def main(argv: list[str] | None = None) -> None:
    """Fetch every source for a date range and write the generic input set.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    import httpx

    from dhw_benchmark.adapters.metrics import fetch_minute_series
    from dhw_benchmark.adapters.prices import (
        fetch_hourly_prices,
        fetch_quarter_prices,
        merge_prices,
        nordpool_day,
    )
    from dhw_benchmark.adapters.recorder import fetch_hourly_energy, fetch_hourly_means
    from dhw_benchmark.adapters.weather import fetch_archived_weather

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="first day (UTC)")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="day after the last (UTC)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ExportConfig.model_validate(yaml.safe_load(args.config.read_text()))
    start = datetime(args.start.year, args.start.month, args.start.day, tzinfo=UTC)
    end = datetime(args.end.year, args.end.month, args.end.day, tzinfo=UTC)
    args.out.mkdir(parents=True, exist_ok=True)
    hours = [start + i * _HOUR for i in range(int((end - start) / _HOUR))]

    terms: dict[str, dict[datetime, float | None]] = {}
    with recorder_engine(config.recorder.db_url) as engine:
        for key in _LOAD_TERMS:
            ids: list[str] = getattr(config.recorder, key)
            terms[key] = hourly_sum(
                [fetch_hourly_energy(engine, sid, start, end) for sid in ids],
                hours,
                zero_before=config.recorder.battery_installed if key in _BATTERY_TERMS else None,
            )
        outdoor = fetch_hourly_means(engine, config.recorder.outdoor_temperature, start, end)
    write_rows(args.out / "actuals.csv", build_actuals(terms, outdoor, start, end))

    tank = config.metrics.tank
    with httpx.Client() as client:
        series = {
            name: fetch_minute_series(
                client, config.metrics.base_url, getattr(tank, name), start, end,
                signal_lookback(name, config.metrics),
            )
            for name in ("t_top", "t_lower", "hp_power", "mode", "compressor_speed", "element_step", "source_temp")
        }
        for name in _OPTIONAL_SIGNALS:
            selector = getattr(tank, name)
            if selector is not None:
                series[name] = fetch_minute_series(
                    client, config.metrics.base_url, selector, start, end, signal_lookback(name, config.metrics)
                )
        write_rows(args.out / "tank.csv", build_tank_rows(series, tank, start, end))

        hourly = fetch_hourly_prices(client, config.prices.hourly_base_url, start, end)
        quarter = (
            fetch_quarter_prices(nordpool_day(config.prices.quarter_area), args.start, args.end - timedelta(days=1))
            if config.prices.quarter_area
            else {}
        )
        write_rows(args.out / "prices.csv", merge_prices(quarter, hourly, start, end))

        weather = config.weather
        write_rows(
            args.out / "weather.csv",
            fetch_archived_weather(
                client, weather.base_url, latitude=weather.latitude, longitude=weather.longitude,
                tilt=weather.tilt, azimuth=weather.azimuth, start=args.start, end=args.end - timedelta(days=1),
            ),
        )


if __name__ == "__main__":
    main()
