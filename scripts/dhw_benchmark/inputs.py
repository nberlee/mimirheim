"""The generic input set of the DHW benchmark and the conversions that build it.

The benchmark harness reads only this input set: prices and household
actuals and archived weather forecasts as CSV files on a 15-minute UTC grid,
and tank and heat pump readings as a CSV file on a 1-minute UTC grid. Anyone can run the benchmark
on their own data by producing these files; the adapters in
``dhw_benchmark.adapters`` produce them from one particular installation.

Conventions (benchmark spec §1):

- Timestamps are timezone-aware UTC. Price and actuals rows mark the start of
  a 15-minute interval and describe ``[ts, ts + 15 min)``; their power
  columns (``*_kw``) are interval means, and hourly energy is spread evenly
  over its four quarters, which preserves energy.
- Tank rows are readings on a 1-minute grid: the value in force at that
  minute (a raw sample held until the next one). This keeps probe readings at
  quarter boundaries, mode changes within a quarter, and run and off lengths,
  which the plant reconstruction and calibration need.
- Missing data stays missing (``None``, an empty CSV field); it is never
  replaced by zero here. Gap bridging is the simulation's job.

This module does not fetch data, simulate anything, or know about any
appliance or data source.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

QUARTER = timedelta(minutes=15)
"""Length of one grid interval."""

Mode = Literal["dhw", "legionella", "space_heating", "idle"]
"""What the heat pump is doing in an interval, in generic terms."""


def _aware_utc(ts: datetime) -> datetime:
    """Return ``ts`` as UTC, rejecting naive timestamps."""
    if ts.tzinfo is None:
        raise ValueError("ts must be timezone-aware")
    return ts.astimezone(UTC)


class _GridRow(BaseModel):
    """Base for one row on a UTC grid; subclasses fix the grid step."""

    # allow_inf_nan=False: a "nan" or "inf" field is rejected rather than
    # passed on, so missing data can only enter as None (an empty field).
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    ts: datetime


class _QuarterRow(_GridRow):
    """Base for one row on the 15-minute UTC grid."""

    @field_validator("ts")
    @classmethod
    def _on_grid(cls, ts: datetime) -> datetime:
        """Require an aware timestamp on a quarter-hour boundary, as UTC."""
        ts = _aware_utc(ts)
        if ts.minute % 15 or ts.second or ts.microsecond:
            raise ValueError(f"ts {ts.isoformat()} is not on a 15-minute boundary")
        return ts


class _MinuteRow(_GridRow):
    """Base for one row on the 1-minute UTC grid."""

    @field_validator("ts")
    @classmethod
    def _on_grid(cls, ts: datetime) -> datetime:
        """Require an aware timestamp on a whole minute, as UTC."""
        ts = _aware_utc(ts)
        if ts.second or ts.microsecond:
            raise ValueError(f"ts {ts.isoformat()} is not on a whole minute")
        return ts


class PriceRow(_QuarterRow):
    """Day-ahead spot price for one interval.

    Attributes:
        ts: Interval start.
        spot_eur_per_kwh: Spot price in EUR/kWh, excluding taxes and supplier
            margins (the retail formulas are applied by the simulation).
    """

    spot_eur_per_kwh: float | None


class ActualsRow(_QuarterRow):
    """What happened in the household in one interval.

    Attributes:
        ts: Interval start.
        base_load_kw: Uncontrolled household load in kW (benchmark spec §3):
            the house without the heat pump, the EV and the battery.
        pv_kw: PV production in kW.
        ev_kw: EV charging in kW, replayed as recorded.
        outdoor_temp_c: Outdoor temperature in degrees Celsius.
    """

    base_load_kw: float | None
    pv_kw: float | None
    ev_kw: float | None
    outdoor_temp_c: float | None


class TankRow(_MinuteRow):
    """Tank and heat pump readings in force at one minute.

    Attributes:
        ts: The minute.
        t_top_c: Top (outlet) tank probe in degrees Celsius.
        t_lower_c: Lower (inlet side) tank probe in degrees Celsius.
        hp_electric_kw: Heat pump electricity in kW, all uses together.
        mode: What the heat pump is doing.
        compressor_on: Whether the compressor is running.
        element_kw: Electric element power in kW (part of hp_electric_kw).
        brine_in_c: Source temperature in degrees Celsius (brine or outdoor
            air, depending on the heat pump).
        band_low_c: The appliance's hot water thermostat start temperature,
            if the installation logs it.
        band_high_c: Its stop temperature, if logged.
    """

    t_top_c: float | None
    t_lower_c: float | None
    hp_electric_kw: float | None
    mode: Mode | None
    compressor_on: bool | None
    element_kw: float | None
    brine_in_c: float | None
    band_low_c: float | None = None
    band_high_c: float | None = None


class WeatherRow(_QuarterRow):
    """Archived weather forecast for one interval.

    The archive keeps forecasts by valid time without the time they were
    issued; how much of it a planner may see at a given moment is the
    simulation's assumption (benchmark spec §2).

    Attributes:
        ts: Interval start.
        temperature_c: Forecast outdoor temperature in degrees Celsius.
        gti_w_m2: Forecast mean irradiance on the PV plane in W/m2.
    """

    temperature_c: float | None
    gti_w_m2: Annotated[float, Field(ge=0.0)] | None


Row = TypeVar("Row", bound=_GridRow)


def counter_deltas(samples: Sequence[tuple[datetime, float]]) -> dict[datetime, float | None]:
    """Turn cumulative counter readings into energy per interval.

    Each interval runs from one reading to the next and is keyed by its start.
    A negative difference is a counter reset; that interval is missing
    rather than negative.

    Args:
        samples: ``(timestamp, cumulative value)`` pairs in time order.

    Returns:
        Energy per interval, keyed by the earlier timestamp; ``None`` for a
        reset.
    """
    deltas: dict[datetime, float | None] = {}
    for (t0, v0), (_, v1) in zip(samples, samples[1:], strict=False):
        delta = v1 - v0
        deltas[t0] = delta if delta >= 0.0 else None
    return deltas


def hourly_energy_to_quarters(hourly_kwh: Mapping[datetime, float | None]) -> dict[datetime, float | None]:
    """Spread hourly energy evenly over its four quarters as mean power.

    Args:
        hourly_kwh: Energy in kWh per hour, keyed by the hour's start.

    Returns:
        Mean power in kW per quarter; the four quarters of a missing hour are
        missing. Energy is preserved: ``sum(kw * 0.25)`` over the quarters
        equals the hour's kWh.
    """
    quarters: dict[datetime, float | None] = {}
    for hour, kwh in hourly_kwh.items():
        for i in range(4):
            quarters[hour + i * QUARTER] = kwh  # kWh in one hour equals mean kW
    return quarters


def quarter_means(
    samples: Sequence[tuple[datetime, float]],
    start: datetime,
    end: datetime,
    max_age: timedelta,
) -> dict[datetime, float | None]:
    """Time-weighted mean per quarter of a sampled signal that holds its value.

    A sample's value holds until the next sample, as a state reported on
    change does. A quarter is missing when no sample is known at its start,
    or when the value held at its start is older than ``max_age``.

    Args:
        samples: ``(timestamp, value)`` pairs in time order.
        start: First quarter start, on the grid.
        end: End of the last quarter, on the grid.
        max_age: A value held at a quarter's start must be younger than this.

    Returns:
        Mean value per quarter start in ``[start, end)``.
    """
    means: dict[datetime, float | None] = {}
    t = start
    i = 0
    n = len(samples)
    while t < end:
        t_end = t + QUARTER
        # Advance to the last sample at or before the quarter start.
        while i + 1 < n and samples[i + 1][0] <= t:
            i += 1
        if n == 0 or samples[i][0] > t or t - samples[i][0] >= max_age:
            means[t] = None
            t = t_end
            continue
        total = 0.0
        cursor = t
        value = samples[i][1]
        j = i + 1
        while j < n and samples[j][0] < t_end:
            total += value * (samples[j][0] - cursor).total_seconds()
            cursor, value = samples[j][0], samples[j][1]
            j += 1
        total += value * (t_end - cursor).total_seconds()
        means[t] = total / QUARTER.total_seconds()
        t = t_end
    return means


def held_minutes(
    samples: Sequence[tuple[datetime, float]],
    start: datetime,
    end: datetime,
    max_age: timedelta,
) -> dict[datetime, float | None]:
    """The reading in force at each minute of a signal that holds its value.

    Args:
        samples: ``(timestamp, value)`` pairs in time order.
        start: First minute, on a whole minute.
        end: End of the range (exclusive).
        max_age: A held reading must be younger than this at each minute
            (the same left-open rule as a metrics range selector).

    Returns:
        The held value per minute in ``[start, end)``; ``None`` when no
        sample is known yet or the held one is older than ``max_age``.
    """
    minute = timedelta(minutes=1)
    held: dict[datetime, float | None] = {}
    i = -1
    t = start
    while t < end:
        while i + 1 < len(samples) and samples[i + 1][0] <= t:
            i += 1
        if i < 0 or t - samples[i][0] >= max_age:
            held[t] = None
        else:
            held[t] = samples[i][1]
        t += minute
    return held


def write_rows(path: Path, rows: Iterable[_GridRow]) -> None:
    """Write grid rows to a CSV file; missing values become empty fields.

    Args:
        path: Destination file.
        rows: Rows of one model type.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    fields = list(type(rows[0]).model_fields)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = row.model_dump()
            data["ts"] = row.ts.isoformat()
            writer.writerow({k: "" if v is None else v for k, v in data.items()})


def read_rows(path: Path, model: type[Row]) -> list[Row]:
    """Read and validate grid rows from a CSV file.

    Args:
        path: Source file.
        model: Row model to validate each line with.

    Returns:
        The validated rows.

    Raises:
        ValueError: If a row is invalid, or timestamps are not strictly
            increasing.
    """
    with path.open(newline="") as fh:
        rows = [
            model.model_validate({k: (None if v == "" else v) for k, v in line.items()})
            for line in csv.DictReader(fh)
        ]
    for a, b in zip(rows, rows[1:], strict=False):
        if b.ts <= a.ts:
            raise ValueError(f"{path.name}: rows must be sorted by ts without duplicates ({b.ts.isoformat()})")
    return rows
