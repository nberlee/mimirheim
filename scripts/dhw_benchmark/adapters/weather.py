"""Fetch archived weather forecasts for the benchmark's ``weather.csv``.

Reads the Open-Meteo historical-forecast archive: forecasts by valid time,
without the time they were issued (benchmark spec §2). Two variables:
outdoor temperature (an instantaneous value at the stamped hour) and global
tilted irradiance on the PV plane (the mean over the hour *before* the
stamp). The irradiance is therefore shifted one hour back, so each row
describes the interval that starts at its timestamp.

This module only reads a public API; it writes nothing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx

from dhw_benchmark.inputs import QUARTER, WeatherRow

_HOUR = timedelta(hours=1)


def fetch_archived_weather(
    client: httpx.Client,
    base_url: str,
    *,
    latitude: float,
    longitude: float,
    tilt: float,
    azimuth: float,
    start: date,
    end: date,
) -> list[WeatherRow]:
    """Archived hourly forecasts for ``start`` to ``end`` (inclusive), on the quarter grid.

    Args:
        client: HTTP client to use.
        base_url: Archive base URL, for example
            ``https://historical-forecast-api.open-meteo.com``.
        latitude: Site latitude in degrees.
        longitude: Site longitude in degrees.
        tilt: PV plane tilt in degrees from horizontal.
        azimuth: PV plane azimuth in degrees (0 = south, as the API defines it).
        start: First day (UTC).
        end: Last day (UTC, inclusive).

    Returns:
        One row per quarter of the days; an hour's values hold for its four
        quarters, and values the response does not contain are missing.

    Raises:
        httpx.HTTPStatusError: If the archive returns an error status.
    """
    response = client.get(
        f"{base_url.rstrip('/')}/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            # One day more: the irradiance of the last hour (23:00-24:00) is
            # stamped at the following midnight.
            "end_date": (end + timedelta(days=1)).isoformat(),
            "hourly": "temperature_2m,global_tilted_irradiance",
            "tilt": tilt,
            "azimuth": azimuth,
            "timezone": "UTC",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    stamps = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in hourly["time"]]
    temperature = dict(zip(stamps, hourly["temperature_2m"], strict=True))
    # Irradiance stamped T is the mean of [T - 1 h, T): key it by T - 1 h.
    irradiance = {t - _HOUR: v for t, v in zip(stamps, hourly["global_tilted_irradiance"], strict=True)}
    rows: list[WeatherRow] = []
    t = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    while t < stop:
        hour = t.replace(minute=0)
        rows.append(WeatherRow(ts=t, temperature_c=temperature.get(hour), gti_w_m2=irradiance.get(hour)))
        t += QUARTER
    return rows
