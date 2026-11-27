"""Read minute-level readings from a metrics query API with range queries.

Used for the tank and heat pump signals, which only a metrics store keeps at
full resolution. Each series is read with ``last_over_time`` over a lookback
at a 60-second step, which yields the reading in force at every minute (the
held value of a state reported on change), matching ``TankRow`` semantics.

This module only issues HTTP GET requests; it writes nothing and knows no
entity names (the selector comes from the caller's private configuration).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import httpx

_STEP = timedelta(minutes=1)
_CHUNK = timedelta(days=7)  # at one point per minute, about 10 000 points per request


def _duration(d: timedelta) -> str:
    """A PromQL duration for a whole number of seconds, e.g. ``6h`` or ``90s``."""
    seconds = int(d.total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def fetch_minute_series(
    client: httpx.Client,
    base_url: str,
    selector: str,
    start: datetime,
    end: datetime,
    lookback: timedelta,
) -> dict[datetime, float | None]:
    """The reading in force at each minute of ``[start, end)`` for one series.

    Args:
        client: HTTP client to use.
        base_url: Base URL of the query API, without ``/api/v1``.
        selector: Series selector, for example ``{entity_id="..."}``. Must
            match exactly one series.
        start: First minute (UTC, whole minute).
        end: End of the range (exclusive).
        lookback: How far back a held reading may lie. The range selector is
            left-open, so a reading exactly ``lookback`` old or older is
            missing, the same rule as ``inputs.held_minutes``.

    Returns:
        The value per minute; ``None`` where the API returned no value or a
        non-finite one (NaN or infinity).

    Raises:
        RuntimeError: If the API reports an error or returns more than one
            series.
    """
    query = f"last_over_time({selector}[{_duration(lookback)}])"
    series: dict[datetime, float | None] = {}
    t = start
    while t < end:
        chunk_end = min(t + _CHUNK, end)
        last = chunk_end - _STEP  # query_range includes both ends
        response = client.get(
            f"{base_url.rstrip('/')}/api/v1/query_range",
            params={"query": query, "start": t.timestamp(), "end": last.timestamp(), "step": "60s"},
            timeout=60.0,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(f"metrics query failed: {body.get('error', body)}")
        result = body["data"]["result"]
        if len(result) > 1:
            raise RuntimeError(f"selector {selector} matched {len(result)} series; it must match one")
        values: dict[datetime, float | None] = {}
        for ts, v in result[0]["values"] if result else []:
            value = float(v)
            values[datetime.fromtimestamp(float(ts), UTC)] = value if math.isfinite(value) else None
        minute = t
        while minute < chunk_end:
            series[minute] = values.get(minute)
            minute += _STEP
        t = chunk_end
    return series
