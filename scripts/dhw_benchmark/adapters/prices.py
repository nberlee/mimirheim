"""Fetch day-ahead spot prices for the benchmark's ``prices.csv``.

Two public sources: the Nord Pool data portal (15-minute prices, but only
about two months back) and the EnergyZero price API (hourly, years back).
``merge_prices`` builds the 15-minute series from both, preferring a
quarter's own price and holding the hourly price otherwise (benchmark spec
§1). Prices are spot prices in EUR/kWh, without taxes or supplier margins.

This module only reads public APIs; it writes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta

import httpx

from dhw_benchmark.inputs import QUARTER, PriceRow

_HOUR = timedelta(hours=1)


def fetch_hourly_prices(
    client: httpx.Client, base_url: str, start: datetime, end: datetime
) -> dict[datetime, float]:
    """Hourly spot prices in EUR/kWh from the EnergyZero API, one request per day.

    Args:
        client: HTTP client to use.
        base_url: API base URL, for example ``https://api.energyzero.nl``.
        start: First hour (UTC).
        end: End of the range (exclusive).

    Returns:
        Price per hour start in ``[start, end)``.
    """
    prices: dict[datetime, float] = {}
    day = start
    while day < end:
        day_end = min(day + timedelta(days=1), end)
        response = client.get(
            f"{base_url.rstrip('/')}/v1/energyprices",
            params={
                "fromDate": day.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "tillDate": (day_end - timedelta(milliseconds=1)).strftime("%Y-%m-%dT%H:%M:%S.999Z"),
                "interval": 4,
                "usageType": 1,
                "inclBtw": "false",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        for entry in response.json().get("Prices", []):
            ts = datetime.fromisoformat(entry["readingDate"].replace("Z", "+00:00")).astimezone(UTC)
            if start <= ts < end:
                prices[ts] = float(entry["price"])
        day = day_end
    return prices


def fetch_quarter_prices(
    fetch_day: Callable[[date], list[tuple[datetime, float]]], first: date, last: date
) -> dict[datetime, float]:
    """Quarter-hour spot prices in EUR/kWh, one delivery day at a time.

    Args:
        fetch_day: Returns ``(period start, EUR/MWh)`` pairs for one delivery
            day, or raises ``LookupError`` when the source no longer serves
            that day. Injected so tests need no network; ``nordpool_day``
            provides the real one.
        first: First delivery day.
        last: Last delivery day (inclusive).

    Returns:
        Price per period start; days the source does not serve are absent.
    """
    prices: dict[datetime, float] = {}
    day = first
    while day <= last:
        try:
            for start, eur_per_mwh in fetch_day(day):
                prices[start.astimezone(UTC)] = eur_per_mwh / 1000.0
        except LookupError:
            pass  # older than the source keeps; the hourly prices cover it
        day += timedelta(days=1)
    return prices


def nordpool_day(area: str) -> Callable[[date], list[tuple[datetime, float]]]:
    """A ``fetch_day`` for ``fetch_quarter_prices`` backed by the Nord Pool data portal.

    Args:
        area: Bidding zone, for example ``NL``.

    Returns:
        A function fetching one delivery day; it raises ``LookupError`` when
        the portal has no data for that day.
    """
    import asyncio

    import aiohttp
    from pynordpool import NordPoolClient
    from pynordpool.const import Currency

    async def _fetch(day: date) -> list[tuple[datetime, float]]:
        async with aiohttp.ClientSession() as session:
            data = await NordPoolClient(session).async_get_delivery_period(
                datetime(day.year, day.month, day.day, tzinfo=UTC), Currency.EUR, [area]
            )
        return [(entry.start, float(entry.entry[area])) for entry in data.entries if area in entry.entry]

    def fetch(day: date) -> list[tuple[datetime, float]]:
        try:
            return asyncio.run(_fetch(day))
        except Exception as exc:
            if is_unavailable_day(exc):
                raise LookupError(f"no prices for {day}") from exc
            raise

    return fetch


def is_unavailable_day(exc: BaseException) -> bool:
    """Whether a Nord Pool client error means the portal does not serve that day.

    The portal answers a day older than it keeps with an authentication
    error (HTTP 401/403), and a day without data with an empty response. The
    client retries and then wraps that error in a generic one, so the cause
    is checked too. Connection and response errors are real failures and
    must not be mistaken for a missing day.

    Args:
        exc: The exception raised by the client.

    Returns:
        True for "no data for that day", False for any other failure.
    """
    from pynordpool.exceptions import NordPoolAuthenticationError, NordPoolEmptyResponseError

    unavailable = (NordPoolAuthenticationError, NordPoolEmptyResponseError)
    return isinstance(exc, unavailable) or isinstance(exc.__cause__, unavailable)


def merge_prices(
    quarter: Mapping[datetime, float], hourly: Mapping[datetime, float], start: datetime, end: datetime
) -> list[PriceRow]:
    """The 15-minute price series: a quarter's own price, else its hour's price.

    Args:
        quarter: Quarter-hour prices in EUR/kWh by period start.
        hourly: Hourly prices in EUR/kWh by hour start.
        start: First quarter (UTC, on the grid).
        end: End of the range (exclusive).

    Returns:
        One row per quarter; ``None`` where neither source has a price.
    """
    rows: list[PriceRow] = []
    t = start
    while t < end:
        hour = t.replace(minute=0)
        price = quarter.get(t, hourly.get(hour))
        rows.append(PriceRow(ts=t, spot_eur_per_kwh=price))
        t += QUARTER
    return rows
