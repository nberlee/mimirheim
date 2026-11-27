"""Unit tests for the price and weather adapters (recorded responses, no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from dhw_benchmark.adapters.prices import fetch_hourly_prices, fetch_quarter_prices, is_unavailable_day, merge_prices
from dhw_benchmark.adapters.weather import fetch_archived_weather
from dhw_benchmark.inputs import PriceRow, WeatherRow

T0 = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)
Q = timedelta(minutes=15)
H = timedelta(hours=1)


def test_hourly_prices_parse_the_response_per_hour() -> None:
    """Each entry's readingDate is the hour start; the price is EUR/kWh."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"Prices": [
            {"readingDate": "2026-11-20T00:00:00Z", "price": 0.16},
            {"readingDate": "2026-11-20T01:00:00Z", "price": 0.15},
        ]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        prices = fetch_hourly_prices(client, "http://prices", T0, T0 + 2 * H)
    assert prices == {T0: 0.16, T0 + H: 0.15}
    assert seen[0].url.params["inclBtw"] == "false"


def test_quarter_prices_convert_mwh_and_skip_unavailable_days() -> None:
    """EUR/MWh becomes EUR/kWh; a day the source no longer serves is skipped."""

    def fetch_day(day: date) -> list[tuple[datetime, float]]:
        if day == date(2026, 11, 21):
            raise LookupError("not available")
        return [(T0, 160.0), (T0 + Q, 150.0)]

    prices = fetch_quarter_prices(fetch_day, date(2026, 11, 20), date(2026, 11, 22))
    assert prices == {T0: 0.16, T0 + Q: 0.15}


def test_merge_prefers_quarter_prices_and_holds_hourly_ones() -> None:
    """A quarter with its own price keeps it; otherwise the hour's price holds."""
    rows = merge_prices({T0 + Q: 0.30}, {T0: 0.20}, T0, T0 + H + Q)
    assert rows == [
        PriceRow(ts=T0, spot_eur_per_kwh=0.20),
        PriceRow(ts=T0 + Q, spot_eur_per_kwh=0.30),
        PriceRow(ts=T0 + 2 * Q, spot_eur_per_kwh=0.20),
        PriceRow(ts=T0 + 3 * Q, spot_eur_per_kwh=0.20),
        PriceRow(ts=T0 + H, spot_eur_per_kwh=None),
    ]


def _weather_handler(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"hourly": {
            "time": ["2026-11-20T00:00", "2026-11-20T01:00", "2026-11-20T02:00"],
            "temperature_2m": [5.0, 6.0, 7.0],
            "global_tilted_irradiance": [0.0, 100.0, 200.0],
        }})

    return httpx.MockTransport(handler)


def test_weather_shifts_irradiance_to_the_hour_it_averages() -> None:
    """Radiation is the mean of the preceding hour, temperature an instant value:
    the 100 W/m2 stamped 01:00 belongs to 00:00-01:00, while 6 C belongs to 01:00."""
    seen: list[httpx.Request] = []
    with httpx.Client(transport=_weather_handler(seen)) as client:
        rows = fetch_archived_weather(
            client, "http://weather", latitude=52.0, longitude=5.0, tilt=35.0, azimuth=0.0,
            start=date(2026, 11, 20), end=date(2026, 11, 20),
        )
    by_ts = {r.ts: r for r in rows}
    assert by_ts[T0] == WeatherRow(ts=T0, temperature_c=5.0, gti_w_m2=100.0)
    assert by_ts[T0 + Q].gti_w_m2 == 100.0
    assert by_ts[T0 + H] == WeatherRow(ts=T0 + H, temperature_c=6.0, gti_w_m2=200.0)
    # The last hour's irradiance is not in the response: missing, not invented.
    assert by_ts[T0 + 2 * H].gti_w_m2 is None
    assert seen[0].url.params["timezone"] == "UTC"


def test_weather_error_raises() -> None:
    """A non-success response is an error, not empty weather."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": True, "reason": "bad"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(httpx.HTTPStatusError):
        fetch_archived_weather(client, "http://weather", latitude=52.0, longitude=5.0, tilt=35.0, azimuth=0.0,
                               start=date(2026, 11, 20), end=date(2026, 11, 20))


def test_weather_requests_the_next_midnight_for_the_last_hour() -> None:
    """The 23:00-24:00 irradiance is stamped at the next midnight, so it is fetched too,
    while the output still covers only the requested days."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"hourly": {
            "time": ["2026-11-20T23:00", "2026-11-21T00:00"],
            "temperature_2m": [4.0, 3.5],
            "global_tilted_irradiance": [0.0, 7.0],
        }})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = fetch_archived_weather(client, "http://weather", latitude=52.0, longitude=5.0, tilt=35.0, azimuth=0.0,
                                      start=date(2026, 11, 20), end=date(2026, 11, 20))
    assert seen[0].url.params["end_date"] == "2026-11-21"
    assert len(rows) == 96
    assert [r.gti_w_m2 for r in rows[-4:]] == [7.0] * 4


def test_only_refused_or_empty_days_count_as_unavailable() -> None:
    """An access refusal or empty response, directly or as the cause of the
    client's wrapper error, means no data; a connection error does not."""
    from pynordpool.exceptions import (
        NordPoolAuthenticationError,
        NordPoolConnectionError,
        NordPoolEmptyResponseError,
        NordPoolError,
    )

    def wrapped(cause: Exception) -> NordPoolError:
        try:
            raise NordPoolError from cause
        except NordPoolError as exc:
            return exc

    assert is_unavailable_day(wrapped(NordPoolAuthenticationError("No access")))
    assert is_unavailable_day(NordPoolEmptyResponseError("Empty response"))
    assert not is_unavailable_day(wrapped(NordPoolConnectionError("API error")))
    assert not is_unavailable_day(OSError("network down"))
