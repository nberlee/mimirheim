"""Unit tests for the metrics and recorder adapters (no network, no real database)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import sqlalchemy as sa

from dhw_benchmark.adapters.metrics import fetch_minute_series
from dhw_benchmark.adapters.recorder import fetch_hourly_energy, fetch_hourly_means

T0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)


def _metrics_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    """A query_range endpoint answering one value per minute: the minute index."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        start = int(float(request.url.params["start"]))
        end = int(float(request.url.params["end"]))
        values = [[t, str((t - int(T0.timestamp())) // 60)] for t in range(start, end + 1, 60)]
        # Leave one minute out to check that gaps stay missing.
        values = [v for v in values if v[0] != int((T0 + timedelta(minutes=3)).timestamp())]
        return httpx.Response(200, json={"status": "success", "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]}})

    return httpx.MockTransport(handler)


def test_metrics_minute_series_covers_the_range_and_keeps_gaps() -> None:
    """Held readings per minute in [start, end); a minute without a value is missing."""
    seen: list[httpx.Request] = []
    with httpx.Client(transport=_metrics_transport(seen)) as client:
        series = fetch_minute_series(
            client, "http://metrics", '{entity_id="x"}', T0, T0 + timedelta(minutes=5), lookback=timedelta(hours=6)
        )
    assert [series[T0 + timedelta(minutes=i)] for i in range(5)] == [0.0, 1.0, 2.0, None, 4.0]
    assert "last_over_time" in seen[0].url.params["query"] and "[6h]" in seen[0].url.params["query"]
    assert seen[0].url.params["step"] == "60s"


def test_metrics_requests_are_chunked() -> None:
    """Ten days at one point per minute are fetched in chunks of at most a week."""
    seen: list[httpx.Request] = []
    with httpx.Client(transport=_metrics_transport(seen)) as client:
        series = fetch_minute_series(client, "http://metrics", "{}", T0, T0 + timedelta(days=10), lookback=timedelta(hours=1))
    assert len(seen) == 2
    assert len(series) == 10 * 24 * 60


def test_metrics_error_status_raises() -> None:
    """An error status from the API is raised, not treated as missing data."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "error", "error": "bad query"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(RuntimeError, match="bad query"):
        fetch_minute_series(client, "http://metrics", "{}", T0, T0 + timedelta(minutes=1), lookback=timedelta(hours=1))


def _recorder() -> sa.Engine:
    """An in-memory database with the recorder's two statistics tables."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT, unit_of_measurement TEXT)"))
        conn.execute(sa.text("CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, sum REAL, mean REAL)"))
        conn.execute(sa.text("INSERT INTO statistics_meta VALUES (1, 'sensor.energy_wh', 'Wh'), (2, 'sensor.temp', '°C'), (3, 'sensor.power', 'W')"))
        base = T0.timestamp()
        rows = [(1, base + h * 3600, s, None) for h, s in enumerate([10_000.0, 11_500.0, 12_000.0, 100.0, 600.0])]
        rows += [(2, base + h * 3600, None, m) for h, m in enumerate([8.0, 7.5])]
        conn.execute(sa.text("INSERT INTO statistics VALUES (:a, :b, :c, :d)"), [dict(a=a, b=b, c=c, d=d) for a, b, c, d in rows])
    return engine


def test_recorder_hourly_energy_is_keyed_by_the_hour_it_was_used() -> None:
    """A statistics row's sum is the counter at the end of its hour: the energy of
    hour h is sum(h) - sum(h-1), in kWh, keyed by h. A reset leaves a gap."""
    energy = fetch_hourly_energy(_recorder(), "sensor.energy_wh", T0, T0 + timedelta(hours=5))
    hour = timedelta(hours=1)
    # Hour 0 has no earlier row to difference against, so it is missing.
    assert energy == {T0: None, T0 + hour: 1.5, T0 + 2 * hour: 0.5, T0 + 3 * hour: None, T0 + 4 * hour: 0.5}


def test_recorder_hourly_means_are_keyed_by_hour_start() -> None:
    """A measurement row's mean belongs to the hour starting at start_ts."""
    means = fetch_hourly_means(_recorder(), "sensor.temp", T0, T0 + timedelta(hours=2))
    assert means == {T0: 8.0, T0 + timedelta(hours=1): 7.5}


def test_recorder_rejects_unknown_ids_and_non_energy_units() -> None:
    """Unknown statistics and counters in a power unit are rejected."""
    with pytest.raises(ValueError, match="not found"):
        fetch_hourly_energy(_recorder(), "sensor.missing", T0, T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="unit"):
        fetch_hourly_energy(_recorder(), "sensor.power", T0, T0 + timedelta(hours=1))


def test_recorder_energy_is_missing_across_absent_rows_and_null_sums() -> None:
    """No energy is accumulated across a gap; each hour after it recovers."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, statistic_id TEXT, unit_of_measurement TEXT)"))
        conn.execute(sa.text("CREATE TABLE statistics (metadata_id INTEGER, start_ts REAL, sum REAL, mean REAL)"))
        conn.execute(sa.text("INSERT INTO statistics_meta VALUES (1, 'sensor.e', 'kWh')"))
        base = T0.timestamp()
        # Rows for hours 0, 2, 3, 4, 5 (hour 1 absent); hour 3 has a null sum.
        sums = {0: 10.0, 2: 12.0, 3: None, 4: 13.0, 5: 13.5}
        conn.execute(
            sa.text("INSERT INTO statistics VALUES (1, :ts, :s, NULL)"),
            [dict(ts=base + h * 3600, s=v) for h, v in sums.items()],
        )
    energy = fetch_hourly_energy(engine, "sensor.e", T0, T0 + timedelta(hours=6))
    hour = timedelta(hours=1)
    # The cumulative value of row h is at the end of hour h, so the energy of
    # hour h needs rows h-1 and h; hours 2, 3 and 4 touch the gap or the null.
    assert energy == {T0: None, T0 + 2 * hour: None, T0 + 3 * hour: None, T0 + 4 * hour: None, T0 + 5 * hour: 0.5}


def test_metrics_non_finite_values_are_missing() -> None:
    """NaN or infinity from the API is a missing reading, never a value."""
    def handler(request: httpx.Request) -> httpx.Response:
        t = int(T0.timestamp())
        return httpx.Response(200, json={"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {}, "values": [[t, "NaN"], [t + 60, "+Inf"], [t + 120, "2.5"]]}]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        series = fetch_minute_series(client, "http://metrics", "{}", T0, T0 + timedelta(minutes=3), lookback=timedelta(hours=1))
    assert list(series.values()) == [None, None, 2.5]
