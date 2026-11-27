"""Read hourly long-term statistics from a home automation recorder database.

Used for the household signals that the recorder keeps for years: energy
counters (grid import and export, PV, EV, heat pump, battery) and
temperature means. Queries are read-only; an SQLite database should be
opened in read-only mode by the caller.

Recorder semantics: a ``statistics`` row starts at ``start_ts`` and covers
one hour. For a counter, ``sum`` is the cumulative value at the end of that
hour; for a measurement, ``mean`` is the hour's mean.

This module does not write to the database and knows no entity names.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import sqlalchemy as sa

from dhw_benchmark.inputs import counter_deltas

_HOUR = timedelta(hours=1)
_ENERGY_TO_KWH = {"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0}

_META = sa.text("SELECT id, unit_of_measurement FROM statistics_meta WHERE statistic_id = :sid")
_ROWS = sa.text(
    "SELECT start_ts, {column} FROM statistics "
    "WHERE metadata_id = :mid AND start_ts >= :start AND start_ts < :end ORDER BY start_ts"
)


def _metadata(conn: sa.Connection, statistic_id: str) -> tuple[int, str | None]:
    """Return the metadata id and unit of one statistic, or raise."""
    row = conn.execute(_META, {"sid": statistic_id}).first()
    if row is None:
        raise ValueError(f"statistic {statistic_id!r} not found in statistics_meta")
    return int(row[0]), row[1]


def fetch_hourly_energy(
    engine: sa.Engine, statistic_id: str, start: datetime, end: datetime
) -> dict[datetime, float | None]:
    """Energy in kWh per hour of ``[start, end)`` from a counter statistic.

    The energy of hour ``h`` is ``sum(h) - sum(h - 1)``; the row before
    ``start`` is read as the seed. A negative difference (counter reset) is
    missing.

    Args:
        engine: Recorder database engine.
        statistic_id: The counter's statistic id.
        start: First hour (UTC, whole hour).
        end: End of the range (exclusive).

    Returns:
        kWh per hour start. An hour whose own row exists but whose previous
        row is absent or has no sum is missing (``None``); hours without
        their own row are absent.

    Raises:
        ValueError: If the statistic does not exist or its unit is not an
            energy unit.
    """
    with engine.connect() as conn:
        mid, unit = _metadata(conn, statistic_id)
        if unit not in _ENERGY_TO_KWH:
            raise ValueError(f"statistic {statistic_id!r} has unit {unit!r}, not an energy unit (Wh, kWh, MWh)")
        rows = conn.execute(
            sa.text(_ROWS.text.format(column="sum")),
            {"mid": mid, "start": (start - _HOUR).timestamp(), "end": end.timestamp()},
        ).all()
    factor = _ENERGY_TO_KWH[unit]
    # Timestamp each cumulative value at the end of its hour, so the
    # difference between two rows is keyed by the hour it was used in. Only
    # rows exactly one hour apart are differenced: across an absent row or a
    # null sum the energy of each hour in between is unknown.
    ends = {datetime.fromtimestamp(float(ts), UTC) + _HOUR: total for ts, total in rows}
    energy: dict[datetime, float | None] = {}
    for t_end, total in ends.items():
        hour = t_end - _HOUR
        if not start <= hour < end:
            continue
        previous = ends.get(hour)
        if total is None or previous is None:
            energy[hour] = None
            continue
        energy[hour] = counter_deltas([(hour, float(previous) * factor), (t_end, float(total) * factor)])[hour]
    return energy


def fetch_hourly_means(
    engine: sa.Engine, statistic_id: str, start: datetime, end: datetime
) -> dict[datetime, float | None]:
    """Hourly means of a measurement statistic in ``[start, end)``.

    Args:
        engine: Recorder database engine.
        statistic_id: The measurement's statistic id.
        start: First hour (UTC, whole hour).
        end: End of the range (exclusive).

    Returns:
        Mean per hour start; hours without a row are absent.

    Raises:
        ValueError: If the statistic does not exist.
    """
    with engine.connect() as conn:
        mid, _ = _metadata(conn, statistic_id)
        rows = conn.execute(
            sa.text(_ROWS.text.format(column="mean")),
            {"mid": mid, "start": start.timestamp(), "end": end.timestamp()},
        ).all()
    return {
        datetime.fromtimestamp(float(ts), UTC): (None if mean is None else float(mean)) for ts, mean in rows
    }
