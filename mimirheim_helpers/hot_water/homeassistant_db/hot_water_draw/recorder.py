"""Home Assistant recorder access for the hot water draw helper.

Reads hourly long-term statistics (the ``statistics`` table joined to
``statistics_meta``) from the Home Assistant recorder database:

- the energy counter of every heat source, turned into per-hour energy by
  differencing the cumulative ``sum`` column of consecutive hourly rows;
- every tank temperature sensor, read from the hourly ``mean`` column and
  normalised to degrees Celsius.

Helpers may only import ``helper_common`` (see
``mimirheim_helpers/common/AGENTS.md``), so this is a local implementation of
the same recorder access the weather base load helper uses. Unlike that
helper it applies no outlier filter to energy: a heat source legitimately has
rare, much larger hours (a resistive element next to a compressor), and a
percentile filter would discard them. Negative deltas and non-adjacent rows
are dropped.

This module does not combine entities or estimate anything. It returns
per-entity hourly values keyed by the UTC epoch second of each hour's start.
A missing key means "no trustworthy value for that hour", never zero.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import quote, quote_plus

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from hot_water_draw.config import HeatSourceConfig, RecorderConfig, TankTemperatureEntityConfig

logger = logging.getLogger(__name__)

_HOUR_S = 3600

_ENERGY_UNIT_TO_KWH: dict[str, float] = {"Wh": 1e-3, "kWh": 1.0, "MWh": 1e3}

_UNIT_QUERY = text(
    "SELECT unit_of_measurement FROM statistics_meta WHERE statistic_id = :entity_id"
)
_SUM_QUERY = text(
    """
    SELECT s.start_ts, s.sum
      FROM statistics s
      JOIN statistics_meta sm ON s.metadata_id = sm.id
     WHERE sm.statistic_id = :entity_id
       AND s.start_ts >= :from_ts
       AND s.start_ts < :end_ts
       AND s.sum IS NOT NULL
     ORDER BY s.start_ts
    """
)
_MEAN_QUERY = text(
    """
    SELECT s.start_ts, s.mean
      FROM statistics s
      JOIN statistics_meta sm ON s.metadata_id = sm.id
     WHERE sm.statistic_id = :entity_id
       AND s.start_ts >= :from_ts
       AND s.start_ts < :end_ts
       AND s.mean IS NOT NULL
     ORDER BY s.start_ts
    """
)


class RecorderError(Exception):
    """Raised when the recorder cannot be read or an entity is unusable.

    Callers log the error and leave the retained forecast unchanged rather
    than publishing a forecast built from partial history.
    """


_SENSITIVE_QUERY_KEYS = ("password", "passwd", "pwd", "secret", "token", "key")
"""Substrings of URL query parameter names whose values are treated as secrets."""


def _parse(db_url: str) -> URL | None:
    """Parse ``db_url``, returning ``None`` when it is not a valid URL."""
    try:
        return make_url(db_url)
    except (ArgumentError, ValueError):
        return None


def _is_sensitive(name: str) -> bool:
    """Return whether a URL query parameter name looks like it carries a secret."""
    return any(k in name.lower() for k in _SENSITIVE_QUERY_KEYS)


def _secrets(url: URL) -> list[str]:
    """Return every credential carried by ``url``, decoded and URL-encoded.

    Covers the URL password and the values of query parameters whose name
    looks sensitive (for example ``password=`` or ``sslpassword=``). Both the
    decoded value and its percent-encoded forms are returned, because a
    message may contain either.
    """
    raw: list[str] = []
    if url.password:
        raw.append(str(url.password))
    for name, value in url.query.items():
        if _is_sensitive(name):
            values = value if isinstance(value, tuple) else (value,)
            raw.extend(str(v) for v in values if v)
    forms: set[str] = set()
    for secret in raw:
        forms.update({secret, quote(secret, safe=""), quote_plus(secret, safe="")})
    return sorted(forms, key=len, reverse=True)


def _scrub(message: str, db_url: str) -> str:
    """Replace every credential of ``db_url`` in ``message`` with ``***``.

    When ``db_url`` cannot be parsed its credentials cannot be located, so
    only the exception type survives: the full message is withheld.
    """
    url = _parse(db_url)
    if url is None:
        return "<message withheld: database URL could not be parsed>"
    for secret in _secrets(url):
        message = message.replace(secret, "***")
    return message


def redact_db_url(db_url: str) -> str:
    """Return ``db_url`` with every credential replaced by ``***``.

    Engine and query errors are logged with their message, and a database URL
    for PostgreSQL or MariaDB carries the password, either in the URL itself
    or as a query parameter. Sensitive query values are replaced before the
    URL is rendered, so percent-encoding cannot hide them from the mask.
    Driver, host, port, database and username are kept because they make the
    message useful.

    Args:
        db_url: The configured SQLAlchemy database URL.

    Returns:
        The URL with credentials masked, or a fixed placeholder when the
        string cannot be parsed as a URL.
    """
    url = _parse(db_url)
    if url is None:
        return "<unparseable database URL>"
    masked_query = {k: ("***" if _is_sensitive(k) else v) for k, v in url.query.items()}
    rendered = url.set(query=masked_query).render_as_string(hide_password=True)
    for secret in _secrets(url):
        rendered = rendered.replace(secret, "***")
    return rendered


_URI_PATH_SAFE = "/:~._-+,=@!$&'()*;"
"""Characters left unescaped in a SQLite ``file:`` URI path.

Everything else, in particular ``%``, ``?``, ``#`` and space, is
percent-encoded so the URI names exactly the configured file and its
``mode=ro`` parameter cannot be cut off as a fragment or a second query.
"""


def _read_only_sqlite_url(url: URL) -> URL:
    """Rewrite a SQLite URL so the database is always opened read-only.

    Every SQLite driver variant is normalised to the ``file:`` URI form with
    ``uri=true`` and ``mode=ro``. SQLite ignores ``mode`` without
    ``uri=true``, and ``mode=ro`` also prevents a missing file from being
    created. ``make_url`` has already percent-decoded the path, so it is
    re-encoded here in both the plain-path and the ``file:`` form. Other query
    options (for example ``timeout``) are kept; any ``mode`` or ``uri`` value
    the user set is overridden.
    """
    database = url.database or ""
    path = database[len("file:") :] if database.startswith("file:") else database
    query = {k: v for k, v in url.query.items() if k not in ("mode", "uri")}
    query.update({"uri": "true", "mode": "ro"})
    return url.set(database=f"file:{quote(path, safe=_URI_PATH_SAFE)}", query=query)


def build_engine(db_url: str) -> Engine:
    """Create a SQLAlchemy engine for the recorder database.

    SQLite URLs, of any driver variant and with or without query options,
    are rewritten so the file is opened read-only: this helper can never
    write to the Home Assistant database. Other backends rely on
    database-side permissions.

    Args:
        db_url: SQLAlchemy database URL of the recorder database.

    Returns:
        A new engine. The caller disposes of it.

    Raises:
        RecorderError: If the URL is malformed or its driver is unavailable.
            The message is scrubbed of credentials and the original
            exception is not chained, because its text may contain them.
    """
    try:
        url = make_url(db_url)
        if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
            return create_engine(
                _read_only_sqlite_url(url), connect_args={"check_same_thread": False}
            )
        return create_engine(url)
    except (ArgumentError, ImportError, ValueError, SQLAlchemyError) as exc:
        message = _scrub(f"{type(exc).__name__}: {exc}", db_url)
        raise RecorderError(
            f"Cannot create database engine for {redact_db_url(db_url)!r}: {message}"
        ) from None


@dataclass(frozen=True)
class RecorderHistory:
    """Hourly history read from the recorder.

    Attributes:
        heat_kwh: Per heat-source energy entity, the electricity in kWh used
            during each hour, keyed by the UTC epoch second of the hour's
            start. Hours without a trustworthy delta are absent.
        tank_temp_c: Per tank temperature entity, the hourly mean in degrees
            Celsius, keyed the same way. Hours without a mean are absent.
    """

    heat_kwh: dict[str, dict[int, float]] = field(default_factory=dict)
    tank_temp_c: dict[str, dict[int, float]] = field(default_factory=dict)


def fetch_history(config: RecorderConfig, *, start_ts: int, end_ts: int) -> RecorderHistory:
    """Read hourly heat-source energy and tank temperatures for ``[start_ts, end_ts)``.

    Args:
        config: Recorder connection and entity configuration.
        start_ts: UTC epoch second of the first hour to return. Must be hour
            aligned. One extra row before it is read to seed the first energy
            delta.
        end_ts: UTC epoch second of the first hour not returned (exclusive).

    Returns:
        The hourly history of every heat source and tank temperature sensor.

    Raises:
        RecorderError: If the engine cannot be created, a query fails, an
            entity is missing from ``statistics_meta``, or an entity's unit is
            not supported.
    """
    engine = build_engine(config.db_url)
    try:
        with engine.connect() as conn:
            heat = {s.energy_entity: _read_energy(conn, s, start_ts, end_ts) for s in config.heat_sources}
            temps = {
                e.entity_id: _read_temperature(conn, e, start_ts, end_ts)
                for e in config.tank_temperature_entities
            }
    except SQLAlchemyError as exc:
        # The driver's message can echo connection parameters, so it is
        # scrubbed and the original exception is not chained into the
        # traceback the caller logs.
        message = _scrub(f"{type(exc).__name__}: {exc}", config.db_url)
        raise RecorderError(
            f"Recorder query failed on {redact_db_url(config.db_url)!r}: {message}"
        ) from None
    finally:
        engine.dispose()
    return RecorderHistory(heat_kwh=heat, tank_temp_c=temps)


def _on_the_hour(start_ts: float) -> bool:
    """Return whether a row's ``start_ts`` is a finite, exact hour boundary.

    Hourly statistics always start on the hour. A row that does not is not
    trusted: truncating it to an integer could make two rows less than an hour
    apart look adjacent.
    """
    value = float(start_ts)
    return math.isfinite(value) and value == int(value) and int(value) % _HOUR_S == 0


def _resolve_unit(conn: Connection, entity_id: str, override: str | None) -> str | None:
    """Return the configured unit override, else the unit in ``statistics_meta``.

    Raises:
        RecorderError: If the entity has no ``statistics_meta`` row, which
            means it has no long-term statistics at all.
    """
    row = conn.execute(_UNIT_QUERY, {"entity_id": entity_id}).fetchone()
    if row is None:
        raise RecorderError(
            f"Entity {entity_id} has no long-term statistics (not in statistics_meta)."
        )
    return override if override is not None else row[0]


def _energy_factor(conn: Connection, source: HeatSourceConfig) -> float:
    """Return the kWh factor of a heat source's counter unit.

    Raises:
        RecorderError: If the entity has no statistics or no energy unit.
    """
    unit = _resolve_unit(conn, source.energy_entity, source.unit)
    if unit not in _ENERGY_UNIT_TO_KWH:
        raise RecorderError(
            f"Entity {source.energy_entity} has unit {unit!r}; an energy unit "
            f"({', '.join(_ENERGY_UNIT_TO_KWH)}) is required."
        )
    return _ENERGY_UNIT_TO_KWH[unit]


def _temperature_converter(conn: Connection, entity: TankTemperatureEntityConfig) -> Callable[[float], float]:
    """Return a function converting the sensor's unit to degrees Celsius.

    Raises:
        RecorderError: If the entity has no statistics or its unit is
            neither °C nor °F.
    """
    unit = _resolve_unit(conn, entity.entity_id, entity.unit)
    if unit == "°C":
        return float
    if unit == "°F":
        return lambda value: (float(value) - 32.0) * 5.0 / 9.0
    raise RecorderError(f"Entity {entity.entity_id} has unit {unit!r}; °C or °F is required.")


def check_entities(config: RecorderConfig) -> None:
    """Verify every configured entity has long-term statistics in a supported unit.

    Run at startup, so a misconfigured entity is reported before the first
    trigger rather than only when a cycle runs.

    Args:
        config: Recorder connection and entity configuration.

    Raises:
        RecorderError: If the database cannot be read, or an entity is
            missing or has an unsupported unit. The message names the entity
            and is scrubbed of credentials.
    """
    engine = build_engine(config.db_url)
    try:
        with engine.connect() as conn:
            for source in config.heat_sources:
                _energy_factor(conn, source)
            for entity in config.tank_temperature_entities:
                _temperature_converter(conn, entity)
    except SQLAlchemyError as exc:
        message = _scrub(f"{type(exc).__name__}: {exc}", config.db_url)
        raise RecorderError(
            f"Recorder check failed on {redact_db_url(config.db_url)!r}: {message}"
        ) from None
    finally:
        engine.dispose()


def _read_energy(conn: Connection, source: HeatSourceConfig, start_ts: int, end_ts: int) -> dict[int, float]:
    """Return hourly kWh deltas of one heat source's energy counter.

    A delta for hour ``h`` is ``sum(h) - sum(h - 1h)``; Home Assistant stores
    the counter value at the end of each hour in that hour's ``sum``. The
    delta is only trusted when both rows exist and are exactly one hour apart
    and the difference is not negative (a decreasing ``sum`` indicates a
    counter reset or recorder corruption).

    Raises:
        RecorderError: If the unit is not an energy unit.
    """
    factor = _energy_factor(conn, source)
    rows = conn.execute(
        _SUM_QUERY,
        {"entity_id": source.energy_entity, "from_ts": start_ts - _HOUR_S, "end_ts": end_ts},
    ).fetchall()
    sums = [(int(r[0]), float(r[1])) for r in rows if _on_the_hour(r[0])]
    deltas: dict[int, float] = {}
    for (prev_ts, prev_sum), (ts, total) in zip(sums, sums[1:], strict=False):
        if ts - prev_ts != _HOUR_S or ts < start_ts:
            continue
        delta = total - prev_sum
        if delta < 0:
            logger.debug("Dropping negative delta %.4f for %s at %d.", delta, source.energy_entity, ts)
            continue
        deltas[ts] = delta * factor
    return deltas


def _read_temperature(
    conn: Connection, entity: TankTemperatureEntityConfig, start_ts: int, end_ts: int
) -> dict[int, float]:
    """Return hourly mean temperatures of one sensor in degrees Celsius.

    Raises:
        RecorderError: If the unit is neither degrees Celsius nor Fahrenheit.
    """
    convert = _temperature_converter(conn, entity)
    rows = conn.execute(
        _MEAN_QUERY, {"entity_id": entity.entity_id, "from_ts": start_ts, "end_ts": end_ts}
    ).fetchall()
    return {int(r[0]): convert(r[1]) for r in rows if _on_the_hour(r[0])}
