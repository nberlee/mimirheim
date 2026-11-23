"""Unit tests for hot_water_draw.recorder.

Uses a real, file-backed SQLite database with the minimal subset of the Home
Assistant recorder schema the module reads (``statistics_meta`` and
``statistics``). No mocking of SQLAlchemy.
"""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from hot_water_draw.config import RecorderConfig
from hot_water_draw.recorder import (
    RecorderError,
    build_engine,
    fetch_history,
    redact_db_url,
)

H = 3600
T0 = 1_767_225_600  # 2026-01-01T00:00:00Z, hour aligned


def _make_db(
    tmp_path: Path,
    meta: list[tuple[int, str, str | None]],
    rows: list[tuple[int, float, float | None, float | None]],
) -> str:
    """Create a recorder database and return its SQLAlchemy URL.

    Args:
        tmp_path: pytest temporary directory.
        meta: ``(metadata_id, statistic_id, unit)`` rows.
        rows: ``(metadata_id, start_ts, mean, sum)`` rows.
    """
    path = tmp_path / "home-assistant_v2.db"
    # A creator avoids URL parsing, so paths containing '?', '#' or '%' work.
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(path))
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE statistics_meta (id INTEGER PRIMARY KEY, "
                "statistic_id TEXT, unit_of_measurement TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE statistics (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "metadata_id INTEGER, start_ts REAL, mean REAL, sum REAL)"
            )
        )
        for mid, sid, unit in meta:
            conn.execute(
                text("INSERT INTO statistics_meta VALUES (:i, :s, :u)"),
                {"i": mid, "s": sid, "u": unit},
            )
        for mid, ts, mean, total in rows:
            conn.execute(
                text(
                    "INSERT INTO statistics (metadata_id, start_ts, mean, sum) "
                    "VALUES (:m, :t, :mean, :sum)"
                ),
                {"m": mid, "t": ts, "mean": mean, "sum": total},
            )
    engine.dispose()
    return f"sqlite:///{path}"


def _cfg(db_url: str, *, sum_unit: str | None = None, temp_unit: str | None = None) -> RecorderConfig:
    return RecorderConfig(
        db_url=db_url,
        heat_sources=[{"energy_entity": "sensor.e", "cop": 1.0, "unit": sum_unit}],
        tank_temperature_entities=[{"entity_id": "sensor.t", "unit": temp_unit}],
    )


def _counter(mid: int, start: int, sums: list[float]) -> list[tuple[int, float, None, float]]:
    """Return hourly cumulative-sum rows starting at ``start``."""
    return [(mid, float(start + i * H), None, s) for i, s in enumerate(sums)]


class TestEnergy:
    def test_hourly_deltas_with_seed_row(self, tmp_path: Path) -> None:
        # The row at T0 - H is the seed: it is used for the first delta only.
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [10.0, 11.0, 13.0, 16.0]),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 3 * H)
        assert hist.heat_kwh["sensor.e"] == {T0: 1.0, T0 + H: 2.0, T0 + 2 * H: 3.0}

    def test_end_ts_is_exclusive(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [10.0, 11.0, 13.0, 16.0]),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 2 * H)
        assert set(hist.heat_kwh["sensor.e"]) == {T0, T0 + H}

    def test_gap_row_drops_the_hour_after_the_gap(self, tmp_path: Path) -> None:
        rows = _counter(1, T0 - H, [10.0, 11.0]) + [(1, float(T0 + 2 * H), None, 15.0)]
        url = _make_db(tmp_path, [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")], rows)
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 3 * H)
        # T0 is valid; T0 + 2H follows a missing hour, so no delta is trusted.
        assert hist.heat_kwh["sensor.e"] == {T0: 1.0}

    def test_negative_delta_drops_the_hour(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [10.0, 11.0, 5.0, 6.0]),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 3 * H)
        assert hist.heat_kwh["sensor.e"] == {T0: 1.0, T0 + 2 * H: 1.0}

    def test_zero_delta_is_kept(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [10.0, 10.0]),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + H)
        assert hist.heat_kwh["sensor.e"] == {T0: 0.0}

    @pytest.mark.parametrize("unit,factor", [("Wh", 1e-3), ("kWh", 1.0), ("MWh", 1e3)])
    def test_units_converted_to_kwh(self, tmp_path: Path, unit: str, factor: float) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", unit), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [0.0, 2.0]),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + H)
        assert hist.heat_kwh["sensor.e"][T0] == pytest.approx(2.0 * factor)

    def test_unit_override_wins(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [0.0, 2000.0]),
        )
        hist = fetch_history(_cfg(url, sum_unit="Wh"), start_ts=T0, end_ts=T0 + H)
        assert hist.heat_kwh["sensor.e"][T0] == pytest.approx(2.0)

    @pytest.mark.parametrize("unit", ["W", "kW", "°C", None])
    def test_non_energy_unit_rejected(self, tmp_path: Path, unit: str | None) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", unit), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, [0.0, 1.0]),
        )
        with pytest.raises(RecorderError, match="sensor.e"):
            fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + H)

    def test_entity_missing_from_metadata_rejected(self, tmp_path: Path) -> None:
        url = _make_db(tmp_path, [(2, "sensor.t", "°C")], [])
        with pytest.raises(RecorderError, match="sensor.e"):
            fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + H)

    def test_rare_large_hours_are_kept(self, tmp_path: Path) -> None:
        """No outlier filter: a resistive element's rare large hour is real heat input."""
        sums = [0.0]
        for i in range(30):
            sums.append(sums[-1] + (6.0 if i == 20 else 0.3))
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            _counter(1, T0 - H, sums),
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 30 * H)
        assert hist.heat_kwh["sensor.e"][T0 + 20 * H] == pytest.approx(6.0)
        assert len(hist.heat_kwh["sensor.e"]) == 30


class TestTemperature:
    def test_mean_read_and_negative_values_kept(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            [(2, float(T0), -4.5, None), (2, float(T0 + H), 1.25, None)],
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 2 * H)
        assert hist.tank_temp_c["sensor.t"] == {T0: -4.5, T0 + H: 1.25}

    def test_null_mean_skipped(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            [(2, float(T0), None, None), (2, float(T0 + H), 3.0, None)],
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 2 * H)
        assert hist.tank_temp_c["sensor.t"] == {T0 + H: 3.0}

    def test_fahrenheit_normalised_to_celsius(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°F")],
            [(2, float(T0), 23.0, None), (2, float(T0 + H), 50.0, None)],
        )
        hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 2 * H)
        assert hist.tank_temp_c["sensor.t"][T0] == pytest.approx(-5.0)
        assert hist.tank_temp_c["sensor.t"][T0 + H] == pytest.approx(10.0)

    def test_temperature_unit_override_wins(self, tmp_path: Path) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")],
            [(2, float(T0), 50.0, None)],
        )
        hist = fetch_history(_cfg(url, temp_unit="°F"), start_ts=T0, end_ts=T0 + H)
        assert hist.tank_temp_c["sensor.t"][T0] == pytest.approx(10.0)

    @pytest.mark.parametrize("unit", ["K", "kWh", None])
    def test_other_units_rejected(self, tmp_path: Path, unit: str | None) -> None:
        url = _make_db(
            tmp_path,
            [(1, "sensor.e", "kWh"), (2, "sensor.t", unit)],
            [(2, float(T0), 5.0, None)],
        )
        with pytest.raises(RecorderError, match="sensor.t"):
            fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + H)


class TestEngineAndErrors:
    def _write_db(self, tmp_path: Path) -> Path:
        url = _make_db(tmp_path, [(1, "sensor.e", "kWh")], [])
        return Path(url[len("sqlite:///") :])

    @pytest.mark.parametrize(
        "template",
        [
            "sqlite:///{path}",
            "sqlite+pysqlite:///{path}",
            "sqlite:///{path}?timeout=30",
            "sqlite:///{path}?mode=ro",
            "sqlite:///file:{path}?uri=true&mode=rw",
            "sqlite:///file:{path}?uri=true",
        ],
    )
    def test_sqlite_variants_opened_read_only(self, tmp_path: Path, template: str) -> None:
        path = self._write_db(tmp_path)
        engine = build_engine(template.format(path=path))
        try:
            with engine.connect() as conn:
                assert conn.execute(text("SELECT COUNT(*) FROM statistics_meta")).scalar() == 1
                with pytest.raises(Exception, match="readonly"):
                    conn.execute(text("INSERT INTO statistics_meta VALUES (9, 'x', 'kWh')"))
        finally:
            engine.dispose()

    def test_relative_sqlite_path_with_query(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_db(tmp_path)
        monkeypatch.chdir(tmp_path)
        engine = build_engine("sqlite:///home-assistant_v2.db?timeout=30")
        try:
            with engine.connect() as conn:
                assert conn.execute(text("SELECT COUNT(*) FROM statistics_meta")).scalar() == 1
        finally:
            engine.dispose()

    def test_sqlite_path_with_space(self, tmp_path: Path) -> None:
        sub = tmp_path / "dir with space"
        sub.mkdir()
        path = self._write_db(sub)
        engine = build_engine(f"sqlite:///{path}")
        try:
            with engine.connect() as conn:
                assert conn.execute(text("SELECT COUNT(*) FROM statistics_meta")).scalar() == 1
        finally:
            engine.dispose()

    @pytest.mark.parametrize("special", ["#", "?", "%"])
    def test_encoded_uri_paths_stay_read_only(self, tmp_path: Path, special: str) -> None:
        sub = tmp_path / f"ha{special}archive"
        sub.mkdir()
        path = self._write_db(sub)
        encoded = str(path).replace("%", "%25").replace("#", "%23").replace("?", "%3F")
        before = {p.name for p in tmp_path.iterdir()}
        engine = build_engine(f"sqlite:///file:{encoded}?uri=true")
        try:
            with engine.connect() as conn:
                assert conn.execute(text("SELECT COUNT(*) FROM statistics_meta")).scalar() == 1
                with pytest.raises(Exception, match="readonly"):
                    conn.execute(text("INSERT INTO statistics_meta VALUES (9, 'x', 'kWh')"))
        finally:
            engine.dispose()
        assert {p.name for p in tmp_path.iterdir()} == before

    def test_missing_sqlite_file_is_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.db"
        with pytest.raises(RecorderError):
            fetch_history(_cfg(f"sqlite:///{missing}"), start_ts=T0, end_ts=T0 + H)
        assert not missing.exists()

    def test_missing_table_raises_recorder_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.db"
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE unrelated (id INTEGER)"))
        engine.dispose()
        with pytest.raises(RecorderError):
            fetch_history(_cfg(f"sqlite:///{path}"), start_ts=T0, end_ts=T0 + H)

    @pytest.mark.parametrize(
        "db_url",
        [
            "nosuchdriver://user:s3cret@host/db",
            "postgresql+psycopg2://user:s3cret@127.0.0.1:1/db",
            "postgresql+psycopg2://user@127.0.0.1:1/db?password=s3cret",
            "postgresql+psycopg2://user@127.0.0.1:1/db?password=s3cret%40x",
            "postgresql+psycopg2://user:s3cret%40x@127.0.0.1:1/db",
            "postgresql://u:s3cret@host:bad/db",
        ],
    )
    def test_errors_never_expose_secrets(self, db_url: str) -> None:
        with pytest.raises(RecorderError) as excinfo:
            fetch_history(_cfg(db_url), start_ts=T0, end_ts=T0 + H)
        rendered = "".join(traceback.format_exception(excinfo.value))
        assert "s3cret" not in str(excinfo.value)
        assert "s3cret" not in rendered

    @pytest.mark.parametrize(
        "db_url",
        [
            "mysql+pymysql://u:s3cret@h/db",
            "postgresql://u@h/db?password=s3cret",
            "postgresql://u@h/db?sslpassword=s3cret&application_name=x",
            "postgresql://u@h/db?password=s3cret%40x",
            "postgresql://u:s3cret%40x@h/db",
        ],
    )
    def test_redact_db_url_masks_secrets(self, db_url: str) -> None:
        redacted = redact_db_url(db_url)
        assert "s3cret" not in redacted
        assert "h/db" in redacted

    @pytest.mark.parametrize("db_url", ["::not a url::", "postgresql://u:s3cret@host:bad/db"])
    def test_redact_db_url_unparseable(self, db_url: str) -> None:
        assert redact_db_url(db_url) == "<unparseable database URL>"


def test_every_source_and_tank_sensor_is_read(tmp_path: Path) -> None:
    url = _make_db(
        tmp_path,
        [(1, "sensor.comp", "kWh"), (2, "sensor.elem", "Wh"), (3, "sensor.top", "°C"), (4, "sensor.low", "°F")],
        _counter(1, T0 - H, [5.0, 5.4]) + _counter(2, T0 - H, [0.0, 1500.0])
        + [(3, float(T0), 52.0, None), (4, float(T0), 104.0, None)],
    )
    cfg = RecorderConfig(
        db_url=url,
        heat_sources=[{"energy_entity": "sensor.comp", "cop": 2.5}, {"energy_entity": "sensor.elem", "cop": 1.0}],
        tank_temperature_entities=[{"entity_id": "sensor.top"}, {"entity_id": "sensor.low"}],
    )
    hist = fetch_history(cfg, start_ts=T0, end_ts=T0 + H)
    assert hist.heat_kwh == {"sensor.comp": {T0: pytest.approx(0.4)}, "sensor.elem": {T0: pytest.approx(1.5)}}
    assert hist.tank_temp_c == {"sensor.top": {T0: 52.0}, "sensor.low": {T0: pytest.approx(40.0)}}


def test_misaligned_rows_are_not_treated_as_adjacent(tmp_path: Path) -> None:
    """Rows 3599.2 s apart, or not on an hour, give no delta and no mean."""
    rows = [
        (1, float(T0 - H) + 0.9, None, 10.0),
        (1, float(T0) + 0.1, None, 11.0),
        (1, float(T0 + H), None, 12.0),
        (1, float(T0 + 2 * H), None, 13.5),
        (2, float(T0) + 0.5, 50.0, None),
        (2, float(T0 + H), 51.0, None),
    ]
    url = _make_db(tmp_path, [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")], rows)
    hist = fetch_history(_cfg(url), start_ts=T0, end_ts=T0 + 3 * H)
    assert hist.heat_kwh["sensor.e"] == {T0 + 2 * H: pytest.approx(1.5)}
    assert hist.tank_temp_c["sensor.t"] == {T0 + H: 51.0}


class TestEligibility:
    def test_eligible_entities_pass(self, tmp_path: Path) -> None:
        from hot_water_draw.recorder import check_entities

        url = _make_db(tmp_path, [(1, "sensor.e", "kWh"), (2, "sensor.t", "°C")], [])
        check_entities(_cfg(url))

    @pytest.mark.parametrize(
        "meta,match",
        [
            ([(2, "sensor.t", "°C")], "sensor.e"),
            ([(1, "sensor.e", "kWh")], "sensor.t"),
            ([(1, "sensor.e", "W"), (2, "sensor.t", "°C")], "sensor.e"),
            ([(1, "sensor.e", "kWh"), (2, "sensor.t", "%")], "sensor.t"),
        ],
    )
    def test_missing_or_wrong_unit_entities_fail(self, tmp_path: Path, meta: list, match: str) -> None:
        from hot_water_draw.recorder import check_entities

        url = _make_db(tmp_path, meta, [])
        with pytest.raises(RecorderError, match=match):
            check_entities(_cfg(url))
