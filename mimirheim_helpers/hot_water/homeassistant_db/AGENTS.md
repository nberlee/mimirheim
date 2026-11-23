# hot_water_draw: Agent Instructions

This tool is a helper package inside the mimirheim repository. It ships in the
single `mimirheim` wheel, alongside the solver and every other helper, and it
runs as its own daemon process communicating over MQTT.

---

## Dependencies

There is one `pyproject.toml`, at the repo root. This tool has no
`pyproject.toml` of its own and must not be given one.

Runtime dependencies belong in the root `pyproject.toml`, under this tool's
extra `hot-water-draw` (SQLAlchemy, tzdata), which is also part of the
`helpers` meta-extra. The optional database drivers are in
`hot-water-draw-postgres` and `hot-water-draw-mysql`.

The only helper package this tool imports is `helper_common`. The recorder
access in `recorder.py` is a local implementation for that reason, even though
other helpers read the same database.

---

## Environment

Run every command from the repo root:

```bash
uv sync --all-extras
uv run pytest mimirheim_helpers/hot_water/homeassistant_db/tests
uv run ruff check .                           # must be clean before a change is done
uv run python -m hot_water_draw --config config.yaml
```

The module path is `hot_water_draw`, published at the top level of the wheel by
`[tool.hatch.build.targets.wheel]`; the container's s6 service
`hot-water-draw` invokes it the same way.

---

## Source of truth

Before writing any code, read:

- `README.md` in this directory: external behaviour, estimation method,
  configuration, output format, fault tolerance.
- `IMPLEMENTATION_DETAILS.md` in the repo root: architectural conventions.
- `mimirheim_helpers/common/AGENTS.md`: the shared daemon and its contract.

---

## Code standards

Apply all mimirheim code standards from the root `AGENTS.md`:

- Complete type annotations on public functions and methods.
- `model_config = ConfigDict(extra="forbid")` on every Pydantic model.
- No bare `except:` or `except Exception:`; catch specific exceptions.
- Google-style docstrings and a module docstring stating scope.
- No emoticons.

Module-specific rules:

- `estimator.py` is pure: no I/O, no logging of data, no wall-clock reads.
  Every input is passed in, keyed by the UTC epoch second of an hour's start.
- A missing hourly value means "no value", never zero. Do not zero-fill.
- `recorder.py` must keep SQLite read-only for every URL form, must never put
  database credentials in an exception message or its chain, and accepts only
  rows that start exactly on the hour. It applies no outlier filter: a heat
  source legitimately has rare, much larger hours.
- The forecast is published before the optional quality record; a failure
  before publishing must publish no forecast, solve trigger or quality record
  (the shared daemon still publishes run statistics).

---

## Project structure

```
mimirheim_helpers/hot_water/homeassistant_db/
  README.md            # external specification (authoritative)
  AGENTS.md            # this file
  hot_water_draw/
    __init__.py
    __main__.py        # daemon: cycle orchestration, startup check, Config Service wiring
    config.py          # Pydantic config schema
    formspec.py        # Config Service FormSpec (ADR-0002, 0006, 0007)
    recorder.py        # HA recorder statistics: energy deltas, temperature means
    estimator.py       # energy balance, daily draw, hourly shape, forecast
    publisher.py       # MQTT publishing of forecast and quality record
  tests/
    unit/
      test_config.py
      test_formspec.py
      test_recorder.py
      test_estimator.py
      test_publisher.py
      test_cycle.py
      test_config_owner_wiring.py
```

---

## MQTT interface

| Direction | Topic | Description |
|-----------|-------|-------------|
| Subscribes | `trigger_topic` | A message fires one cycle (`{prefix}/input/tools/hot_water/trigger` by default) |
| Publishes | `output_topic` | Retained draw forecast, `horizon_hours + 1` hourly steps |
| Publishes | `mimir_trigger_topic` | Empty trigger after the forecast, if `signal_mimir` |
| Publishes | `quality_topic` | Retained estimate diagnostics after a successful forecast, if set |
| Publishes | `stats_topic` | Retained run statistics from the shared daemon, if set |

The tool never imports from `mimirheim/` and never calls `build_and_solve()`.
