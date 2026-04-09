# config-editor — Agent Instructions

This tool is **not independent of mimirheim**. Unlike the other helpers, it
imports directly from `mimirheim.config.schema` and from every helper config
module to generate the JSON schemas served by its API. It is packaged as part
of the root mimirheim wheel and has no separate `pyproject.toml`.

---

## Critical architectural distinction

config-editor is not an input tool. It has no MQTT connections, no trigger
topics, and no daemon cycle. It is a web interface that reads and writes the
YAML configuration files in `/config`. It exposes mimirheim's and all helpers'
Pydantic schemas through a JSON API so the front end can render a validated
editing form.

Because it imports from the mimirheim and helper config modules, it cannot be
run in isolation from the main package environment.

---

## Environment setup

config-editor runs inside the root mimirheim virtual environment. All commands
must be run from the repository root, not from this directory.

```bash
cd /path/to/hioo                             # repository root

uv sync --all-extras                         # install all dependencies including config-editor
uv run pytest mimirheim_helpers/config_editor/tests   # run tests for this module only
uv run pytest                                # run the full test suite
uv run python -m config_editor --config config-editor.yaml   # run the editor
```

---

## Source of truth

Before writing any code, read:
- `README.md` in this directory — external behaviour, HTTP API contract, configuration schema.
- `IMPLEMENTATION_DETAILS.md` in the repo root — Pydantic conventions, docstring format, code standards.

---

## Code standards

Apply all mimirheim code standards from the root `AGENTS.md` to this module
without exception:

- All public functions and methods must have complete type annotations.
- All Pydantic models must set `model_config = ConfigDict(extra="forbid")`.
- Never use a bare `except:` or `except Exception:` without logging with full traceback.
- Google-style docstrings on all public classes and functions.
- Module-level docstring on every module.
- No emoticons in code, comments, or documentation.

---

## Project structure

```
mimirheim_helpers/config_editor/
  README.md                   # external specification (authoritative)
  AGENTS.md                   # this file
  config_editor/
    __init__.py
    __main__.py               # entry point: config load, server start, signal handling
    config.py                 # ConfigEditorConfig Pydantic model; load_config()
    server.py                 # ConfigEditorServer: HTTP server and all API handlers
    static/
      index.html
      app.js
      style.css
  tests/
    conftest.py
    unit/
      test_config.py          # ConfigEditorConfig schema and load_config() tests
```

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the single-page editor frontend |
| `GET` | `/static/<file>` | Serve frontend static assets |
| `GET` | `/api/schema` | MimirheimConfig JSON Schema (cached at startup) |
| `GET` | `/api/config` | Current `mimirheim.yaml` as a parsed dict |
| `POST` | `/api/config` | Validate and write `mimirheim.yaml` |
| `GET` | `/api/helper-configs` | Enabled state and config dict for every helper |
| `GET` | `/api/helper-schemas` | JSON Schema for every helper config model |
| `POST` | `/api/helper-config/<filename>` | Enable (write) or disable (delete) a helper config file |

All JSON API responses use `Content-Type: application/json`. Write operations
use atomic temp-file + `os.replace` to prevent partial writes.

---

## IP allowlist

When running as a HA add-on, the `CONFIG_EDITOR_ALLOWED_IP` environment variable
is set by `container/etc/cont-init.d/00-options-env.sh` to the container's
default gateway IP (the ingress proxy). `load_config()` reads this variable and
sets `cfg.allowed_ip`; the server then rejects non-matching IPs with HTTP 403.

When the variable is absent (plain Docker), `allowed_ip` is `None` and no IP
restriction applies.

---

## Testing approach

Tests for this module live in `mimirheim_helpers/config_editor/tests/`. They are
discovered and run by the root `pytest` configuration.

- Config schema tests (`test_config.py`): verify `ConfigEditorConfig` defaults,
  field bounds, `extra="forbid"`, and the `CONFIG_EDITOR_ALLOWED_IP` env override.
- Server and CRUD tests (`tests/unit/test_config_editor_server.py` and
  `tests/unit/test_config_editor_crud_generic.py` in the root `tests/` directory):
  verify API behaviour in-process without a live socket.

Write the config test first (TDD). Server behaviour is tested in the root test
suite. Integration tests (live socket, full HTTP round-trip) require no MQTT
broker but do require a running `ConfigEditorServer`; mark them with
`@pytest.mark.integration`.
