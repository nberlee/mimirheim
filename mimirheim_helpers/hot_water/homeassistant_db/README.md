# hot_water_draw: hot water draw forecast

Estimates how much hot water a household draws per day and at which hours from
the Home Assistant recorder, and publishes an hourly draw forecast for a
mimirheim thermal boiler.

## Contents

1. [Purpose](#1-purpose)
2. [How it works](#2-how-it-works)
3. [Configuration](#3-configuration)
4. [Output format](#4-output-format)
5. [Running](#5-running)
6. [Fault tolerance](#6-fault-tolerance)
7. [Scheduling](#7-scheduling)
8. [Prerequisites](#8-prerequisites)

---

## 1. Purpose

A mimirheim thermal boiler in the standby loss mode (`standby_loss_w` or
`standby_loss_kw_per_k` with `ambient_temp_c`) and `use_draw_forecast: true`
subtracts a forecast hot water draw per step. This helper produces that
forecast from what the tank actually did: its temperature and the energy its
heat sources used. mimirheim then schedules heating ahead of the expected
draws under the selected strategy, with any dip below `min_temp_c`
minimised first.

---

## 2. How it works

### Energy balance

Per hour the recorder provides each tank temperature sensor's mean and each
heat source's energy. Between the means of two consecutive hours the tank
gains the heat delivered (each source's electricity times its COP), loses its
standby loss (`loss * (T - T_room)`), and changes its stored energy
(`capacity * T`). Whatever is missing is the draw:

```
draw = heat in - standby loss - change in stored energy
```

The interval between two hourly means spans the second half of one hour and
the first half of the next, so heat and loss are taken as the average of both
hours, and the draw is attributed to the later hour (a half-hour shift).

### Daily draw

Over each run of at least two consecutive hours where every sensor and every
heat source has a value (a segment; single usable hours are discarded), the
draws of its intervals add up to total heat in, minus total standby loss,
minus the change in stored energy. Heating that straddles an hour boundary
cancels out. A segment of `N` hours has `N - 1` intervals, and the daily draw
is `24 * max(total_draw, 0) / sum(N - 1)` over all segments. A negative total
(for example a COP set too low) is clipped to zero and flagged.

### Hourly profile

Only hours where that hour and both neighbours had no heating give a clean
view of the draw, so only those are samples. They are grouped by local hour of
day and by weekday or weekend, optionally weighted towards recent days
(`lookback_decay`). An hour with too few samples (`min_samples_per_hour`) uses
the same hour over both day types; if even that is short, the whole profile is
flat. Each day type's profile is normalised to sum to one; a day type with no
draw at all (at most 1 Wh) is flat.

### Forecast

Each hour's draw power is the daily draw times its local hour's share. On a
daylight saving day the 23 or 25 hours each take their own local hour's
share. Without any usable history, `fallback_daily_draw_kwh` is spread flat.

---

## 3. Configuration

See `mimirheim_helpers/examples/hot-water-draw.yaml` for a commented example.

| Field | Default | Description |
|---|---|---|
| `mimir_thermal_boiler_name` | `dhw` | mimirheim `thermal_boilers` device name; the output topic derives from it |
| `homeassistant.db_url` | required | SQLAlchemy URL of the recorder database (SQLite is opened read-only) |
| `homeassistant.tank_temperature_entities` | required | Tank temperature sensors, averaged per hour |
| `homeassistant.heat_sources` | required | `{energy_entity, cop, unit?}` per heat source |
| `tank.volume_liters` | required | Water volume |
| `tank.standby_loss_w` / `standby_loss_kw_per_k` | one required | Exactly one: datasheet standing loss (W at 45 K) or kW/K |
| `tank.ambient_temp_c` | required | Room temperature around the tank |
| `model.timezone` | `Europe/Amsterdam` | IANA timezone for hour of day and weekday/weekend |
| `model.lookback_days` | 14 | Days of history (7 to 90) |
| `model.lookback_decay` | 1.0 | Per-day weight of profile samples |
| `model.min_samples_per_hour` | 3 | Samples an hour needs for its own value |
| `model.heating_threshold_kwh` | 0.01 | Hourly energy above which a source counts as heating |
| `model.fallback_daily_draw_kwh` | 6.0 | Daily draw (kWh of heat) without usable history |
| `model.horizon_hours` | 168 | Hours of forecast |

The MQTT fields (`mqtt`, `trigger_topic`, `output_topic`, `signal_mimir`,
`mimir_trigger_topic`, `stats_topic`, `quality_topic`, `ha_discovery`) follow
the other helpers.

---

## 4. Output format

Retained on `output_topic` (default
`{prefix}/input/thermal_boiler/{mimir_thermal_boiler_name}/draw_forecast`):
`horizon_hours + 1` hourly steps from the start of the current hour, the last
marking the horizon end. `kw` is the average draw in kW of heat over the hour.

```json
[
  {"ts": "2026-01-20T12:00:00+00:00", "kw": 0.0812},
  {"ts": "2026-01-20T13:00:00+00:00", "kw": 0.0645}
]
```

With `quality_topic` set, a retained record of the estimate follows:

```json
{"ts": "2026-01-20T12:07:00Z", "daily_draw_kwh": 5.82, "segments": 3,
 "segment_hours": 330, "samples": 214, "clipped_samples": 9,
 "cells_by_level": {"cell": 40, "pooled": 8, "flat": 0},
 "fallback_used": false, "negative_total_clipped": false}
```

---

## 5. Running

```bash
uv run python -m hot_water_draw --config hot-water-draw.yaml
```

In the container, set `ENABLE_HOT_WATER_DRAW=true` and place the file at
`/config/hot-water-draw.yaml`; in the Home Assistant add-on, enable
"Enable Hot water draw".

---

## 6. Fault tolerance

- At startup every configured entity is checked for long-term statistics in a
  supported unit. A problem is logged as an error naming the entity; the
  helper keeps running so the configuration can be corrected in the editor.
- A recorder failure during a cycle publishes no forecast, solve trigger or
  quality record, so mimirheim keeps the last retained forecast. The shared
  daemon still publishes the failed cycle's run statistics when `stats_topic`
  is set.
- The quality record is published after the forecast and is best effort.

---

## 7. Scheduling

Draw habits change slowly. Trigger once a day, for example from the scheduler:

```yaml
- "55 23 * * *": mimir/input/tools/hot_water/trigger
```

---

## 8. Prerequisites

- Tank temperature sensors with long-term statistics (`state_class:
  measurement`).
- An energy counter (`state_class: total` or `total_increasing`) per heat
  source. A single combined counter with an average COP works too, but
  separate counters per source give a better balance.
- History: the fallback daily draw is used only while no segment of at least
  two consecutive usable hours exists. Shorter history already gives a daily
  estimate; until each hour of day has `min_samples_per_hour` clean samples
  (pooled over weekday and weekend if needed), the profile is flat.
