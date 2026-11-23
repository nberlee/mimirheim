"""Hot water draw estimation from hourly tank temperatures and heat input.

Turns the recorder history of one tank into a daily draw (scale) and an
hourly profile per day type (shape), and expands them into an hourly
forecast. Pure functions: no I/O, no clock reads.

The energy balance of the tank between two consecutive hourly means is::

    D(h) = Q_int(h) - S_int(h) - (E(h) - E(h-1))

with ``E(h) = capacity * T(h)`` the stored energy at the mean of hour ``h``,
``Q(h)`` the heat delivered in hour ``h`` (each source's electricity times
its COP), ``S(h) = loss * (T(h) - T_room) * 1 h`` the standby loss, and
``Q_int`` and ``S_int`` the averages of hours ``h-1`` and ``h`` (the
interval between the two means covers the second half of one hour and the
first half of the next). ``D(h)`` is the draw in that interval, attributed to
hour ``h``, a documented half-hour shift.

Scale: over a maximal run of hours where the temperature and every source's
energy are known (a segment), the sum of ``D`` telescopes to total heat in,
minus total standby loss, minus the change in stored energy, so heating that
straddles hour boundaries cancels. The daily draw is 24 times the sum of
segment draws divided by their number of intervals.

Shape: hourly samples ``D(h)`` only where hours ``h-1``, ``h`` and ``h+1`` are
known and heat-free, because a mean over a heating hour does not show the
draw cleanly. Negative samples are clipped to zero and counted.

This module does not read the recorder, publish anything, or know about
MQTT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from hot_water_draw.config import HeatSourceConfig, ModelConfig, TankConfig
from hot_water_draw.recorder import RecorderHistory

_HOUR_S = 3600
_DAY_TYPES = ("weekday", "weekend")
_MIN_SHAPE_TOTAL_KWH = 1e-3


@dataclass(frozen=True)
class DrawQuality:
    """How the estimate was obtained, published for diagnosis.

    Attributes:
        segments: Number of usable segments (runs of at least two hours).
        segment_hours: Total hours in those segments.
        samples: Shape samples used.
        clipped_samples: Shape samples that were negative and set to zero.
        cells_by_level: Count of the 48 cells (24 hours x 2 day types) per
            fallback level: ``cell``, ``pooled`` or ``flat``.
        fallback_used: True when no segment existed and the configured
            fallback daily draw was used.
        negative_total_clipped: True when the window's total draw came out
            negative and was set to zero.
    """

    segments: int
    segment_hours: int
    samples: int
    clipped_samples: int
    cells_by_level: dict[str, int] = field(default_factory=dict)
    fallback_used: bool = False
    negative_total_clipped: bool = False


@dataclass(frozen=True)
class DrawEstimate:
    """Daily draw and hourly profile of one tank.

    Attributes:
        daily_draw_kwh: Mean daily draw in kWh thermal.
        shape: Per day type (``weekday``, ``weekend``), 24 shares that sum to
            one, indexed by local hour of day.
        quality: Diagnostics of the estimate.
    """

    daily_draw_kwh: float
    shape: dict[str, list[float]]
    quality: DrawQuality


def estimate(
    history: RecorderHistory,
    sources: list[HeatSourceConfig],
    tank: TankConfig,
    model: ModelConfig,
    *,
    window_end_ts: int,
) -> DrawEstimate:
    """Estimate the daily draw and its hourly shape from recorder history.

    Args:
        history: Hourly heat-source energy and tank temperature means.
        sources: Heat sources, for their COP.
        tank: Tank capacity and standby loss.
        model: Estimation settings.
        window_end_ts: UTC epoch second at which the lookback window ends
            (exclusive, hour aligned). Sample ages are measured from here.

    Returns:
        The estimate. With no usable segment, the fallback daily draw with a
        flat shape.
    """
    start_ts = window_end_ts - model.lookback_days * 24 * _HOUR_S
    hours = list(range(start_ts, window_end_ts, _HOUR_S))
    cap = tank.capacity_kwh_per_k()
    loss = tank.loss_kw_per_k()

    temp: dict[int, float] = {}
    for h in hours:
        values = [series.get(h) for series in history.tank_temp_c.values()]
        if values and all(v is not None for v in values):
            temp[h] = sum(values) / len(values)  # type: ignore[arg-type]

    heat: dict[int, float] = {}
    heating: dict[int, bool] = {}
    for h in hours:
        deltas = [history.heat_kwh.get(s.energy_entity, {}).get(h) for s in sources]
        if all(d is not None for d in deltas):
            heat[h] = sum(d * s.cop for d, s in zip(deltas, sources, strict=True))  # type: ignore[operator]
            heating[h] = any(d > model.heating_threshold_kwh for d in deltas)  # type: ignore[operator]

    def usable(h: int) -> bool:
        return h in temp and h in heat

    def draw(h: int) -> float:
        prev = h - _HOUR_S
        q_int = (heat[prev] + heat[h]) / 2.0
        s_int = loss * ((temp[prev] + temp[h]) / 2.0 - tank.ambient_temp_c)
        return q_int - s_int - cap * (temp[h] - temp[prev])

    # Scale over segments.
    segments, segment_hours, intervals, total = 0, 0, 0, 0.0
    run: list[int] = []
    for h in [*hours, None]:
        if h is not None and usable(h):
            run.append(h)
            continue
        if len(run) >= 2:
            segments += 1
            segment_hours += len(run)
            intervals += len(run) - 1
            total += sum(draw(x) for x in run[1:])
        run = []

    if intervals == 0:
        flat = {d: [1.0 / 24] * 24 for d in _DAY_TYPES}
        quality = DrawQuality(0, 0, 0, 0, {"cell": 0, "pooled": 0, "flat": 48}, fallback_used=True)
        return DrawEstimate(model.fallback_daily_draw_kwh, flat, quality)

    negative_total = total < 0.0
    daily = 24.0 * max(total, 0.0) / intervals

    # Shape samples.
    tz = ZoneInfo(model.timezone)
    cells: dict[tuple[str, int], list[tuple[float, float]]] = {}
    samples = clipped = 0
    for h in hours:
        neighbours = (h - _HOUR_S, h, h + _HOUR_S)
        if not all(usable(x) for x in neighbours) or any(heating[x] for x in neighbours):
            continue
        value = draw(h)
        if value < 0.0:
            clipped += 1
            value = 0.0
        samples += 1
        local = datetime.fromtimestamp(h, UTC).astimezone(tz)
        day_type = "weekend" if local.weekday() >= 5 else "weekday"
        age_days = (window_end_ts - h) / (24 * _HOUR_S)
        cells.setdefault((day_type, local.hour), []).append((model.lookback_decay**age_days, value))

    shape, levels = _shape(cells, model.min_samples_per_hour)
    quality = DrawQuality(
        segments, segment_hours, samples, clipped, levels, negative_total_clipped=negative_total
    )
    return DrawEstimate(daily, shape, quality)


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    weight = sum(w for w, _ in items)
    return sum(w * v for w, v in items) / weight if weight > 0 else 0.0


def _shape(
    cells: dict[tuple[str, int], list[tuple[float, float]]], min_samples: int
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Return the normalised shape per day type and the count per fallback level.

    A cell uses its own weighted mean when it has ``min_samples`` samples,
    else its hour pooled over both day types; if even the pooled hour lacks
    samples, the whole shape is flat.
    """
    values: dict[str, list[float]] = {d: [0.0] * 24 for d in _DAY_TYPES}
    levels = {"cell": 0, "pooled": 0, "flat": 0}
    cell_level: dict[tuple[str, int], str] = {}
    for hour in range(24):
        pooled = [s for d in _DAY_TYPES for s in cells.get((d, hour), [])]
        for d in _DAY_TYPES:
            own = cells.get((d, hour), [])
            if len(own) >= min_samples:
                values[d][hour] = _weighted_mean(own)
                levels["cell"] += 1
                cell_level[(d, hour)] = "cell"
            elif len(pooled) >= min_samples:
                values[d][hour] = _weighted_mean(pooled)
                levels["pooled"] += 1
                cell_level[(d, hour)] = "pooled"
            else:
                flat = {d: [1.0 / 24] * 24 for d in _DAY_TYPES}
                return flat, {"cell": 0, "pooled": 0, "flat": 48}
    shape: dict[str, list[float]] = {}
    for d in _DAY_TYPES:
        total = sum(values[d])
        # Policy: a day type whose 24 hourly means sum to at most 1 Wh has no
        # usable shape. A draw-free history otherwise normalises rounding
        # noise (around 1e-16 kWh) into an arbitrary profile. The daily scale
        # is unaffected; only the profile of that day type becomes flat.
        if total > _MIN_SHAPE_TOTAL_KWH:
            shape[d] = [v / total for v in values[d]]
            continue
        # A day type with no draw at all has nothing to normalise; it falls
        # back to flat, and its 24 cells are reported as such.
        shape[d] = [1.0 / 24] * 24
        for hour in range(24):
            levels[cell_level[(d, hour)]] -= 1
        levels["flat"] += 24
    return shape, levels


def forecast(result: DrawEstimate, model: ModelConfig, *, now_ts: int) -> list[tuple[int, float]]:
    """Expand an estimate into an hourly forecast.

    Each hour's average draw power (kW) is the daily draw times the share of
    its local hour for its local day type. On a daylight saving day the
    23 or 25 hours each take their own local hour's share, so that day's total
    deviates from the daily draw by one hour's share.

    Args:
        result: The estimate.
        model: For the timezone and horizon.
        now_ts: Current UTC epoch second. The forecast starts at the start of
            the hour containing it, so the current hour is covered.

    Returns:
        ``(hour_start_ts, kw)`` pairs, one per hour, ``horizon_hours + 1``
        long: the last one starts at the horizon end and marks it, like the
        other helpers' forecasts.
    """
    tz = ZoneInfo(model.timezone)
    first = now_ts // _HOUR_S * _HOUR_S
    points: list[tuple[int, float]] = []
    for i in range(model.horizon_hours + 1):
        ts = first + i * _HOUR_S
        local = datetime.fromtimestamp(ts, UTC).astimezone(tz)
        day_type = "weekend" if local.weekday() >= 5 else "weekday"
        points.append((ts, result.daily_draw_kwh * result.shape[day_type][local.hour]))
    return points
