"""Unit tests for hot_water_draw.estimator.

Histories are built synthetically so the true draw is known exactly. The
standby loss over an interval between two hourly means is taken
trapezoidally, so a tank cooling only by standby follows the exact recursion
``T(h) = (T(h-1) * (1 - k/2) + k * T_room) / (1 + k/2)`` with
``k = loss_kw_per_k * 1 h / capacity``; such a tank must give zero draw.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from hot_water_draw.config import HeatSourceConfig, ModelConfig, TankConfig
from hot_water_draw.estimator import estimate, forecast
from hot_water_draw.recorder import RecorderHistory

H = 3600
TANK = TankConfig(volume_liters=200.0, standby_loss_kw_per_k=0.002, ambient_temp_c=18.0)
CAP = TANK.capacity_kwh_per_k()
K = TANK.loss_kw_per_k() * 1.0 / CAP
SOURCES = [HeatSourceConfig(energy_entity="sensor.heat", cop=2.0)]
MODEL = ModelConfig(timezone="UTC", lookback_days=14, min_samples_per_hour=3)
# Monday 2026-01-05 00:00 UTC, so day types are easy to read.
START = int(datetime(2026, 1, 5, tzinfo=UTC).timestamp())
END = START + 14 * 24 * H


def _standby_next(prev: float) -> float:
    return (prev * (1 - K / 2) + K * TANK.ambient_temp_c) / (1 + K / 2)


def _history(
    hours: int,
    draws_kwh: dict[int, float] | None = None,
    heat_kwh: dict[int, float] | None = None,
    t0: float = 55.0,
    missing_temp: set[int] | None = None,
) -> RecorderHistory:
    """Simulate hourly means of a tank with standby loss, draws and heat input.

    The interval between the means of hour h-1 and hour h loses the standby
    loss (trapezoid), the draw attributed to hour h, and gains half the heat
    of hours h-1 and h. Heat in kWh electric is converted with the COP.
    """
    draws_kwh = draws_kwh or {}
    heat_kwh = heat_kwh or {}
    missing_temp = missing_temp or set()
    temps: dict[int, float] = {}
    energy: dict[int, float] = {}
    temp = t0
    for i in range(hours):
        ts = START + i * H
        energy[ts] = heat_kwh.get(i, 0.0)
        if i > 0:
            q_int = (energy[ts - H] + energy[ts]) * SOURCES[0].cop / 2
            # Solve the trapezoid balance for the new mean exactly.
            temp = (temp * (1 - K / 2) + K * TANK.ambient_temp_c + (q_int - draws_kwh.get(i, 0.0)) / CAP) / (1 + K / 2)
        if i not in missing_temp:
            temps[ts] = temp
    return RecorderHistory(heat_kwh={"sensor.heat": energy}, tank_temp_c={"sensor.t": temps})


def _estimate(history: RecorderHistory, model: ModelConfig = MODEL):
    return estimate(history, SOURCES, TANK, model, window_end_ts=END)


def test_standby_only_gives_zero_draw() -> None:
    result = _estimate(_history(14 * 24))
    assert result.daily_draw_kwh == pytest.approx(0.0, abs=1e-9)
    assert result.quality.fallback_used is False
    assert result.quality.segments == 1


def test_constant_heating_with_matching_rise_gives_zero_draw() -> None:
    heat = {i: 0.5 for i in range(14 * 24)}
    assert _estimate(_history(14 * 24, heat_kwh=heat)).daily_draw_kwh == pytest.approx(0.0, abs=1e-9)


def test_a_two_hour_segment_with_heating_gives_zero_draw() -> None:
    hist = _history(2, heat_kwh={0: 0.4, 1: 0.4})
    assert _estimate(hist).daily_draw_kwh == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("heat", [{0: 0.4, 1: 0.0}, {0: 0.0, 1: 0.4}, {0: 0.4, 1: 0.4}])
def test_edge_heat_is_integrated_exactly_with_a_known_draw(heat: dict[int, float]) -> None:
    """A known 0.3 kWh draw is recovered exactly whatever hour of the segment heats.

    A positive draw keeps the total positive, so undercounting heat at an edge
    cannot hide behind clipping.
    """
    hist = _history(2, heat_kwh=heat, draws_kwh={1: 0.3})
    assert _estimate(hist).daily_draw_kwh == pytest.approx(24 * 0.3, rel=1e-9)


def test_known_draws_across_a_gap_with_heating_at_the_edges() -> None:
    heat = {19: 0.5, 21: 0.5}
    draws = {10: 1.0, 19: 0.4, 30: 1.0}
    result = _estimate(_history(48, draws_kwh=draws, heat_kwh=heat, missing_temp={20}))
    assert result.daily_draw_kwh == pytest.approx(24 * 2.4 / 45, rel=1e-9)


def test_a_capacity_mistake_would_show() -> None:
    """Independent numbers: an extra 1 K drop of 200 L is 0.232556 kWh of draw.

    Lowering the second mean by 1 K also lowers the trapezoid standby loss of
    the interval by half of 2 W/K, so the draw is 0.232556 + 0.001 kWh.
    """
    hist = _history(2)
    temps = hist.tank_temp_c["sensor.t"]
    temps[START + H] -= 1.0
    assert _estimate(hist).daily_draw_kwh == pytest.approx(24 * (0.232556 + 0.001), rel=1e-5)


def test_several_sensors_and_sources_with_missing_values() -> None:
    """Temperatures are averaged; an hour missing any sensor or source is unusable."""
    base = _history(6, draws_kwh={2: 0.5, 4: 0.5})
    temps = base.tank_temp_c["sensor.t"]
    top = {ts: t + 2.0 for ts, t in temps.items()}
    low = {ts: t - 2.0 for ts, t in temps.items() if ts != START + 3 * H}
    second = {ts: 0.0 for ts in base.heat_kwh["sensor.heat"] if ts != START + 5 * H}
    sources = [*SOURCES, HeatSourceConfig(energy_entity="sensor.elem", cop=1.0)]
    hist = RecorderHistory(
        heat_kwh={"sensor.heat": base.heat_kwh["sensor.heat"], "sensor.elem": second},
        tank_temp_c={"sensor.top": top, "sensor.low": low},
    )
    result = estimate(hist, sources, TANK, MODEL, window_end_ts=END)
    # Usable hours: 0, 1, 2, 4 (3 lacks a sensor, 5 lacks a source). Segments
    # 0-2 (2 intervals, draw at 2) and the single hour 4 (not a segment).
    assert result.quality.segments == 1
    assert result.daily_draw_kwh == pytest.approx(24 * 0.5 / 2, rel=1e-9)


def test_a_single_usable_hour_is_not_a_segment() -> None:
    result = _estimate(_history(3, missing_temp={1}))
    assert result.quality.fallback_used is True


def test_a_negative_total_is_clipped_and_flagged() -> None:
    """A tank that warms without recorded heat (a COP set too low, say) gives a negative balance."""
    hist = _history(48)
    temps = hist.tank_temp_c["sensor.t"]
    for ts in temps:
        if ts >= START + 10 * H:
            temps[ts] += 5.0
    result = _estimate(hist)
    assert result.daily_draw_kwh == 0.0
    assert result.quality.negative_total_clipped is True


def test_a_sparse_weekend_hour_borrows_the_pooled_value() -> None:
    """Draws on weekdays only at 07:00 with min 3 samples: weekend cells have 4 samples each,
    which is enough; with min 5 they fall back to the pooled hour."""
    draws = {d * 24 + 7: 2.0 for d in range(14) if d % 7 < 5}
    strict = ModelConfig(timezone="UTC", lookback_days=14, min_samples_per_hour=5)
    result = _estimate(_history(14 * 24, draws_kwh=draws), strict)
    assert result.quality.cells_by_level == {"cell": 24, "pooled": 24, "flat": 0}
    assert result.shape["weekend"][7] == pytest.approx(result.shape["weekday"][7], rel=0.5)


def test_zero_sum_day_types_are_reported_flat() -> None:
    result = _estimate(_history(14 * 24))
    assert result.quality.cells_by_level == {"cell": 0, "pooled": 0, "flat": 48}


def test_hours_next_to_heating_are_not_shape_samples() -> None:
    """Heating at 07:00 daily removes the samples of hours 6, 7 and 8 every day."""
    heat = {d * 24 + 7: 0.5 for d in range(14)}
    result = _estimate(_history(14 * 24, heat_kwh=heat))
    # Eligible without heating: every hour except the first and the last.
    assert result.quality.samples == (14 * 24 - 2) - 14 * 3


def test_known_draws_are_recovered() -> None:
    """2 kWh each day at 07:00 and 1 kWh at 21:00 -> 3 kWh per day."""
    draws = {d * 24 + 7: 2.0 for d in range(14)} | {d * 24 + 21: 1.0 for d in range(14)}
    result = _estimate(_history(14 * 24, draws_kwh=draws))
    # The first hour has no interval before it, so 14 days have 14*24 - 1 intervals.
    assert result.daily_draw_kwh == pytest.approx(24 * 42.0 / (14 * 24 - 1), rel=1e-9)
    weekday = result.shape["weekday"]
    assert weekday[7] == pytest.approx(2 / 3, abs=1e-9)
    assert weekday[21] == pytest.approx(1 / 3, abs=1e-9)
    assert sum(weekday) == pytest.approx(1.0)


def test_a_gap_splits_a_segment_and_draws_beside_it_still_count() -> None:
    draws = {10: 1.0, 30: 1.0}
    result = _estimate(_history(48, draws_kwh=draws, missing_temp={20}))
    assert result.quality.segments == 2
    # Intervals: hours 1..19 (19) and 22..47 (26) = 45; the gap removes 20 and 21.
    assert result.quality.segment_hours == 48 - 1
    assert result.daily_draw_kwh == pytest.approx(24 * 2.0 / 45, rel=1e-9)


def test_heating_across_an_hour_boundary_is_not_a_draw() -> None:
    """A heater from 00:30 to 01:30, simulated per minute and averaged per hour."""
    minutes = 6 * 60
    temp, temps_min = 45.0, []
    heat_per_hour = {0: 0.0, 1: 0.0}
    for m in range(minutes):
        on = 30 <= m < 90
        heat_kw_el = 2.0 if on else 0.0
        if on:
            heat_per_hour[m // 60] = heat_per_hour.get(m // 60, 0.0) + heat_kw_el / 60
        temp += (heat_kw_el * SOURCES[0].cop - TANK.loss_kw_per_k() * (temp - 18.0)) / 60 / CAP
        temps_min.append(temp)
    temps = {START + h * H: sum(temps_min[h * 60:(h + 1) * 60]) / 60 for h in range(6)}
    energy = {START + h * H: heat_per_hour.get(h, 0.0) for h in range(6)}
    hist = RecorderHistory(heat_kwh={"sensor.heat": energy}, tank_temp_c={"sensor.t": temps})
    # Over a whole segment the heat and the stored energy cancel; only the
    # trapezoid standby approximation remains, which is tiny here.
    assert _estimate(hist).daily_draw_kwh == pytest.approx(0.0, abs=0.05)


def test_draws_hidden_in_heating_hours_still_count_in_the_scale() -> None:
    """Draws always at 07:00 while heating at 07:00: no shape sample, but the scale sees them."""
    draws = {d * 24 + 7: 2.0 for d in range(14)}
    heat = {d * 24 + 7: 0.5 for d in range(14)}
    result = _estimate(_history(14 * 24, draws_kwh=draws, heat_kwh=heat))
    assert result.daily_draw_kwh == pytest.approx(24 * 28.0 / (14 * 24 - 1), rel=1e-9)
    # Hours 6, 7 and 8 never have a heating-free neighbourhood, so the shape
    # cannot be completed and falls back to flat.
    assert result.quality.cells_by_level["flat"] == 48
    assert result.shape["weekday"] == pytest.approx([1 / 24] * 24)


def test_zero_demand_gives_a_zero_forecast() -> None:
    result = _estimate(_history(14 * 24))
    points = forecast(result, MODEL, now_ts=END)
    assert all(kw == 0.0 for _, kw in points)


def test_cold_start_uses_the_fallback_daily_draw_flat() -> None:
    result = _estimate(RecorderHistory(heat_kwh={"sensor.heat": {}}, tank_temp_c={"sensor.t": {}}))
    assert result.quality.fallback_used is True
    assert result.daily_draw_kwh == MODEL.fallback_daily_draw_kwh
    points = forecast(result, MODEL, now_ts=END)
    assert all(kw == pytest.approx(6.0 / 24) for _, kw in points)


def test_weekday_and_weekend_shapes_differ() -> None:
    draws: dict[int, float] = {}
    for d in range(14):
        weekend = (d % 7) >= 5
        draws[d * 24 + (10 if weekend else 7)] = 2.0
    result = _estimate(_history(14 * 24, draws_kwh=draws))
    assert max(range(24), key=lambda h: result.shape["weekday"][h]) == 7
    assert max(range(24), key=lambda h: result.shape["weekend"][h]) == 10


def test_recency_decay_favours_recent_days() -> None:
    """Week one draws at 07:00, week two at 08:00; with decay 08:00 dominates."""
    draws = {d * 24 + (7 if d < 7 else 8): 2.0 for d in range(14) if d % 7 < 5}
    decayed = ModelConfig(timezone="UTC", lookback_days=14, min_samples_per_hour=3, lookback_decay=0.8)
    result = _estimate(_history(14 * 24, draws_kwh=draws), decayed)
    assert result.shape["weekday"][8] > result.shape["weekday"][7]


def test_negative_samples_are_clipped_and_counted() -> None:
    """A sensor bump that raises the tank without heat gives a negative sample."""
    draws = {d * 24 + 7: 2.0 for d in range(14)}
    hist = _history(14 * 24, draws_kwh=draws)
    temps = hist.tank_temp_c["sensor.t"]
    # Day 4 (a Friday) at 04:00: the bump makes that hour's sample negative.
    temps[START + 100 * H] += 0.5
    result = _estimate(hist)
    assert result.quality.clipped_samples >= 1
    assert all(v >= 0.0 for v in result.shape["weekday"])


def test_forecast_is_hourly_from_the_current_hour_with_an_endpoint() -> None:
    """The current hour is covered, and one extra point marks the horizon end."""
    draws = {d * 24 + 7: 2.0 for d in range(14)}
    result = _estimate(_history(14 * 24, draws_kwh=draws))
    points = forecast(result, MODEL, now_ts=END + 600)
    assert points[0][0] == END
    assert len(points) == MODEL.horizon_hours + 1
    assert all(b[0] - a[0] == H for a, b in zip(points, points[1:], strict=False))
    by_hour = {datetime.fromtimestamp(ts, UTC).hour: kw for ts, kw in points[:24]}
    assert by_hour[7] == pytest.approx(result.daily_draw_kwh * result.shape["weekday"][7])


def test_the_autumn_dst_day_has_25_local_hours() -> None:
    draws = {d * 24 + 7: 2.0 for d in range(14)}
    result = _estimate(_history(14 * 24, draws_kwh=draws))
    model = ModelConfig(timezone="Europe/Amsterdam")
    now = int(datetime(2026, 10, 24, 22, tzinfo=UTC).timestamp())
    tz = ZoneInfo("Europe/Amsterdam")
    day = [
        (ts, kw) for ts, kw in forecast(result, model, now_ts=now)
        if datetime.fromtimestamp(ts, tz).date() == datetime(2026, 10, 25).date()
    ]
    assert len(day) == 25
    twos = [kw for ts, kw in day if datetime.fromtimestamp(ts, tz).hour == 2]
    assert len(twos) == 2 and twos[0] == twos[1]


def test_a_dst_day_uses_the_local_hour_of_each_interval() -> None:
    """On 2026-03-29 Europe/Amsterdam skips 02:00; each UTC hour takes its local hour's cell."""
    draws = {d * 24 + 7: 2.0 for d in range(14)}
    result = _estimate(_history(14 * 24, draws_kwh=draws))
    model = ModelConfig(timezone="Europe/Amsterdam")
    now = int(datetime(2026, 3, 28, 23, tzinfo=UTC).timestamp())
    points = forecast(result, model, now_ts=now)
    tz = ZoneInfo("Europe/Amsterdam")
    day = [(ts, kw) for ts, kw in points if datetime.fromtimestamp(ts, tz).date() == datetime(2026, 3, 29).date()]
    assert len(day) == 23
    for ts, kw in day:
        local = datetime.fromtimestamp(ts, tz)
        daytype = "weekend" if local.weekday() >= 5 else "weekday"
        assert kw == pytest.approx(result.daily_draw_kwh * result.shape[daytype][local.hour])


@pytest.mark.parametrize(
    "daily_total_kwh,flat",
    [(0.5e-3, True), (1e-3, True), (2e-3, False)],
)
def test_a_day_type_needs_more_than_1_wh_of_shape(daily_total_kwh: float, flat: bool) -> None:
    """The threshold applies to the sum of the 24 unnormalised hourly means of a day type."""
    from hot_water_draw.estimator import _shape

    per_hour = daily_total_kwh / 24
    cells = {("weekday", h): [(1.0, per_hour)] * 3 for h in range(24)}
    # Weekday 07:00 carries a little more, so a retained shape is not flat.
    cells[("weekday", 7)] = [(1.0, per_hour * 1.5)] * 3
    cells |= {("weekend", h): [(1.0, 0.0)] * 3 for h in range(24)}
    if daily_total_kwh == 1e-3:
        cells[("weekday", 7)] = [(1.0, per_hour)] * 3  # exactly 1 Wh
    shape, levels = _shape(cells, 3)
    assert shape["weekend"] == pytest.approx([1 / 24] * 24)
    if flat:
        assert shape["weekday"] == pytest.approx([1 / 24] * 24)
        assert levels == {"cell": 0, "pooled": 0, "flat": 48}
    else:
        assert shape["weekday"][7] > 1 / 24
        assert levels == {"cell": 24, "pooled": 0, "flat": 24}
