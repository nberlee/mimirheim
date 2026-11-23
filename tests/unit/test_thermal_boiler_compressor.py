"""Unit tests for the compressor heat source of ThermalBoilerDevice.

A compressor delivers a constant heat output while it runs, and its electric
power depends on the tank temperature at the start of the step and on the
heat source temperature. These tests check the temperature dynamics, the exact
electric power, the per-source heating ceiling and the behaviour of a tank that
starts above that ceiling. Legacy (``elec_power_kw``) tanks are covered by
test_thermal_boiler_constraints.py and must be unaffected.

Default compressor tank: 200 L, heat output 2.0 kW (2.15 K per 15-minute
step), electric power 1.1 kW at 35 °C and 1.9 kW at 58 °C, cooling 2 K/h.
"""

from __future__ import annotations

from typing import Any

import pytest

from mimirheim.config.schema import ThermalBoilerConfig
from mimirheim.core.bundle import ThermalBoilerInputs
from mimirheim.core.context import ModelContext
from mimirheim.core.solver_backend import CBCSolverBackend
from mimirheim.devices.thermal_boiler import ThermalBoilerDevice, _WATER_THERMAL_CAP_KWH_PER_LITRE_K

_DT = 0.25
_CAP = 200.0 * _WATER_THERMAL_CAP_KWH_PER_LITRE_K
_RISE = 2.0 * _DT / _CAP
_COOL = 2.0 * _DT


def _config(
    setpoint_c: float = 60.0,
    max_temp_c: float = 60.0,
    elec_kw_per_k_source: float = 0.0,
    min_run_steps: int = 0,
    wear_cost: float = 0.0,
    heat_output_kw: float = 2.0,
    curve: tuple[float, float] = (1.1, 1.9),
) -> ThermalBoilerConfig:
    return ThermalBoilerConfig(
        volume_liters=200.0,
        setpoint_c=setpoint_c,
        min_temp_c=40.0,
        cooling_rate_k_per_hour=2.0,
        min_run_steps=min_run_steps,
        wear_cost_eur_per_kwh=wear_cost,
        compressor={
            "heat_output_kw": heat_output_kw,
            "elec_kw_at_temp": [
                {"temp_c": 35.0, "elec_kw": curve[0]},
                {"temp_c": 58.0, "elec_kw": curve[1]},
            ],
            "max_temp_c": max_temp_c,
            "elec_kw_per_k_source": elec_kw_per_k_source,
        },
    )


def _build(
    cfg: ThermalBoilerConfig, current_temp_c: float = 45.0, horizon: int = 4, source_temp_c: float | None = None
) -> tuple[ModelContext, ThermalBoilerDevice]:
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=horizon, dt=_DT)
    device = ThermalBoilerDevice(name="dhw", config=cfg)
    device.add_variables(ctx)
    device.add_constraints(
        ctx, inputs=ThermalBoilerInputs(current_temp_c=current_temp_c, source_temp_c=source_temp_c)
    )
    return ctx, device


def _value(ctx: ModelContext, expr: Any) -> float:
    return float(expr) if isinstance(expr, (int, float)) else ctx.solver.var_value(expr)


def _on(ctx: ModelContext, device: ThermalBoilerDevice, t: int) -> int:
    return round(ctx.solver.var_value(device._heater_on[t]))


def _prior(ctx: ModelContext, device: ThermalBoilerDevice, t: int, t0: float) -> float:
    return t0 if t == 0 else device.planned_temp_c(ctx, t - 1)


def test_dynamics_use_the_compressor_heat_output() -> None:
    ctx, device = _build(_config())
    for t in ctx.T:
        ctx.solver.add_constraint(device._heater_on[t] == 1)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    for t in ctx.T:
        assert device.planned_temp_c(ctx, t) == pytest.approx(45.0 + (t + 1) * (_RISE - _COOL), abs=1e-6)


@pytest.mark.parametrize("sense", ["min", "max"])
def test_electric_power_is_exact_at_the_start_of_step_temperature(sense: str) -> None:
    """The product on[t] * T[t-1] is exact in both objective directions, on and off."""
    cfg = _config()
    ctx, device = _build(cfg)
    pattern = [1, 0, 1, 1]
    for t, on in enumerate(pattern):
        ctx.solver.add_constraint(device._heater_on[t] == on)
    draw = sum(-device.net_power(t) for t in ctx.T)
    if sense == "min":
        ctx.solver.set_objective_minimize(draw)
    else:
        ctx.solver.set_objective_maximize(draw)
    ctx.solver.solve()
    assert cfg.compressor is not None
    for t, on in enumerate(pattern):
        expected = cfg.compressor.elec_kw(_prior(ctx, device, t, 45.0)) if on else 0.0
        assert -_value(ctx, device.net_power(t)) == pytest.approx(expected, abs=1e-6)


def test_source_temperature_shifts_the_power() -> None:
    cfg = _config(elec_kw_per_k_source=0.02)
    assert cfg.compressor is not None
    for source, expected_shift in ((0.0, -0.2), (None, 0.0)):
        ctx, device = _build(cfg, horizon=1, source_temp_c=source)
        ctx.solver.add_constraint(device._heater_on[0] == 1)
        ctx.solver.set_objective_minimize(device._T_tank[0])
        ctx.solver.solve()
        expected = cfg.compressor.elec_kw(45.0) + expected_shift
        assert -_value(ctx, device.net_power(0)) == pytest.approx(expected, abs=1e-6)


def _heat_as_much_as_possible(ctx: ModelContext, device: ThermalBoilerDevice) -> None:
    ctx.solver.set_objective_maximize(sum(device._heater_on[t] for t in ctx.T))
    ctx.solver.solve()


@pytest.mark.parametrize("setpoint_c,max_temp_c", [(60.0, 55.0), (50.0, 60.0)])
def test_heating_peak_stays_below_the_lower_of_source_limit_and_setpoint(
    setpoint_c: float, max_temp_c: float
) -> None:
    ceiling = min(setpoint_c, max_temp_c)
    ctx, device = _build(_config(setpoint_c=setpoint_c, max_temp_c=max_temp_c), current_temp_c=44.0, horizon=8)
    _heat_as_much_as_possible(ctx, device)
    assert sum(_on(ctx, device, t) for t in ctx.T) > 0
    for t in ctx.T:
        if _on(ctx, device, t):
            assert _prior(ctx, device, t, 44.0) + _RISE <= ceiling + 1e-6


def test_compressor_cannot_start_when_the_step_would_pass_its_limit() -> None:
    ctx, device = _build(_config(), current_temp_c=60.0 - _RISE + 0.01, horizon=1)
    _heat_as_much_as_possible(ctx, device)
    assert _on(ctx, device, 0) == 0


def test_initially_overheated_tank_cools_passively() -> None:
    """A tank left above every limit (e.g. by the appliance's own program) stays feasible."""
    ctx, device = _build(_config(setpoint_c=55.0, max_temp_c=55.0), current_temp_c=65.0, horizon=8)
    _heat_as_much_as_possible(ctx, device)
    for t in ctx.T:
        assert device.planned_temp_c(ctx, t) <= 65.0 + 1e-6
        if _on(ctx, device, t):
            assert _prior(ctx, device, t, 65.0) + _RISE <= 55.0 + 1e-6
    assert device.planned_temp_c(ctx, 0) == pytest.approx(65.0 - _COOL, abs=1e-6)


def test_wear_cost_uses_the_compressor_power() -> None:
    cfg = _config(wear_cost=0.1)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, horizon=1)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    (term,) = device.objective_terms(0)
    assert _value(ctx, term) == pytest.approx(0.1 * cfg.compressor.elec_kw(45.0) * _DT, abs=1e-6)


def test_min_run_applies_to_the_compressor() -> None:
    # A run already going at t=0 is ticket 05 (carry-over); start from off.
    ctx, device = _build(_config(min_run_steps=3), current_temp_c=45.0, horizon=4)
    ctx.solver.add_constraint(device._heater_on[0] == 0)
    ctx.solver.add_constraint(device._heater_on[1] == 1)
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    ctx.solver.solve()
    assert [_on(ctx, device, t) for t in ctx.T] == [0, 1, 1, 1]


@pytest.mark.parametrize("curve", [(1.1, 1.9), (1.5, 1.5), (1.9, 1.1)], ids=["rising", "flat", "falling"])
def test_source_temperature_shifts_the_power_at_every_step(curve: tuple[float, float]) -> None:
    """The source shift applies after t=0 too, with and without the z product."""
    cfg = _config(elec_kw_per_k_source=0.02, curve=curve)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, source_temp_c=0.0)
    pattern = [1, 0, 1, 1]
    for t, on in enumerate(pattern):
        ctx.solver.add_constraint(device._heater_on[t] == on)
    ctx.solver.set_objective_minimize(sum(-device.net_power(t) for t in ctx.T))
    ctx.solver.solve()
    for t, on in enumerate(pattern):
        expected = cfg.compressor.elec_kw(_prior(ctx, device, t, 45.0), 0.0) if on else 0.0
        assert -_value(ctx, device.net_power(t)) == pytest.approx(expected, abs=1e-6)


def test_floor_forces_heating() -> None:
    """Without heating the tank would cool below min_temp_c in the first step.

    A one-step horizon tests the explicit floor on the last step: for earlier
    steps the exact linearisation of the next step already implies it.
    """
    ctx, device = _build(_config(), current_temp_c=40.2, horizon=1)
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    assert ctx.solver.solve() in ("optimal", "feasible")
    assert _on(ctx, device, 0) == 1
    for t in ctx.T:
        assert device.planned_temp_c(ctx, t) >= 40.0 - 1e-6


def test_tank_slightly_below_the_floor_recovers_in_one_step() -> None:
    ctx, device = _build(_config(), current_temp_c=39.0, horizon=2)
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    assert ctx.solver.solve() in ("optimal", "feasible")
    assert _on(ctx, device, 0) == 1
    assert device.planned_temp_c(ctx, 0) == pytest.approx(39.0 - _COOL + _RISE, abs=1e-6)


def test_tank_too_far_below_the_floor_is_infeasible() -> None:
    """One full step of heating cannot restore the floor, so the solve fails."""
    ctx, device = _build(_config(), current_temp_c=37.0, horizon=1)
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    assert ctx.solver.solve() == "infeasible"


def test_start_power_is_clamped_to_the_validated_range() -> None:
    """Below min_temp_c - 5 the power is taken at that bound, where it is known to be positive."""
    cfg = _config(heat_output_kw=7.0)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, current_temp_c=34.0, horizon=1)
    ctx.solver.set_objective_minimize(device._heater_on[0])
    assert ctx.solver.solve() in ("optimal", "feasible")
    assert _on(ctx, device, 0) == 1
    assert -_value(ctx, device.net_power(0)) == pytest.approx(cfg.compressor.elec_kw(35.0), abs=1e-6)


# ---------------------------------------------------------------------------
# Minimum run carried over from the previous solve (current_run_minutes)
# ---------------------------------------------------------------------------


def _build_running(
    current_run_minutes: float | None, current_temp_c: float = 45.0, horizon: int = 6, min_run_steps: int = 4
) -> tuple[ModelContext, ThermalBoilerDevice]:
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=horizon, dt=_DT)
    device = ThermalBoilerDevice(name="dhw", config=_config(min_run_steps=min_run_steps))
    device.add_variables(ctx)
    device.add_constraints(
        ctx,
        inputs=ThermalBoilerInputs(current_temp_c=current_temp_c, current_run_minutes=current_run_minutes),
    )
    return ctx, device


def _min_heating(ctx: ModelContext, device: ThermalBoilerDevice) -> list[int]:
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    assert ctx.solver.solve() in ("optimal", "feasible")
    return [_on(ctx, device, t) for t in ctx.T]


@pytest.mark.parametrize("run_minutes,forced", [(30.0, 2), (20.0, 3), (45.0, 1)])
def test_a_run_short_of_its_minimum_continues(run_minutes: float, forced: int) -> None:
    """min_run 4 steps = 60 min; the remaining minutes round up to whole steps."""
    ctx, device = _build_running(run_minutes)
    assert _min_heating(ctx, device) == [1] * forced + [0] * (6 - forced)


@pytest.mark.parametrize("run_minutes", [60.0, 90.0])
def test_a_run_past_its_minimum_may_stop_at_once(run_minutes: float) -> None:
    ctx, device = _build_running(run_minutes)
    assert _min_heating(ctx, device) == [0] * 6


def test_a_start_at_the_first_step_runs_the_full_minimum() -> None:
    """Off before the solve: switching on at t=0 is a start and must last min_run steps."""
    ctx, device = _build_running(0.0)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    assert _min_heating(ctx, device) == [1, 1, 1, 1, 0, 0]


def test_a_continuing_run_is_not_a_new_start() -> None:
    """Running past the minimum: staying on at t=0 imposes no new minimum."""
    ctx, device = _build_running(75.0)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    assert _min_heating(ctx, device) == [1, 0, 0, 0, 0, 0]


def test_unknown_run_time_keeps_the_previous_behaviour() -> None:
    ctx, device = _build_running(None)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    assert _min_heating(ctx, device) == [1, 0, 0, 0, 0, 0]


def test_forced_continuation_is_truncated_at_the_horizon_end() -> None:
    ctx, device = _build_running(15.0, horizon=2)
    assert _min_heating(ctx, device) == [1, 1]


def test_forced_continuation_stops_at_the_heating_ceiling() -> None:
    """The appliance stops at its limit whatever its minimum run; the model must stay feasible.

    From 54 °C each step adds rise - cool = 1.65 K, and the ceiling is 60 °C:
    step 0 peaks at 56.15, step 1 at 57.80, step 2 at 59.45 and step 3 would
    peak at 61.10. Only the first three of the four remaining steps are forced.
    """
    ctx, device = _build_running(0.001, current_temp_c=54.0, horizon=6)
    assert _min_heating(ctx, device) == [1, 1, 1, 0, 0, 0]


def test_repeated_solves_complete_one_minimum_run() -> None:
    """A run started in one solve is carried through the next solves."""
    plans = []
    for run_minutes in (0.0, 15.0, 30.0, 45.0, 60.0):
        ctx, device = _build_running(run_minutes)
        if run_minutes == 0.0:
            ctx.solver.add_constraint(device._heater_on[0] == 1)
        plans.append(_min_heating(ctx, device)[0])
    assert plans == [1, 1, 1, 1, 0]


@pytest.mark.parametrize("offset", [0.0, 5e-10, -5e-10])
def test_carry_over_and_start_ceiling_agree_at_the_boundary(offset: float) -> None:
    """Both checks treat a first-step peak at the ceiling the same, so the model stays feasible."""
    ctx, device = _build_running(15.0, current_temp_c=60.0 - _RISE + offset, horizon=4)
    plan = _min_heating(ctx, device)
    assert plan[0] == 1


def test_carry_over_is_dropped_when_the_first_step_clearly_passes_the_ceiling() -> None:
    ctx, device = _build_running(15.0, current_temp_c=60.0 - _RISE + 1e-6, horizon=4)
    assert _min_heating(ctx, device) == [0, 0, 0, 0]


def test_active_source_reports_the_compressor() -> None:
    ctx, device = _build(_config(), horizon=2)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    ctx.solver.add_constraint(device._heater_on[1] == 0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    assert [device.active_source(ctx, t) for t in ctx.T] == ["compressor", "off"]


def test_terminal_value_uses_the_compressor_cop_at_the_setpoint() -> None:
    """Heat left in the tank is valued at the electricity the compressor would need for it."""
    cfg = _config(setpoint_c=55.0, max_temp_c=55.0)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, horizon=1)
    ctx.solver.add_constraint(device._heater_on[0] == 0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    cop_ref = 2.0 / cfg.compressor.elec_kw(55.0)
    expected = (45.0 - _COOL - 40.0) * _CAP / cop_ref
    assert _value(ctx, device.terminal_soc_var(ctx)) == pytest.approx(expected, abs=1e-6)
