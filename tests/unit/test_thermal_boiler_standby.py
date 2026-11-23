"""Unit tests for the standby loss mode of ThermalBoilerDevice.

In standby mode the tank loses heat in proportion to its temperature above the
room (``standby_loss_kw_per_k * (T - ambient_temp_c)``) and, with a draw
forecast, loses the forecast hot water draw per step. The floor is soft at the
device level: ``shortfall[t]`` measures how far the tank dips below
``min_temp_c``, and ``floor_shortfall_kh`` sums it in K·h for the solve to
bound. Legacy tanks (``cooling_rate_k_per_hour``) keep their exact model.

Default standby tank: 200 L, 3 kW resistive heater, 2 W/K loss, room 18 °C.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from mimirheim.config.schema import MimirheimConfig, ThermalBoilerConfig
from mimirheim.core.bundle import SolveBundle, ThermalBoilerInputs
from mimirheim.core.context import ModelContext
from mimirheim.core.model_builder import build_and_solve
from mimirheim.core.solver_backend import CBCSolverBackend
from mimirheim.devices.thermal_boiler import ThermalBoilerDevice, _WATER_THERMAL_CAP_KWH_PER_LITRE_K

_DT = 0.25
_CAP = 200.0 * _WATER_THERMAL_CAP_KWH_PER_LITRE_K
_LOSS = 0.002
_AMBIENT = 18.0


def _config(
    use_draw_forecast: bool = True, setpoint_c: float = 55.0, cop: float = 1.0, min_run_steps: int = 0
) -> ThermalBoilerConfig:
    return ThermalBoilerConfig(
        volume_liters=200.0,
        elec_power_kw=3.0,
        cop=cop,
        min_run_steps=min_run_steps,
        setpoint_c=setpoint_c,
        min_temp_c=40.0,
        standby_loss_kw_per_k=_LOSS,
        ambient_temp_c=_AMBIENT,
        use_draw_forecast=use_draw_forecast,
    )


def _build(
    cfg: ThermalBoilerConfig, draw_kw: list[float] | None, t0: float = 45.0
) -> tuple[ModelContext, ThermalBoilerDevice]:
    horizon = len(draw_kw) if draw_kw is not None else 4
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=horizon, dt=_DT)
    device = ThermalBoilerDevice(name="dhw", config=cfg)
    device.add_variables(ctx)
    device.add_constraints(ctx, inputs=ThermalBoilerInputs(current_temp_c=t0, draw_kw=draw_kw))
    return ctx, device


def _next(prior: float, on: int, draw: float, cop: float = 1.0) -> float:
    return prior + (3.0 * cop * on - _LOSS * (prior - _AMBIENT) - draw) * _DT / _CAP


def _value(ctx: ModelContext, expr: Any) -> float:
    return float(expr) if isinstance(expr, (int, float)) else ctx.solver.var_value(expr)


def test_standby_dynamics_subtract_loss_and_draw() -> None:
    draws = [0.0, 1.0, 0.0, 2.0]
    pattern = [1, 0, 0, 1]
    ctx, device = _build(_config(), draws)
    for t, on in enumerate(pattern):
        ctx.solver.add_constraint(device._heater_on[t] == on)
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    ctx.solver.solve()
    prior = 45.0
    for t in ctx.T:
        prior = _next(prior, pattern[t], draws[t])
        assert ctx.solver.var_value(device._T_tank[t]) == pytest.approx(prior, abs=1e-6)


def test_without_a_draw_forecast_only_standby_loss_applies() -> None:
    ctx, device = _build(_config(use_draw_forecast=False), None)
    for t in ctx.T:
        ctx.solver.add_constraint(device._heater_on[t] == 0)
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    ctx.solver.solve()
    prior = 45.0
    for t in ctx.T:
        prior = _next(prior, 0, 0.0)
        assert ctx.solver.var_value(device._T_tank[t]) == pytest.approx(prior, abs=1e-6)


def test_heating_uses_the_cop() -> None:
    ctx, device = _build(_config(cop=3.0), [0.0, 0.0])
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    ctx.solver.add_constraint(device._heater_on[1] == 0)
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    ctx.solver.solve()
    assert ctx.solver.var_value(device._T_tank[0]) == pytest.approx(_next(45.0, 1, 0.0, cop=3.0), abs=1e-6)


def test_min_run_applies_in_standby_mode() -> None:
    ctx, device = _build(_config(min_run_steps=3), [0.0, 0.0, 0.0, 0.0])
    ctx.solver.add_constraint(device._heater_on[0] == 0)
    ctx.solver.add_constraint(device._heater_on[1] == 1)
    ctx.solver.set_objective_minimize(sum(device._heater_on[t] for t in ctx.T))
    ctx.solver.solve()
    assert [round(ctx.solver.var_value(device._heater_on[t])) for t in ctx.T] == [0, 1, 1, 1]


def test_draws_are_ignored_when_the_forecast_is_off() -> None:
    """A bundle that carries draws for a tank without use_draw_forecast does not change its dynamics."""
    ctx, device = _build(_config(use_draw_forecast=False), [5.0, 5.0])
    for t in ctx.T:
        ctx.solver.add_constraint(device._heater_on[t] == 0)
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    ctx.solver.solve()
    assert ctx.solver.var_value(device._T_tank[0]) == pytest.approx(_next(45.0, 0, 0.0), abs=1e-6)


def test_a_draw_forecast_tank_needs_draws() -> None:
    with pytest.raises(ValueError, match="draw"):
        _build(_config(use_draw_forecast=True), None)


def test_an_uncoverable_draw_becomes_shortfall_far_below_the_floor() -> None:
    """A 50 kW draw in one step takes the tank below 0 °C in the model; it stays feasible."""
    ctx, device = _build(_config(), [50.0, 0.0])
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    assert ctx.solver.solve() in ("optimal", "feasible")
    t0_end = ctx.solver.var_value(device._T_tank[0])
    assert t0_end < 0.0
    assert ctx.solver.var_value(device._shortfall[0]) == pytest.approx(40.0 - t0_end, abs=1e-6)
    expected_kh = sum(ctx.solver.var_value(device._shortfall[t]) * _DT for t in ctx.T)
    assert _value(ctx, device.floor_shortfall_kh(ctx)) == pytest.approx(expected_kh, abs=1e-6)


def test_holding_shortfall_at_zero_is_the_hard_floor() -> None:
    ctx, device = _build(_config(), [40.0, 0.0])
    ctx.solver.add_constraint(device.floor_shortfall_kh(ctx) <= 0.0)
    ctx.solver.set_objective_minimize(device._heater_on[0])
    assert ctx.solver.solve() == "infeasible"


def test_the_setpoint_stays_a_hard_cap() -> None:
    ctx, device = _build(_config(setpoint_c=46.0), [0.0, 0.0, 0.0, 0.0])
    ctx.solver.set_objective_maximize(sum(device._heater_on[t] for t in ctx.T))
    ctx.solver.solve()
    for t in ctx.T:
        assert ctx.solver.var_value(device._T_tank[t]) <= 46.0 + 1e-6


def test_legacy_tanks_have_no_shortfall() -> None:
    cfg = ThermalBoilerConfig(
        volume_liters=200.0, elec_power_kw=3.0, setpoint_c=55.0, min_temp_c=40.0, cooling_rate_k_per_hour=2.0
    )
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=2, dt=_DT)
    device = ThermalBoilerDevice(name="dhw", config=cfg)
    device.add_variables(ctx)
    device.add_constraints(ctx, inputs=ThermalBoilerInputs(current_temp_c=45.0))
    assert device.floor_shortfall_kh(ctx) is None
    assert device._shortfall == {}


def _system(draw_kw: list[float] | None, use_draw_forecast: bool = True) -> tuple[SolveBundle, MimirheimConfig]:
    n = 4
    config = MimirheimConfig.model_validate(
        {
            "mqtt": {"host": "localhost", "client_id": "test"},
            "grid": {"import_limit_kw": 20.0, "export_limit_kw": 20.0},
            "static_loads": {"base": {}},
            "thermal_boilers": {
                "dhw": {
                    "volume_liters": 200.0,
                    "elec_power_kw": 3.0,
                    "setpoint_c": 55.0,
                    "min_temp_c": 40.0,
                    "standby_loss_kw_per_k": _LOSS,
                    "ambient_temp_c": _AMBIENT,
                    "use_draw_forecast": use_draw_forecast,
                }
            },
        }
    )
    bundle = SolveBundle(
        solve_time_utc=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        horizon_prices=[0.25] * n,
        horizon_export_prices=[0.10] * n,
        horizon_confidence=[1.0] * n,
        pv_forecast=[0.0] * n,
        base_load_forecast=[0.5] * n,
        thermal_boiler_inputs={"dhw": ThermalBoilerInputs(current_temp_c=45.0, draw_kw=draw_kw)},
    )
    return bundle, config


def test_build_and_solve_absorbs_an_uncoverable_draw() -> None:
    """The soft floor keeps the solve feasible; see test_floor_shortfall_presolve.py for its priority."""
    bundle, config = _system([0.5, 0.5, 0.5, 0.5])
    assert build_and_solve(bundle, config).solve_status in ("optimal", "feasible")
    bundle, config = _system([40.0, 0.0, 0.0, 0.0])
    assert build_and_solve(bundle, config).solve_status in ("optimal", "feasible")


def test_build_and_solve_rejects_a_missing_draw_series() -> None:
    bundle, config = _system(None)
    with pytest.raises(ValueError, match="draw"):
        build_and_solve(bundle, config)
