"""Unit tests for the element heat source of ThermalBoilerDevice.

The element converts electricity to heat one to one and runs at any power
between zero and its rating, so it can stop exactly at a target temperature.
It never runs in the same step as the compressor, and it heats only while the
step peak stays at or below ``min(element.max_temp_c, setpoint_c)``.

Default tank: 200 L, element 6.0 kW (6.45 K per 15-minute step at full
power), cooling 2 K/h.
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
_COOL = 2.0 * _DT
_COMPRESSOR = {
    "heat_output_kw": 2.0,
    "elec_kw_at_temp": [{"temp_c": 35.0, "elec_kw": 1.1}, {"temp_c": 58.0, "elec_kw": 1.9}],
    "max_temp_c": 55.0,
}


def _config(
    setpoint_c: float = 67.0,
    element_max_c: float = 67.0,
    with_compressor: bool = False,
    wear_cost: float = 0.0,
) -> ThermalBoilerConfig:
    data: dict[str, Any] = {
        "volume_liters": 200.0,
        "setpoint_c": setpoint_c,
        "min_temp_c": 40.0,
        "cooling_rate_k_per_hour": 2.0,
        "wear_cost_eur_per_kwh": wear_cost,
        "element": {"power_kw": 6.0, "max_temp_c": element_max_c},
    }
    if with_compressor:
        data["compressor"] = _COMPRESSOR
    return ThermalBoilerConfig.model_validate(data)


def _build(
    cfg: ThermalBoilerConfig, current_temp_c: float = 45.0, horizon: int = 4
) -> tuple[ModelContext, ThermalBoilerDevice]:
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=horizon, dt=_DT)
    device = ThermalBoilerDevice(name="dhw", config=cfg)
    device.add_variables(ctx)
    device.add_constraints(ctx, inputs=ThermalBoilerInputs(current_temp_c=current_temp_c))
    return ctx, device


def _value(ctx: ModelContext, expr: Any) -> float:
    return float(expr) if isinstance(expr, (int, float)) else ctx.solver.var_value(expr)


def _elem_kw(ctx: ModelContext, device: ThermalBoilerDevice, t: int) -> float:
    return ctx.solver.var_value(device._elem_kw[t])


def _prior(ctx: ModelContext, device: ThermalBoilerDevice, t: int, t0: float) -> float:
    return t0 if t == 0 else device.planned_temp_c(ctx, t - 1)


def test_element_heats_one_to_one() -> None:
    ctx, device = _build(_config())
    ctx.solver.add_constraint(device._elem_kw[0] == 4.0)
    ctx.solver.set_objective_minimize(sum(device._elem_kw[t] for t in ctx.T))
    ctx.solver.solve()
    assert device.planned_temp_c(ctx, 0) == pytest.approx(45.0 - _COOL + 4.0 * _DT / _CAP, abs=1e-6)
    assert -_value(ctx, device.net_power(0)) == pytest.approx(4.0, abs=1e-6)


def test_partial_power_reaches_a_target_without_overshoot() -> None:
    """From 55 °C the element reaches at least 64 °C and never passes 67 °C."""
    ctx, device = _build(_config(), current_temp_c=55.0, horizon=4)
    ctx.solver.add_constraint(device._T_tank[3] >= 64.0)
    ctx.solver.set_objective_minimize(sum(device._elem_kw[t] for t in ctx.T))
    ctx.solver.solve()
    assert device.planned_temp_c(ctx, 3) == pytest.approx(64.0, abs=1e-6)
    for t in ctx.T:
        peak = _prior(ctx, device, t, 55.0) + _elem_kw(ctx, device, t) * _DT / _CAP
        assert peak <= 67.0 + 1e-6


@pytest.mark.parametrize("setpoint_c,element_max_c", [(67.0, 60.0), (58.0, 67.0)])
def test_heating_peak_stays_below_the_lower_of_element_limit_and_setpoint(
    setpoint_c: float, element_max_c: float
) -> None:
    ceiling = min(setpoint_c, element_max_c)
    ctx, device = _build(_config(setpoint_c=setpoint_c, element_max_c=element_max_c), horizon=6)
    ctx.solver.set_objective_maximize(sum(device._elem_kw[t] for t in ctx.T))
    ctx.solver.solve()
    assert sum(_elem_kw(ctx, device, t) for t in ctx.T) > 0
    for t in ctx.T:
        peak = _prior(ctx, device, t, 45.0) + _elem_kw(ctx, device, t) * _DT / _CAP
        assert peak <= ceiling + 1e-6


def test_initially_hot_tank_keeps_the_element_idle_and_cools() -> None:
    ctx, device = _build(_config(), current_temp_c=70.0, horizon=4)
    ctx.solver.set_objective_maximize(sum(device._elem_kw[t] for t in ctx.T))
    ctx.solver.solve()
    for t in ctx.T:
        assert _elem_kw(ctx, device, t) == pytest.approx(0.0, abs=1e-6)
        assert device.planned_temp_c(ctx, t) == pytest.approx(70.0 - (t + 1) * _COOL, abs=1e-6)


def test_element_and_compressor_never_share_a_step() -> None:
    cfg = _config(setpoint_c=67.0, with_compressor=True)
    ctx, device = _build(cfg, current_temp_c=41.0, horizon=6)
    # Reward both heat sources so the solver would run them together if allowed.
    ctx.solver.set_objective_maximize(
        sum(device._heater_on[t] + device._elem_on[t] for t in ctx.T)
    )
    ctx.solver.solve()
    for t in ctx.T:
        on = round(ctx.solver.var_value(device._heater_on[t]))
        elem_on = round(ctx.solver.var_value(device._elem_on[t]))
        assert on + elem_on <= 1


def test_net_power_sums_compressor_and_element() -> None:
    cfg = _config(with_compressor=True)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, current_temp_c=45.0, horizon=2)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    ctx.solver.add_constraint(device._elem_kw[1] == 3.0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    assert -_value(ctx, device.net_power(0)) == pytest.approx(cfg.compressor.elec_kw(45.0), abs=1e-6)
    assert -_value(ctx, device.net_power(1)) == pytest.approx(3.0, abs=1e-6)
    # Energy balance: both sources heat the same tank.
    t0_end = 45.0 - _COOL + 2.0 * _DT / _CAP
    assert device.planned_temp_c(ctx, 0) == pytest.approx(t0_end, abs=1e-6)
    assert device.planned_temp_c(ctx, 1) == pytest.approx(t0_end - _COOL + 3.0 * _DT / _CAP, abs=1e-6)


def test_element_floor_is_enforced_on_the_last_step() -> None:
    ctx, device = _build(_config(), current_temp_c=40.2, horizon=1)
    ctx.solver.set_objective_minimize(device._elem_kw[0])
    assert ctx.solver.solve() in ("optimal", "feasible")
    assert device.planned_temp_c(ctx, 0) == pytest.approx(40.0, abs=1e-6)


def test_wear_cost_applies_to_the_compressor_only() -> None:
    cfg = _config(wear_cost=0.1, with_compressor=True)
    assert cfg.compressor is not None
    ctx, device = _build(cfg, horizon=2)
    ctx.solver.add_constraint(device._heater_on[0] == 1)
    ctx.solver.add_constraint(device._elem_kw[1] == 2.0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    (wear_0,) = device.objective_terms(0)
    (wear_1,) = device.objective_terms(1)
    assert _value(ctx, wear_0) == pytest.approx(0.1 * cfg.compressor.elec_kw(45.0) * _DT, abs=1e-6)
    assert _value(ctx, wear_1) == pytest.approx(0.0, abs=1e-6)


def test_element_only_tank_has_no_wear_cost() -> None:
    ctx, device = _build(_config(wear_cost=0.1), horizon=1)
    assert device.objective_terms(0) == []


def test_active_source_reports_the_element_only_when_it_delivers_power() -> None:
    ctx, device = _build(_config(), horizon=2)
    ctx.solver.add_constraint(device._elem_kw[0] == 2.0)
    ctx.solver.add_constraint(device._elem_on[1] == 1)
    ctx.solver.add_constraint(device._elem_kw[1] == 0.0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    assert [device.active_source(ctx, t) for t in ctx.T] == ["element", "off"]


def test_element_only_terminal_value_uses_cop_one() -> None:
    ctx, device = _build(_config(), horizon=1)
    ctx.solver.add_constraint(device._elem_kw[0] == 0.0)
    ctx.solver.set_objective_minimize(device._T_tank[0])
    ctx.solver.solve()
    assert _value(ctx, device.terminal_soc_var(ctx)) == pytest.approx((45.0 - _COOL - 40.0) * _CAP, abs=1e-6)
