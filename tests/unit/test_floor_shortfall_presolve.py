"""Unit tests for the lexicographic soft tank floor in build_and_solve.

Standby-mode thermal boilers have a soft floor. Before anything else,
build_and_solve minimises the total floor shortfall alone and then holds it as
a hard bound, so no later objective (any strategy, the battery-care probe, the
terminal value) can trade tank temperature for money or import volume. When
the floor can be held, the bound is zero and the tank behaves exactly like a
hard floor; when a forecast draw cannot be covered, the solve stays feasible.

The pre-solve has four outcomes (see ``_bound_floor_shortfall``): no standby
tank, bounded, proven infeasible, and a time limit without a solution, which
continues minimising shortfall and marks the result degraded. The time spent
is deducted once from the cycle budget.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

import mimirheim.core.model_builder as model_builder
from mimirheim.config.schema import MimirheimConfig
from mimirheim.core.bundle import SolveBundle, ThermalBoilerInputs
from mimirheim.core.model_builder import build_and_solve
from mimirheim.core.post_process import apply_gain_threshold
from mimirheim.core.solver_backend import CBCSolverBackend

_N = 4


def _config(strategy_tank: dict | None = None, threshold: float = 0.0, time_limit: float = 12.0) -> MimirheimConfig:
    tank = strategy_tank or {
        "volume_liters": 200.0,
        "elec_power_kw": 3.0,
        "setpoint_c": 55.0,
        "min_temp_c": 40.0,
        "standby_loss_kw_per_k": 0.002,
        "ambient_temp_c": 18.0,
        "use_draw_forecast": True,
    }
    return MimirheimConfig.model_validate(
        {
            "mqtt": {"host": "localhost", "client_id": "test"},
            "grid": {"import_limit_kw": 20.0, "export_limit_kw": 20.0},
            "static_loads": {"base": {}},
            "thermal_boilers": {"dhw": tank},
            "objectives": {"min_dispatch_gain_eur": threshold},
            "solver": {"time_limit_seconds": time_limit},
        }
    )


def _bundle(draw_kw: list[float] | None, t0: float = 45.0, strategy: str = "minimize_cost",
            prices: list[float] | None = None) -> SolveBundle:
    return SolveBundle(
        strategy=strategy,
        solve_time_utc=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        horizon_prices=prices or [0.25] * _N,
        horizon_export_prices=[0.10] * _N,
        horizon_confidence=[1.0] * _N,
        pv_forecast=[0.0] * _N,
        base_load_forecast=[0.5] * _N,
        thermal_boiler_inputs={"dhw": ThermalBoilerInputs(current_temp_c=t0, draw_kw=draw_kw)},
    )


class TestSoftFloor:
    def test_an_uncoverable_draw_no_longer_fails_the_solve(self) -> None:
        result = build_and_solve(_bundle([40.0, 0.0, 0.0, 0.0]), _config())
        assert result.solve_status in ("optimal", "feasible")
        assert result.strategy_degraded is False
        # The least shortfall needs every possible kWh of heat in the draw step.
        assert result.schedule[0].devices["dhw"].kw == pytest.approx(-3.0)

    @pytest.mark.parametrize("strategy", ["minimize_cost", "minimize_consumption", "balanced"])
    def test_a_holdable_floor_is_hard_in_every_strategy(self, strategy: str) -> None:
        """Cooling 0.01 K below the floor in one step: every strategy must heat, as with a hard floor."""
        # Loss over one step at 40.01 °C: 0.002 kW/K * 22.01 K * 0.25 h / 0.2326 kWh/K = 0.0473 K.
        prices = [5.0, 0.25, 0.25, 0.25]
        result = build_and_solve(_bundle(None, t0=40.04, strategy=strategy, prices=prices), _config(
            strategy_tank={
                "volume_liters": 200.0, "elec_power_kw": 3.0, "setpoint_c": 55.0, "min_temp_c": 40.0,
                "standby_loss_kw_per_k": 0.002, "ambient_temp_c": 18.0,
            }
        ))
        assert result.solve_status in ("optimal", "feasible")
        assert result.schedule[0].devices["dhw"].kw == pytest.approx(-3.0)


def _wrap_solve(monkeypatch: pytest.MonkeyPatch, first: str, proved: bool) -> list[float]:
    """Make the first solver call return ``first``; record every time limit."""
    real = CBCSolverBackend.solve
    limits: list[float] = []

    def solve(self: CBCSolverBackend, time_limit_seconds: float = 59.0) -> str:
        limits.append(time_limit_seconds)
        if len(limits) == 1 and first != "real":
            self.last_status_proved_infeasible = proved
            return first
        return real(self, time_limit_seconds=time_limit_seconds)

    monkeypatch.setattr(CBCSolverBackend, "solve", solve)
    return limits


class TestOutcomes:
    def test_legacy_tanks_add_no_solve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limits = _wrap_solve(monkeypatch, "real", False)
        tank = {"volume_liters": 200.0, "elec_power_kw": 3.0, "setpoint_c": 55.0, "min_temp_c": 40.0,
                "cooling_rate_k_per_hour": 2.0}
        build_and_solve(_bundle(None), _config(strategy_tank=tank))
        assert len(limits) == 1

    def test_a_standby_tank_adds_one_pre_solve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limits = _wrap_solve(monkeypatch, "real", False)
        build_and_solve(_bundle([0.5] * _N), _config(time_limit=12.0))
        assert len(limits) == 2
        assert limits[0] == pytest.approx(12.0 / 6.0)

    def test_a_timeout_without_a_solution_keeps_minimising_shortfall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wrap_solve(monkeypatch, "infeasible", proved=False)
        result = build_and_solve(_bundle([40.0, 0.0, 0.0, 0.0]), _config())
        assert result.solve_status in ("optimal", "feasible")
        assert result.strategy_degraded is True
        assert result.schedule[0].devices["dhw"].kw == pytest.approx(-3.0)

    def test_a_proven_infeasible_pre_solve_adds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is bound or fixed; the main solve decides and reports."""
        _wrap_solve(monkeypatch, "infeasible", proved=True)
        result = build_and_solve(_bundle([0.5] * _N), _config())
        assert result.solve_status in ("optimal", "feasible")
        assert result.strategy_degraded is False


class TestBudget:
    def test_preliminary_time_is_deducted_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-solve and probe each use their full allowance; the strategy splits what is left."""
        total = 12.0
        limits = _wrap_solve(monkeypatch, "real", False)
        clock = iter([0.0, total / 6.0] + [total / 6.0] * 50)
        monkeypatch.setattr(model_builder, "_monotonic", lambda: next(clock))
        monkeypatch.setattr(model_builder, "_probe_care_targets", lambda ctx, batteries, config: total / 3.0)
        build_and_solve(_bundle([0.5] * _N, strategy="minimize_consumption"), _config(time_limit=total))
        remaining = total - total / 6.0 - total / 3.0
        # pre-solve, consumption phase 1, final solve
        assert limits == pytest.approx([total / 6.0, remaining / 2.0, remaining / 2.0])


class TestPostProcessing:
    def test_a_degraded_result_is_not_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wrap_solve(monkeypatch, "infeasible", proved=False)
        config = _config(threshold=1000.0)
        bundle = _bundle([40.0, 0.0, 0.0, 0.0])
        result = build_and_solve(bundle, config)
        assert result.strategy_degraded is True
        # A small positive gain that the threshold would otherwise suppress.
        result = result.model_copy(update={"naive_cost_eur": 1.0, "optimised_cost_eur": 0.9})
        assert apply_gain_threshold(result, bundle, config) is result
        assert apply_gain_threshold(
            result.model_copy(update={"strategy_degraded": False}), bundle, config
        ).dispatch_suppressed


def test_the_bound_is_in_place_before_the_battery_care_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe's witness must already respect the shortfall bound."""
    calls: list[str] = []
    real_bound = model_builder._bound_floor_shortfall

    def bound(*args: object) -> tuple[float, str]:
        calls.append("bound")
        return real_bound(*args)

    def probe(*args: object) -> float:
        calls.append("probe")
        return 0.0

    monkeypatch.setattr(model_builder, "_bound_floor_shortfall", bound)
    monkeypatch.setattr(model_builder, "_probe_care_targets", probe)
    build_and_solve(_bundle([0.5] * _N), _config())
    assert calls == ["bound", "probe"]


# ---------------------------------------------------------------------------
# Quantitative guarantee: the published schedule never dips further than the
# least achievable shortfall. Checked independently of the solver internals by
# replaying the heater schedule through the tank equation in Python and
# comparing with a separate device-only minimisation.
# ---------------------------------------------------------------------------

_LOSS, _ROOM, _MIN, _CAP = 0.002, 18.0, 40.0, 200.0 * 4186 / 3600 / 1000


def _replayed_shortfall_kh(result, draws: list[float], t0: float, heat_kw: float = 3.0) -> float:
    temp, total = t0, 0.0
    for t, step in enumerate(result.schedule):
        on = 1.0 if step.devices["dhw"].kw < -1e-6 else 0.0
        temp = temp + (heat_kw * on - _LOSS * (temp - _ROOM) - draws[t]) * 0.25 / _CAP
        total += max(0.0, _MIN - temp) * 0.25
    return total


def _least_shortfall_kh(draws: list[float], t0: float) -> float:
    from mimirheim.config.schema import ThermalBoilerConfig
    from mimirheim.core.context import ModelContext
    from mimirheim.devices.thermal_boiler import ThermalBoilerDevice

    cfg = ThermalBoilerConfig(
        volume_liters=200.0, elec_power_kw=3.0, setpoint_c=55.0, min_temp_c=_MIN,
        standby_loss_kw_per_k=_LOSS, ambient_temp_c=_ROOM, use_draw_forecast=True,
    )
    ctx = ModelContext(solver=CBCSolverBackend(), horizon=len(draws), dt=0.25)
    device = ThermalBoilerDevice(name="dhw", config=cfg)
    device.add_variables(ctx)
    device.add_constraints(ctx, inputs=ThermalBoilerInputs(current_temp_c=t0, draw_kw=draws))
    ctx.solver.set_objective_minimize(device.floor_shortfall_kh(ctx))
    assert ctx.solver.solve() == "optimal"
    return ctx.solver.var_value(device.floor_shortfall_kh(ctx))


_DRAWS = [12.0, 12.0, 0.0, 0.0]


@pytest.mark.parametrize("strategy", ["minimize_cost", "minimize_consumption", "balanced"])
def test_no_strategy_dips_below_the_least_shortfall(strategy: str) -> None:
    """Expensive heating tempts every strategy to save; none may exceed S*."""
    best = _least_shortfall_kh(_DRAWS, 45.0)
    assert best > 0.0
    bundle = _bundle(_DRAWS, strategy=strategy, prices=[2.0, 2.0, 0.1, 0.1])
    result = build_and_solve(bundle, _config())
    assert _replayed_shortfall_kh(result, _DRAWS, 45.0) <= best + 1e-3 + 1e-6


def test_a_time_limited_incumbent_still_bounds_the_shortfall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-solve that stops with an incumbent ("feasible") still bounds, by the incumbent value."""
    real = CBCSolverBackend.solve
    calls: list[int] = []

    def solve(self: CBCSolverBackend, time_limit_seconds: float = 59.0) -> str:
        calls.append(1)
        status = real(self, time_limit_seconds=time_limit_seconds)
        return "feasible" if len(calls) == 1 and status == "optimal" else status

    monkeypatch.setattr(CBCSolverBackend, "solve", solve)
    result = build_and_solve(_bundle(_DRAWS, prices=[2.0, 2.0, 0.1, 0.1]), _config())
    assert result.strategy_degraded is False
    assert _replayed_shortfall_kh(result, _DRAWS, 45.0) <= _least_shortfall_kh(_DRAWS, 45.0) + 1e-3 + 1e-6


def _care_config(import_limit_kw: float, care: bool = True, threshold: float = 0.0) -> MimirheimConfig:
    battery: dict = {
        "capacity_kwh": 10.0,
        "min_soc_kwh": 1.0,
        "charge_segments": [{"power_max_kw": 5.0, "efficiency": 1.0}],
        "discharge_segments": [{"power_max_kw": 5.0, "efficiency": 1.0}],
    }
    if care:
        battery["soc_ratchet"] = {"enabled": True, "target_pct": 97.0, "hold_hours": 0.0}
    return MimirheimConfig.model_validate(
        {
            "mqtt": {"host": "localhost", "client_id": "test"},
            "grid": {"import_limit_kw": import_limit_kw, "export_limit_kw": 20.0},
            "objectives": {"min_dispatch_gain_eur": threshold},
            "batteries": {"home": battery},
            "static_loads": {"base": {}},
            "thermal_boilers": {
                "dhw": {
                    "volume_liters": 200.0, "elec_power_kw": 3.0, "setpoint_c": 55.0, "min_temp_c": _MIN,
                    "standby_loss_kw_per_k": _LOSS, "ambient_temp_c": _ROOM, "use_draw_forecast": True,
                }
            },
        }
    )


def test_the_battery_care_probe_cannot_buy_battery_charge_with_tank_shortfall() -> None:
    """With 6 kW of import, a due balance charge (5 kW) and the heater (3 kW) compete.

    Minimising the battery gap alone would charge flat out and leave the
    heater off; the shortfall bound keeps the heater running first.
    """
    from datetime import timedelta

    t0_utc = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
    bundle = SolveBundle(
        solve_time_utc=t0_utc,
        horizon_prices=[0.2] * _N,
        horizon_export_prices=[0.1] * _N,
        horizon_confidence=[1.0] * _N,
        pv_forecast=[0.0] * _N,
        base_load_forecast=[0.0] * _N,
        battery_inputs={"home": {"soc_kwh": 1.0, "last_full_utc": t0_utc - timedelta(days=7) + timedelta(hours=1)}},
        thermal_boiler_inputs={"dhw": ThermalBoilerInputs(current_temp_c=45.0, draw_kw=_DRAWS)},
    )
    result = build_and_solve(bundle, _care_config(import_limit_kw=6.0))
    assert result.solve_status in ("optimal", "feasible")
    assert result.battery_care["home"].enforced_target_kwh is not None
    assert _replayed_shortfall_kh(result, _DRAWS, 45.0) <= _least_shortfall_kh(_DRAWS, 45.0) + 1e-3 + 1e-6


# ---------------------------------------------------------------------------
# Fault paths and the degraded fallback
# ---------------------------------------------------------------------------


def test_a_genuinely_infeasible_model_reports_infeasible() -> None:
    """A load the grid cannot supply stays an infeasible solve; the pre-solve does not hide it."""
    bundle = _bundle([0.5] * _N)
    bundle = bundle.model_copy(update={"base_load_forecast": [30.0] * _N})
    assert build_and_solve(bundle, _config()).solve_status == "infeasible"


def test_an_exhausted_fallback_reports_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    """No solution in the pre-solve and none in the fallback: the existing no-solution path."""

    def solve(self: CBCSolverBackend, time_limit_seconds: float = 59.0) -> str:
        self.last_status_proved_infeasible = False
        return "infeasible"

    monkeypatch.setattr(CBCSolverBackend, "solve", solve)
    assert build_and_solve(_bundle([40.0, 0.0, 0.0, 0.0]), _config()).solve_status == "infeasible"


@pytest.mark.parametrize("strategy", ["minimize_cost", "balanced"])
def test_a_degraded_dispatch_with_battery_support_survives_post_processing(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """With 2.5 kW of import and a 3 kW heater, heating needs the battery; nothing may undo that."""
    _wrap_solve(monkeypatch, "infeasible", proved=False)
    config = _care_config(import_limit_kw=2.5, care=False, threshold=1000.0)
    bundle = SolveBundle(
        strategy=strategy,
        solve_time_utc=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        horizon_prices=[0.2] * _N,
        horizon_export_prices=[0.1] * _N,
        horizon_confidence=[1.0] * _N,
        pv_forecast=[0.0] * _N,
        base_load_forecast=[0.0] * _N,
        battery_inputs={"home": {"soc_kwh": 8.0}},
        thermal_boiler_inputs={"dhw": ThermalBoilerInputs(current_temp_c=45.0, draw_kw=[40.0, 0.0, 0.0, 0.0])},
    )
    result = build_and_solve(bundle, config)
    assert result.strategy_degraded is True
    assert result.schedule[0].devices["dhw"].kw == pytest.approx(-3.0)
    assert all(step.grid_import_kw <= 2.5 + 1e-6 for step in result.schedule)
    assert apply_gain_threshold(result, bundle, config) is result
