"""Unit tests for scripts/dhw_benchmark/plant.py: the tank model and draw reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from dhw_benchmark.plant import (
    KWH_PER_LITRE_KELVIN,
    CoilSystem,
    HeatInput,
    PlantParams,
    PlantState,
    TankPair,
    _minimise,
    reconstruct_draws,
    step_minute,
)

C = KWH_PER_LITRE_KELVIN
Provider = Callable[[int, PlantState], HeatInput]


def _params(**overrides: object) -> PlantParams:
    """A two-node 380 L chain, one tank, no exchange or losses."""
    data: dict = dict(
        volumes_l=[190.0, 190.0], tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.0)],
        loss_kw_per_k=[0.0, 0.0], room_temp_c=18.0, cold_temp_c=10.0,
    )
    data.update(overrides)
    return PlantParams(**data)


def _four(**overrides: object) -> PlantParams:
    """Two tanks of two nodes each (90/90/100/100 L), no exchange or losses."""
    data: dict = dict(
        volumes_l=[90.0, 90.0, 100.0, 100.0],
        tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.0), TankPair(lower=2, upper=3, k_kw_per_k=0.0)],
        loss_kw_per_k=[0.0] * 4, room_temp_c=18.0, cold_temp_c=10.0,
    )
    data.update(overrides)
    return PlantParams(**data)


def _none(n: int) -> HeatInput:
    """No heat in an n-node plant."""
    return HeatInput(direct_kw=[0.0] * n)


def _no_heat(n: int) -> Provider:
    """A heat provider that never heats."""

    def provider(minute: int, state: PlantState) -> HeatInput:
        return _none(n)

    return provider


def _energy(params: PlantParams, state: PlantState) -> float:
    """Heat stored above the cold-water temperature, in kWh."""
    return sum(v * (t - params.cold_temp_c) * C for v, t in zip(params.volumes_l, state.temps_c, strict=True))


def test_params_reject_invalid_input() -> None:
    """Volumes positive, lists matching, pairs valid and disjoint, no unknown fields."""
    with pytest.raises(ValidationError):
        _params(volumes_l=[190.0, 0.0])
    with pytest.raises(ValidationError):
        _params(loss_kw_per_k=[0.0])
    with pytest.raises(ValidationError):
        _params(tank_pairs=[TankPair(lower=0, upper=2, k_kw_per_k=0.0)])
    with pytest.raises(ValidationError):
        _four(tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.0), TankPair(lower=1, upper=2, k_kw_per_k=0.0)])
    with pytest.raises(ValidationError):
        _params(colour="blue")


def test_cold_water_temperature_is_clipped() -> None:
    """The inlet temperature is kept in 5-20 C, so 40 - T_cold >= 20 K."""
    assert _params(cold_temp_c=2.0).cold_temp_c == 5.0
    assert _params(cold_temp_c=38.0).cold_temp_c == 20.0


def test_direct_heat_is_stored_exactly() -> None:
    """An hour of 2 kW into the inlet node stores 2 kWh there."""
    params = _params()
    state = PlantState(temps_c=[30.0, 50.0])
    before = _energy(params, state)
    for _ in range(60):
        state, _ = step_minute(params, state, HeatInput(direct_kw=[2.0, 0.0]), draw_kwh=0.0)
    assert _energy(params, state) - before == pytest.approx(2.0)
    assert state.temps_c[1] == pytest.approx(50.0)


def test_a_coil_system_stores_its_output_exactly_and_never_overshoots() -> None:
    """Codex's counterexample: eps1 = eps2 = 1, K = 10 kW/K (alpha = 1, beta = 10),
    p = 1 kW, COP 2, tanks at 60 and 30 C. The exact solution stores its output and
    keeps both coil nodes between the starting temperatures."""
    params = _four()
    state = PlantState(temps_c=[60.0, 60.0, 30.0, 30.0])
    before = _energy(params, state)
    coil = CoilSystem(first_node=0, second_node=2, q_kw=2.0, alpha=1.0, beta_kw_per_k=10.0)
    after, _ = step_minute(params, state, HeatInput(direct_kw=[0.0] * 4, coil=coil), draw_kwh=0.0)
    assert _energy(params, after) - before == pytest.approx(2.0 / 60.0)
    assert 30.0 <= after.temps_c[0] <= 60.0 and 30.0 <= after.temps_c[2] <= 60.0


def test_a_coil_system_without_exchange_splits_by_alpha() -> None:
    """With beta = 0 the coils take fixed shares alpha and 1 - alpha of the output."""
    params = _four()
    coil = CoilSystem(first_node=2, second_node=0, q_kw=6.0, alpha=0.25, beta_kw_per_k=0.0)
    # Upper nodes warmer, so heating the lower nodes does not invert a tank.
    start = PlantState(temps_c=[40.0, 45.0, 40.0, 45.0])
    after, _ = step_minute(params, start, HeatInput(direct_kw=[0.0] * 4, coil=coil), draw_kwh=0.0)
    assert after.temps_c[2] - 40.0 == pytest.approx(0.25 * 6.0 / 60.0 / (100.0 * C))
    assert after.temps_c[0] - 40.0 == pytest.approx(0.75 * 6.0 / 60.0 / (90.0 * C))


def test_standby_loss_cools_towards_the_room() -> None:
    """Each node loses loss * (T - T_room) over the minute."""
    params = _params(loss_kw_per_k=[0.002, 0.002])
    state, _ = step_minute(params, PlantState(temps_c=[50.0, 50.0]), _none(2), draw_kwh=0.0)
    expected = 50.0 - 0.002 * (50.0 - 18.0) / 60.0 / (190.0 * C)
    assert state.temps_c == pytest.approx([expected, expected])


def test_exchange_within_a_tank_is_exact_and_never_overshoots() -> None:
    """Even a huge conductance only brings the two nodes towards their
    capacity-weighted mean, conserving energy, with unequal volumes too."""
    params = PlantParams(
        volumes_l=[50.0, 150.0], tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=100.0)],
        loss_kw_per_k=[0.0, 0.0], room_temp_c=18.0, cold_temp_c=10.0,
    )
    state = PlantState(temps_c=[20.0, 60.0])
    before = _energy(params, state)
    after, _ = step_minute(params, state, _none(2), draw_kwh=0.0)
    mean = (50.0 * 20.0 + 150.0 * 60.0) / 200.0
    assert _energy(params, after) == pytest.approx(before)
    assert 20.0 <= after.temps_c[0] <= mean + 1e-9
    assert mean - 1e-9 <= after.temps_c[1] <= 60.0


def test_an_inverted_tank_mixes_and_tanks_do_not_mix_with_each_other() -> None:
    """A tank whose lower node is warmer mixes to its mean in the minute (hot water
    rises); a warm first tank does not mix into a cold second tank."""
    after, _ = step_minute(_four(), PlantState(temps_c=[60.0, 40.0, 20.0, 25.0]), _none(4), draw_kwh=0.0)
    assert after.temps_c[0] == pytest.approx(50.0) and after.temps_c[1] == pytest.approx(50.0)
    assert after.temps_c[2:] == pytest.approx([20.0, 25.0])


def test_a_hot_outlet_delivers_the_full_draw_and_loses_exactly_that_heat() -> None:
    """At 55 C the household mixes down: 0.5 kWh requested is 0.5 kWh delivered,
    and the stored heat drops by exactly 0.5 kWh (cold water replaces the outflow)."""
    params = _params()
    state = PlantState(temps_c=[30.0, 55.0])
    before = _energy(params, state)
    after, draw = step_minute(params, state, _none(2), draw_kwh=0.5)
    assert (draw.requested_kwh, draw.delivered_kwh, draw.unmet_kwh) == pytest.approx((0.5, 0.5, 0.0))
    assert before - _energy(params, after) == pytest.approx(0.5)


def test_a_lukewarm_outlet_delivers_part_of_the_draw() -> None:
    """At 30 C (cold 10 C) the household takes V40 = q / (c * 30 K) at 30 C:
    delivered = c * V40 * 20 K, so two thirds; the rest is unmet."""
    _, draw = step_minute(_params(), PlantState(temps_c=[30.0, 30.0]), _none(2), draw_kwh=0.3)
    assert draw.delivered_kwh == pytest.approx(0.2)
    assert draw.unmet_kwh == pytest.approx(0.1)


def test_an_outlet_at_cold_water_temperature_delivers_nothing() -> None:
    """At the cold-water temperature the outlet gives no useful heat: all unmet."""
    _, draw = step_minute(_params(), PlantState(temps_c=[10.0, 10.0]), _none(2), draw_kwh=0.3)
    assert (draw.delivered_kwh, draw.unmet_kwh) == pytest.approx((0.0, 0.3))


def test_a_draw_larger_than_the_tank_stops_at_one_tank_volume() -> None:
    """With unequal nodes (100 L and 280 L) a huge request moves exactly the
    380 L once, in shifts of at most 100 L: 100, 100, 100, then 80. The outlet
    crosses below 40 C before the last shift, and the rest is unmet."""
    params = _params(volumes_l=[100.0, 280.0])
    huge = 1000.0
    after, draw = step_minute(params, PlantState(temps_c=[60.0, 60.0]), _none(2), draw_kwh=huge)
    outlet = [60.0]
    for litres in (100.0, 100.0, 100.0):
        outlet.append((outlet[-1] * (280.0 - litres) + (60.0 if len(outlet) == 1 else 10.0) * litres) / 280.0)
    expected_outlet = (outlet[3] * 200.0 + 10.0 * 80.0) / 280.0
    assert after.temps_c == pytest.approx([10.0, expected_outlet])
    delivered = C * (100 * (60 - 10) + 100 * (outlet[1] - 10) + 100 * (outlet[2] - 10) + 80 * (outlet[3] - 10))
    assert draw.delivered_kwh == pytest.approx(delivered)
    assert draw.delivered_kwh + draw.unmet_kwh == pytest.approx(huge)


def test_heating_comes_before_the_standby_loss() -> None:
    """The loss of a heating minute is computed from the heated temperature."""
    params = _params(loss_kw_per_k=[0.01, 0.0])
    cap = 190.0 * C
    heated = 40.0 + 6.0 / 60.0 / cap
    expected = heated - 0.01 * (heated - 18.0) / 60.0 / cap
    # The upper node is warmer, so the heated lower node does not invert the tank.
    after, _ = step_minute(params, PlantState(temps_c=[40.0, 50.0]), HeatInput(direct_kw=[6.0, 0.0]), draw_kwh=0.0)
    assert after.temps_c[0] == pytest.approx(expected)


def test_plug_flow_moves_the_cold_front_up_the_chain() -> None:
    """A draw pulls cold water into the inlet node first; the outlet stays hot longer."""
    after, _ = step_minute(_four(), PlantState(temps_c=[55.0] * 4), _none(4), draw_kwh=1.0)
    assert after.temps_c[0] < after.temps_c[1] <= after.temps_c[2] <= after.temps_c[3]
    assert after.temps_c[3] == pytest.approx(55.0)


def _simulate(
    params: PlantParams, start: PlantState, provider: Provider, draws: list[float], lower: int = 0, top: int = -1
) -> list[tuple[float, float]]:
    """Probe readings at each quarter end of a simulation with known draws."""
    state = start
    probes = []
    for q, draw in enumerate(draws):
        for m in range(15):
            state, _ = step_minute(params, state, provider(q * 15 + m, state), draw_kwh=draw / 15)
        probes.append((state.temps_c[lower], state.temps_c[top]))
    return probes


def test_reconstruction_recovers_known_draws() -> None:
    """Simulate known draws, then recover them from the probes at each quarter end:
    non-negative and close to the truth."""
    params = _params(loss_kw_per_k=[0.001, 0.001], tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.01)])
    draws = [0.0, 1.2, 0.0, 0.4, 2.0, 0.0, 0.0, 0.8]

    def provider(minute: int, state: PlantState) -> HeatInput:
        return HeatInput(direct_kw=[3.0 if 2 <= minute // 15 <= 4 else 0.0, 0.0])

    probes = _simulate(params, PlantState(temps_c=[35.0, 55.0]), provider, draws)
    result = reconstruct_draws(params, PlantState(temps_c=[35.0, 55.0]), provider, probes, lower_node=0, top_node=1)
    assert all(d >= 0.0 for d in result.draws_kwh)
    assert result.draws_kwh == pytest.approx(draws, abs=0.02)
    assert max(r for r in result.residual_k if r is not None) < 0.05


def test_reconstruction_recomputes_coil_heat_along_each_candidate() -> None:
    """With a state-dependent coil system in four nodes, observed at nodes 0 and 3,
    the draws are recovered: each candidate gets heat computed on its own trajectory."""
    params = _four(tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.02), TankPair(lower=2, upper=3, k_kw_per_k=0.02)])

    def provider(minute: int, state: PlantState) -> HeatInput:
        on = 15 <= minute < 45
        # The output depends on the state, as a COP depending on the tank would.
        q = 4.0 + 0.05 * (45.0 - state.temps_c[2]) if on else 0.0
        coil = CoilSystem(first_node=0, second_node=2, q_kw=q, alpha=0.6, beta_kw_per_k=0.3) if on else None
        return HeatInput(direct_kw=[0.0] * 4, coil=coil)

    start = PlantState(temps_c=[30.0, 38.0, 44.0, 50.0])
    draws = [0.5, 0.0, 1.0, 0.3]
    probes = _simulate(params, start, provider, draws, lower=0, top=3)
    result = reconstruct_draws(params, start, provider, probes, lower_node=0, top_node=3)
    assert result.draws_kwh == pytest.approx(draws, abs=0.03)


def test_reconstruction_recovers_a_draw_from_a_lukewarm_tank() -> None:
    """Requested heat can exceed the heat stored when the outlet is lukewarm:
    a 5 kWh demand from a tank at 15/20 C (cold 10 C) is still recovered."""
    start = PlantState(temps_c=[15.0, 20.0])
    probes = _simulate(_params(), start, _no_heat(2), [5.0])
    result = reconstruct_draws(_params(), start, _no_heat(2), probes, lower_node=0, top_node=1)
    assert result.draws_kwh == pytest.approx([5.0], abs=0.05)


def test_reconstruction_recovers_a_draw_larger_than_one_tank_volume() -> None:
    """The volume cap is per minute: a 20 kWh quarter demand from a lukewarm tank
    moves more than one tank volume over the quarter and is still recovered."""
    start = PlantState(temps_c=[15.0, 20.0])
    probes = _simulate(_params(), start, _no_heat(2), [20.0])
    result = reconstruct_draws(_params(), start, _no_heat(2), probes, lower_node=0, top_node=1)
    assert result.draws_kwh == pytest.approx([20.0], rel=0.01)


def test_reconstruction_bridges_a_missing_probe_with_the_fallback() -> None:
    """A quarter with a missing probe uses the caller's fallback draw to advance the
    state, is marked as not fitted, and the next quarter is fitted again."""
    start = PlantState(temps_c=[30.0, 55.0])
    probes = _simulate(_params(), start, _no_heat(2), [0.6, 0.9, 0.3])
    probes[1] = (None, probes[1][1])
    result = reconstruct_draws(
        _params(), start, _no_heat(2), probes, lower_node=0, top_node=1, fallback_draws_kwh=[None, 0.9, None]
    )
    assert result.fitted == [True, False, True]
    assert result.draws_kwh == pytest.approx([0.6, 0.9, 0.3], abs=0.02)
    assert result.residual_k[1] is None


def test_reconstruction_refuses_a_missing_probe_without_a_fallback() -> None:
    """Without a fallback the caller must end the segment: an error, not a guess."""
    with pytest.raises(ValueError, match="end the segment"):
        reconstruct_draws(_params(), PlantState(temps_c=[30.0, 55.0]), _no_heat(2), [(None, 50.0)], lower_node=0, top_node=1)


def test_reconstruction_rejects_a_wrong_fallback_length_or_probe_node() -> None:
    """Fallbacks need one value per quarter; probe nodes must exist."""
    with pytest.raises(ValueError):
        reconstruct_draws(_params(), PlantState(temps_c=[30.0, 55.0]), _no_heat(2), [(30.0, 55.0)],
                          lower_node=0, top_node=1, fallback_draws_kwh=[])
    with pytest.raises(ValueError):
        reconstruct_draws(_params(), PlantState(temps_c=[30.0, 55.0]), _no_heat(2), [(30.0, 55.0)],
                          lower_node=0, top_node=5)


def test_minimise_takes_the_left_end_of_a_plateau() -> None:
    """When the objective is flat beyond some point, the smallest minimiser wins."""
    assert _minimise(lambda x: max(1.0 - x, 0.0) ** 2, 0.0, 100.0) == pytest.approx(1.0, abs=1e-3)


def test_minimise_prefers_the_smaller_of_equal_minima() -> None:
    """Two exact zeros: the smaller argument is returned."""
    assert _minimise(lambda x: min((x - 0.3) ** 2, (x - 5.0) ** 2), 0.0, 100.0) == pytest.approx(0.3, abs=1e-3)
