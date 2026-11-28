"""Unit tests for scripts/dhw_benchmark/calib_heat.py: the calibration's heat input model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dhw_benchmark.calib_heat import (
    INVALID,
    CoilLaw,
    CopDomain,
    CopModel,
    compressor_electric_kw,
    estimate_aux_kw,
    heat_input,
    solve_coils,
)
from dhw_benchmark.inputs import TankRow

T0 = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
M = timedelta(minutes=1)
COP = CopModel(cop_ref=4.5, c1=0.1, c2=-0.15)
LAW = CoilLaw(eps_1=0.5, eps_2=0.5, flow_kw_per_k=1.0, first="A")


def _row(i: int = 0, **kw: object) -> TankRow:
    """An idle minute; overrides per field."""
    data: dict = dict(
        ts=T0 + i * M, t_top_c=50.0, t_lower_c=30.0, hp_electric_kw=0.06, mode="idle",
        compressor_on=False, element_kw=0.0, brine_in_c=None,
    )
    data.update(kw)
    return TankRow(**data)


def test_aux_power_is_the_median_of_quiet_idle_minutes() -> None:
    """Pumps and controls draw a little even when nothing heats."""
    rows = [_row(i, hp_electric_kw=0.05 + 0.01 * (i % 3)) for i in range(90)]
    rows.append(_row(90, mode="dhw", compressor_on=True, hp_electric_kw=1.6))
    aux, flagged = estimate_aux_kw(rows)
    assert aux == pytest.approx(0.06) and not flagged


def test_aux_power_without_enough_idle_minutes_is_zero_and_flagged() -> None:
    """Fewer than 60 quiet idle minutes give no estimate."""
    assert estimate_aux_kw([_row(i) for i in range(30)]) == (0.0, True)


def test_compressor_electricity_only_while_the_coil_heats() -> None:
    """Total minus element minus aux in dhw or legionella minutes with the compressor on."""
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=True, hp_electric_kw=1.66), 0.06) == pytest.approx(1.6)
    assert compressor_electric_kw(_row(mode="space_heating", compressor_on=True, hp_electric_kw=2.0), 0.06) == 0.0
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=False, hp_electric_kw=0.06), 0.06) == 0.0


def test_negative_residuals_unknown_state_and_missing_power_are_invalid() -> None:
    """Below -0.1 kW, an unknown compressor state or missing power while making hot
    water is invalid; exactly at the limit counts as zero."""
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=True, hp_electric_kw=1.2, element_kw=1.1), 0.2) == 0.0
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=True, hp_electric_kw=1.19, element_kw=1.1), 0.2) is INVALID
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=None, hp_electric_kw=1.6), 0.06) is INVALID
    assert compressor_electric_kw(_row(mode="dhw", compressor_on=True, hp_electric_kw=None), 0.06) is INVALID


def test_cop_is_written_around_the_reference_point() -> None:
    """COP = cop_ref + c1 (source - 10) + c2 (circuit - 45)."""
    assert COP.cop(source_c=10.0, circuit_c=45.0) == pytest.approx(4.5)
    assert COP.cop(source_c=12.0, circuit_c=50.0) == pytest.approx(4.5 + 0.2 - 0.75)


def test_coils_solve_circuit_temperature_and_cop_together() -> None:
    """The solved circuit temperature satisfies both the coil balance and the COP law."""
    p, source, t1, t2 = 1.6, 11.0, 35.0, 45.0
    solved = solve_coils(p, source, t1, t2, COP, LAW)
    d = 0.5 + 0.5 - 0.25
    q1 = 0.5 * 1.0 * (solved.t_in_c - t1)
    q2 = 0.5 * 1.0 * ((1 - 0.5) * solved.t_in_c + 0.5 * t1 - t2)
    assert q1 + q2 == pytest.approx(solved.coil.q_kw)
    assert solved.coil.q_kw == pytest.approx(p * COP.cop(source, solved.t_in_c))
    assert solved.cop == pytest.approx(COP.cop(source, solved.t_in_c))
    assert (solved.coil.alpha, solved.coil.beta_kw_per_k) == pytest.approx((0.5 / d, 0.25 / d))


def test_the_coil_order_decides_which_node_is_first() -> None:
    """Tank A first means the first coil node is tank A's lower node (0), else tank B's (2)."""
    assert solve_coils(1.6, 11.0, 35.0, 45.0, COP, LAW).coil.first_node == 0
    other = CoilLaw(eps_1=0.5, eps_2=0.5, flow_kw_per_k=1.0, first="B")
    assert solve_coils(1.6, 11.0, 45.0, 35.0, COP, other).coil.first_node == 2


def test_downstream_clamping_keeps_the_balance_consistent() -> None:
    """Outside the domain: the source is clamped first; a circuit temperature outside
    the range re-solves with the COP fixed at the edge, so Q = p x COP_edge."""
    domain = CopDomain(source_min=8.0, source_max=13.0, circuit_min=40.0, circuit_max=46.0)
    clamped = solve_coils(1.6, 20.0, 50.0, 55.0, COP, LAW, domain=domain)
    assert clamped.clamped
    edge = COP.cop(13.0, 46.0)
    assert clamped.cop == pytest.approx(edge)
    assert clamped.coil.q_kw == pytest.approx(1.6 * edge)
    inside = solve_coils(1.6, 11.0, 38.0, 41.0, COP, LAW, domain=CopDomain(8.0, 13.0, 30.0, 60.0))
    assert not inside.clamped


def test_heat_input_builds_the_coil_system_and_puts_the_element_in_tank_a_top() -> None:
    """A heating minute gives a coil system on the coil nodes; the element heats node 1;
    space heating and idle give nothing; an unknown heating minute is invalid."""
    temps = [35.0, 40.0, 45.0, 50.0]
    legionella = _row(mode="legionella", compressor_on=True, hp_electric_kw=7.86, element_kw=6.2, brine_in_c=11.0)
    result = heat_input(legionella, 0.06, temps, COP, LAW)
    assert result is not INVALID
    assert result.direct_kw == [0.0, 6.2, 0.0, 0.0]
    assert result.coil is not None and result.coil.q_kw > 0.0
    idle = heat_input(_row(), 0.06, temps, COP, LAW)
    assert idle.coil is None and idle.direct_kw == [0.0] * 4
    sh = heat_input(_row(mode="space_heating", compressor_on=True, hp_electric_kw=8.0, element_kw=6.0), 0.06, temps, COP, LAW)
    assert sh.coil is None and sh.direct_kw == [0.0] * 4
    assert heat_input(_row(mode="dhw", compressor_on=True, hp_electric_kw=1.6, brine_in_c=None), 0.06, temps, COP, LAW) is INVALID
    assert heat_input(_row(mode="dhw", compressor_on=False, element_kw=None), 0.06, temps, COP, LAW) is INVALID


def test_the_start_of_minute_cop_is_held_through_the_minute() -> None:
    """AJ counterexample: cop_ref 1, c2 -0.1, both coil nodes at 44.9 C. The COP solved at
    the minute's start is in range and fixes Q for the whole minute: the stored heat
    rises by exactly Q x 1 min, although the nodes pass 45 C during the minute."""
    from dhw_benchmark.plant import KWH_PER_LITRE_KELVIN, HeatInput, PlantParams, PlantState, TankPair, step_minute

    cop = CopModel(cop_ref=1.0, c1=0.0, c2=-0.1)
    law = CoilLaw(eps_1=1.0, eps_2=1.0, flow_kw_per_k=10.0, first="A")
    solved = solve_coils(1.0, 10.0, 44.9, 44.9, cop, law)
    assert solved.t_in_c == pytest.approx(45.0)
    assert solved.cop == pytest.approx(1.0) and not solved.clamped
    plant = PlantParams(
        volumes_l=[1.0, 90.0, 1.0, 100.0],
        tank_pairs=[TankPair(lower=0, upper=1, k_kw_per_k=0.0), TankPair(lower=2, upper=3, k_kw_per_k=0.0)],
        loss_kw_per_k=[0.0] * 4, room_temp_c=20.0, cold_temp_c=10.0,
    )
    start = PlantState(temps_c=[44.9, 60.0, 44.9, 60.0])
    end, _ = step_minute(plant, start, HeatInput(direct_kw=[0.0] * 4, coil=solved.coil), 0.0)
    stored = sum(v * KWH_PER_LITRE_KELVIN * (b - a) for v, a, b in zip(plant.volumes_l, start.temps_c, end.temps_c, strict=True))
    assert stored == pytest.approx(solved.coil.q_kw / 60.0)
    assert max(end.temps_c[0], end.temps_c[2]) > 45.0


def test_clipping_is_decided_per_minute_from_the_start_state() -> None:
    """A warming tank enters the clipped branch, and leaves it again once it cools."""
    domain = CopDomain(source_min=8.0, source_max=13.0, circuit_min=30.0, circuit_max=50.0)
    states = [(38.0, 40.0), (50.0, 52.0), (38.0, 40.0)]
    flags = [solve_coils(1.6, 11.0, t1, t2, COP, LAW, domain=domain).clamped for t1, t2 in states]
    assert flags == [False, True, False]
    edge = solve_coils(1.6, 11.0, 50.0, 52.0, COP, LAW, domain=domain)
    assert edge.cop == pytest.approx(COP.cop(11.0, 50.0))


def test_the_cop_is_capped_to_one_to_eight_with_a_consistent_balance() -> None:
    """Above 8 the minute runs at COP 8, below 1 at COP 1 (AL); the coil balance holds."""
    law = CoilLaw(eps_1=0.5, eps_2=0.5, flow_kw_per_k=1.0, first="A")
    high = solve_coils(1.6, 10.0, 25.0, 30.0, CopModel(cop_ref=8.5, c1=0.0, c2=-0.1), law)
    assert high.cop == 8.0 and high.capped and high.uncapped_cop > 8.0
    low = solve_coils(1.6, 10.0, 55.0, 60.0, CopModel(cop_ref=1.2, c1=0.0, c2=-0.1), law)
    assert low.cop == 1.0 and low.capped and low.uncapped_cop < 1.0
    for solved, t1, t2 in ((high, 25.0, 30.0), (low, 55.0, 60.0)):
        assert solved.coil.q_kw == pytest.approx(1.6 * solved.cop)
        q1 = 0.5 * (solved.t_in_c - t1)
        q2 = 0.5 * (0.5 * solved.t_in_c + 0.5 * t1 - t2)
        assert q1 + q2 == pytest.approx(solved.coil.q_kw)
    inside = solve_coils(1.6, 11.0, 35.0, 45.0, COP, LAW)
    assert not inside.capped and inside.uncapped_cop == inside.cop


def test_an_edge_cop_from_domain_clamping_is_capped_too() -> None:
    """Codex example (AL): the 30 C edge would give COP 8.5; the minute uses 8."""
    cop = CopModel(cop_ref=7.0, c1=0.0, c2=-0.1)
    law = CoilLaw(eps_1=1.0, eps_2=1.0, flow_kw_per_k=1.0, first="A")
    domain = CopDomain(source_min=5.0, source_max=15.0, circuit_min=30.0, circuit_max=50.0)
    solved = solve_coils(1.0, 10.0, 20.0, 20.0, cop, law, domain=domain)
    assert solved.cop == 8.0 and solved.coil.q_kw == pytest.approx(8.0)
    assert solved.t_in_c == pytest.approx(20.0 + 8.0)  # balance at Q = 8 with D = 1, K = 1


def test_validity_depends_on_aux_through_the_source_reading() -> None:
    """0.10 kW with aux 0.10 is no compressor heat and needs no source reading; with aux
    0.05 the compressor heats and the missing source makes the minute invalid."""
    from dhw_benchmark.calib_heat import heat_input_valid

    row = _row(mode="dhw", compressor_on=True, hp_electric_kw=0.10, brine_in_c=None)
    assert heat_input_valid(row, 0.10)
    assert heat_input(row, 0.10, [35.0, 40.0, 45.0, 50.0], COP, LAW) is not INVALID
    assert not heat_input_valid(row, 0.05)
    assert heat_input(row, 0.05, [35.0, 40.0, 45.0, 50.0], COP, LAW) is INVALID
