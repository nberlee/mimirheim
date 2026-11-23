"""ThermalBoilerDevice — models a thermal boiler (resistive or heat pump DHW) in the MILP.

This module implements the Device Protocol for a thermal boiler. In legacy mode
(``elec_power_kw``) an electric immersion heater and a heat pump domestic hot water
boiler use the same model: a binary on/off variable per step, a linear temperature
dynamics constraint (rise when on, cooling loss when off), and hard bounds on the
tank temperature, differing only in the COP. With a ``compressor`` the heat output
is fixed and the electric power follows the tank and source temperature, with a
heating ceiling per step.

The module does not import from ``mimirheim.io``. The current tank temperature is passed
as a ``ThermalBoilerInputs`` argument to ``add_constraints``. All solver interactions
go through ``ModelContext.solver``.
"""

from typing import Any

from mimirheim.config.schema import ThermalBoilerConfig
from mimirheim.core.bundle import ThermalBoilerInputs
from mimirheim.core.context import ModelContext

# Specific heat capacity of water: 4186 J/(kg·K) = 4186/3600 Wh/(kg·K).
# One litre of water weighs approximately 1 kg.
# Converting to kWh/(litre·K): 4186 / 3600 / 1000 ≈ 0.001163 kWh/(L·K).
# Used to convert electrical energy input (kWh) to temperature rise (K):
#   ΔT = kWh_thermal / (volume_L × _WATER_THERMAL_CAP_KWH_PER_LITRE_K)
_WATER_THERMAL_CAP_KWH_PER_LITRE_K: float = 4186 / 3600 / 1000  # kWh/(L·K)


class ThermalBoilerDevice:
    """Models a thermal boiler as a MILP sub-problem.

    Each instance corresponds to one entry in ``config.thermal_boilers``. The
    model builder creates one ``ThermalBoilerDevice`` per named boiler in the
    config, calls ``add_variables`` once, then calls ``add_constraints`` with
    the live tank temperature from the current ``SolveBundle``.

    The device exposes the following solver variables after ``add_variables``:

        _T_tank[t]    — Water temperature at the end of step t, in °C.
        _heater_on[t] — Binary; 1 = heater active during step t, 0 = off.
        _start[t]     — Binary sentinel (only when min_run_steps > 1); 1 if
                        the heater turns on at step t.
        _z[t]         — Compressor mode only, t >= 1 and a non-zero power
                        slope: heater_on[t] * T_tank[t-1], linearised exactly.

    With ``config.compressor`` set, ``_heater_on`` is the compressor's on/off
    state and the electric power depends on the start-of-step temperature and
    the source temperature (see ``_add_compressor_constraints``).

    Attributes:
        name: Device name matching the key in ``config.thermal_boilers``.
        config: Static thermal boiler configuration.
    """

    def __init__(self, name: str, config: ThermalBoilerConfig) -> None:
        """Initialise the ThermalBoilerDevice.

        Args:
            name: Device name, matching the key in ``MimirheimConfig.thermal_boilers``.
            config: Validated static configuration for this thermal boiler.
        """
        if config.element is not None:
            # The element model is added in a later change of this feature;
            # until then a tank configured with one is refused here rather
            # than built without the element.
            raise NotImplementedError(
                f"Thermal boiler {name!r}: element heat sources are not supported yet."
            )
        self.name = name
        self.config = config

        # Solver variables populated by add_variables.
        self._T_tank: dict[int, Any] = {}
        self._heater_on: dict[int, Any] = {}
        # _start[t] = 1 when the heater transitions from off (t-1) to on (t).
        # Only populated when min_run_steps > 1.
        self._start: dict[int, Any] = {}
        # Compressor mode only: _z[t] = heater_on[t] * T_tank[t-1] for t >= 1,
        # and the electric power expression per step (see add_constraints).
        self._z: dict[int, Any] = {}
        self._elec_kw: dict[int, Any] = {}

        self._dt: float = 0.25  # set from ctx in add_variables

    def add_variables(self, ctx: ModelContext) -> None:
        """Declare all MILP variables for this thermal boiler.

        For each time step t in ctx.T:

        - ``_T_tank[t]``: Water temperature at the end of step t, in degrees
          Celsius. Lower bound: ``min_temp_c − 5`` (small numerical slack below
          the hard bound, which is enforced as a constraint). Upper bound:
          ``setpoint_c + 5`` (corresponding slack above the setpoint). These
          loose variable bounds prevent the LP from becoming infeasible due to
          numerical noise; the hard temperature bounds are enforced explicitly
          in ``add_constraints``.

        - ``_heater_on[t]``: Binary variable. 1 = the heating element or heat
          pump compressor is active during step t. 0 = off. The on/off state
          determines the thermal power added to the tank at this step.

        If ``config.min_run_steps > 1``, for each t from 1 to T−1:

        - ``_start[t]``: Binary sentinel. Set to 1 by the solver when the
          heater transitions from off to on (heater_on[t]=1, heater_on[t-1]=0).
          Used by the minimum run constraint to enforce that once started, the
          heater stays on for at least ``min_run_steps`` consecutive steps.
          Not created for t=0 (no previous step to compare) or when the minimum
          run length is 0 or 1 (no minimum run to enforce).

        Args:
            ctx: The current solve context. Variables are registered on
                ``ctx.solver``.
        """
        self._dt = ctx.dt
        cfg = self.config

        for t in ctx.T:
            if cfg.compressor is None:
                # T_tank[t]: tank temperature at the end of step t.
                # A small slack on the bounds (±5°C) keeps the LP feasible under
                # minor numerical noise while the real bounds are handled as
                # explicit solver constraints in add_constraints.
                self._T_tank[t] = ctx.solver.add_var(
                    lb=cfg.min_temp_c - 5.0,
                    ub=cfg.setpoint_c + 5.0,
                )
            else:
                # T_tank[t] in °C, compressor mode. No fixed upper bound: a tank
                # left hot by other heating may start above every limit and
                # cools passively. add_constraints adds the upper bound once
                # the starting temperature is known. See IMPLEMENTATION_DETAILS.md
                # §8, subsection "Thermal boiler compressor".
                self._T_tank[t] = ctx.solver.add_var(lb=cfg.tank_min_temp_bound_c())

            # heater_on[t]: binary on/off for the heating element.
            self._heater_on[t] = ctx.solver.add_var(lb=0.0, ub=1.0, integer=True)

            if cfg.compressor is not None and t >= 1 and cfg.compressor.curve_slope_kw_per_k() != 0.0:
                # z[t]: heater_on[t] * T_tank[t-1], in °C when on and 0 when off.
                # Needed because the compressor's electric power depends on the
                # start-of-step temperature only while it runs; bounds are set
                # by the exact linearisation in add_constraints.
                self._z[t] = ctx.solver.add_var(lb=-1e30)

        if cfg.min_run_steps > 1:
            # start[t] sentinels: needed to detect on-transitions and enforce
            # the minimum consecutive run length.
            for t in range(1, len(ctx.T)):
                self._start[t] = ctx.solver.add_var(lb=0.0, ub=1.0, integer=True)

    def add_constraints(self, ctx: ModelContext, inputs: ThermalBoilerInputs) -> None:
        """Add all MILP constraints for this thermal boiler.

        Adds, per step: the temperature dynamics equation (``T_tank[t] =
        T_tank[t-1] - cool_per_step + heat_rise_per_step * heater_on[t]``,
        with ``inputs.current_temp_c`` as the initial condition at ``t=0``),
        hard bounds (``min_temp_c <= T_tank[t] <= setpoint_c``, enforced as
        explicit constraints because the dynamics can otherwise push past
        tightened variable bounds), and, when ``min_run_steps > 1``, the
        minimum run-time constraint. See IMPLEMENTATION_DETAILS.md §8,
        subsection "Thermal boiler and DHW tank dynamics", for the tank model
        derivation and the start-sentinel run-length mechanism.

        Args:
            ctx: The current solve context.
            inputs: Live temperature reading for this boiler. ``current_temp_c``
                is the initial tank temperature for the first step.
        """
        cfg = self.config
        if cfg.compressor is not None:
            self._add_compressor_constraints(ctx, inputs)
            self._add_min_run_constraints(ctx)
            return

        # Thermal parameters derived from static config (plain floats, not
        # solver variables): cool_per_step is the unconditional temperature
        # drop per step when off (insulation losses plus hot water draws);
        # heat_rise_per_step is the rise per step when on (elec_power_kw *
        # cop * dt / thermal_cap_kwh_per_k — cop=1.0 for resistive elements,
        # >1 for heat pumps).
        thermal_cap_kwh_per_k = cfg.volume_liters * _WATER_THERMAL_CAP_KWH_PER_LITRE_K
        cool_per_step = cfg.cooling_rate_k_per_hour * ctx.dt
        heat_rise_per_step = cfg.elec_power_kw * cfg.effective_cop() * ctx.dt / thermal_cap_kwh_per_k

        for t in ctx.T:
            # prior_temp: inputs.current_temp_c at t=0, else the previous
            # step's solver variable.
            prior_temp: Any = inputs.current_temp_c if t == 0 else self._T_tank[t - 1]
            ctx.solver.add_constraint(
                self._T_tank[t]
                == prior_temp - cool_per_step + heat_rise_per_step * self._heater_on[t]
            )

            # Hard temperature bounds as explicit constraints, not variable
            # bounds — the dynamics can otherwise push past a tightened bound.
            ctx.solver.add_constraint(self._T_tank[t] >= cfg.min_temp_c)
            ctx.solver.add_constraint(self._T_tank[t] <= cfg.setpoint_c)

        self._add_min_run_constraints(ctx)

    def _add_min_run_constraints(self, ctx: ModelContext) -> None:
        """Add the minimum run-length constraints on ``_heater_on``."""
        cfg = self.config
        # Minimum run length (e.g. a compressor that must run in blocks). See
        # IMPLEMENTATION_DETAILS.md §8, subsection "Thermal boiler and DHW
        # tank dynamics", for the start-sentinel mechanism.
        if cfg.min_run_steps > 1:
            for t in range(1, len(ctx.T)):
                ctx.solver.add_constraint(
                    self._start[t] >= self._heater_on[t] - self._heater_on[t - 1]
                )
                ctx.solver.add_constraint(self._start[t] <= self._heater_on[t])

                for tau in range(1, cfg.min_run_steps):
                    if t + tau < len(ctx.T):
                        ctx.solver.add_constraint(
                            self._heater_on[t + tau] >= self._start[t]
                        )

    def _add_compressor_constraints(self, ctx: ModelContext, inputs: ThermalBoilerInputs) -> None:
        """Add dynamics, bounds, heating ceiling and electric power for a compressor tank.

        Within a step, heat is added first and the cooling loss is removed
        afterwards, so the highest temperature in step t is
        ``T_tank[t-1] + rise`` and the lowest is the lower endpoint. See
        IMPLEMENTATION_DETAILS.md §8, subsection "Thermal boiler compressor".

        Args:
            ctx: The current solve context.
            inputs: Live readings. ``current_temp_c`` is the starting
                temperature; ``source_temp_c`` (None means the reference
                temperature) shifts the electric power.
        """
        cfg = self.config
        comp = cfg.compressor
        assert comp is not None
        thermal_cap_kwh_per_k = cfg.volume_liters * _WATER_THERMAL_CAP_KWH_PER_LITRE_K
        cool_per_step = cfg.cooling_rate_k_per_hour * ctx.dt
        # rise: temperature gain in K over one step of compressor operation.
        rise = comp.heat_output_kw * ctx.dt / thermal_cap_kwh_per_k
        # ceiling: highest temperature heating may reach within a step, in °C.
        # The setpoint is the heating target; the source limit is physical.
        ceiling = min(comp.max_temp_c, cfg.setpoint_c)
        t0 = inputs.current_temp_c
        # lower, upper: bounds on every T_tank[t] in °C. Heating only happens
        # below the ceiling, so the tank never exceeds max(ceiling, start).
        lower = cfg.min_temp_c
        upper = max(cfg.tank_max_temp_c(), t0)
        slope = comp.curve_slope_kw_per_k()
        # Electric power while running at tank temperature T:
        #   elec_kw(T) = intercept + slope * T, with the source shift folded in.
        intercept = comp.elec_kw(0.0, inputs.source_temp_c)

        for t in ctx.T:
            on = self._heater_on[t]
            prior: Any = t0 if t == 0 else self._T_tank[t - 1]
            # Tank temperature after heating and cooling in step t.
            ctx.solver.add_constraint(self._T_tank[t] == prior - cool_per_step + rise * on)
            # Hard floor, as in legacy mode; the upper bound follows from the
            # ceiling and keeps the linearisation below exact.
            ctx.solver.add_constraint(self._T_tank[t] >= lower)
            ctx.solver.add_constraint(self._T_tank[t] <= upper)

            if t == 0:
                # Start-of-step temperature is a measured constant: running is
                # allowed only when the step peak stays at or below the
                # ceiling, and the power is exact without linearisation. The
                # temperature is clamped to the validated range, where the
                # power is known to be positive.
                if t0 + rise > ceiling:
                    ctx.solver.add_constraint(on == 0)
                t0_clamped = min(max(t0, cfg.tank_min_temp_bound_c()), cfg.tank_max_temp_c())
                self._elec_kw[t] = comp.elec_kw(t0_clamped, inputs.source_temp_c) * on
                continue

            # Peak ceiling: T_tank[t-1] + rise <= ceiling while on. When off,
            # the big-M term (the largest possible excess) relaxes it, so a hot
            # tank can still cool passively.
            big_m = max(0.0, upper + rise - ceiling)
            ctx.solver.add_constraint(prior + rise <= ceiling + big_m * (1 - on))

            if t not in self._z:
                # slope == 0: the power does not depend on the tank temperature.
                self._elec_kw[t] = intercept * on
                continue
            z = self._z[t]
            # Exact linearisation of z = on * T_tank[t-1] for a binary on and
            # T_tank[t-1] in [lower, upper]: on = 0 forces z = 0, on = 1
            # forces z = T_tank[t-1]. See IMPLEMENTATION_DETAILS.md §8,
            # subsection "Thermal boiler compressor".
            ctx.solver.add_constraint(z >= lower * on)
            ctx.solver.add_constraint(z <= upper * on)
            ctx.solver.add_constraint(z >= prior - upper * (1 - on))
            ctx.solver.add_constraint(z <= prior - lower * (1 - on))
            self._elec_kw[t] = intercept * on + slope * z

    def planned_temp_c(self, ctx: ModelContext, t: int) -> float:
        """Planned tank temperature at the end of step t after a solve, in °C.

        Args:
            ctx: The solved model context.
            t: Zero-based time step index.

        Returns:
            The solved value of ``_T_tank[t]`` in degrees Celsius.
        """
        return float(ctx.solver.var_value(self._T_tank[t]))

    def net_power(self, t: int) -> Any:
        """Net AC power drawn from the home bus at step t, in kW.

        The boiler draws ``elec_power_kw`` from the AC bus when active and
        zero when off. The sign follows the Device Protocol convention:
        positive = producing power (injection), negative = consuming power
        (draw). A heater is always a consumer, so net_power is always <= 0.

        The return value is a solver expression because ``_heater_on[t]`` is a
        solver variable. The expression evaluates to exactly ``-elec_power_kw``
        when ``heater_on[t]=1`` and to 0 when ``heater_on[t]=0``.

        Args:
            t: Zero-based time step index.

        With a compressor, the draw is the compressor's electric power at the
        start-of-step tank temperature (see ``_add_compressor_constraints``).

        Returns:
            Solver expression for ``−elec_power_kw × heater_on[t]``, or the
            negated compressor power expression.
        """
        if self.config.compressor is not None:
            return -self._elec_kw[t]
        return -self.config.elec_power_kw * self._heater_on[t]

    def objective_terms(self, t: int) -> list[Any]:
        """Return objective cost terms for time step t.

        The only optional cost term is the wear (cycling) cost, which penalises
        electrical consumption to discourage unnecessary short cycles. For
        resistive elements, ``wear_cost_eur_per_kwh`` should be 0.0. For heat
        pump compressors, a small positive value (e.g. 0.01 EUR/kWh) adds an
        energy-cost-independent penalty on top of the minimum run constraint.

        When ``wear_cost_eur_per_kwh`` is zero (the default), this method
        returns an empty list and contributes nothing to the objective.

        Args:
            t: Zero-based time step index.

        Returns:
            List containing zero or one solver expressions.
        """
        if self.config.wear_cost_eur_per_kwh <= 0.0:
            return []
        if self.config.compressor is not None:
            return [self.config.wear_cost_eur_per_kwh * self._elec_kw[t] * self._dt]
        return [
            self.config.wear_cost_eur_per_kwh
            * self.config.elec_power_kw
            * self._heater_on[t]
            * self._dt
        ]

    def terminal_soc_var(self, ctx: ModelContext) -> Any | None:
        """Return a solver expression proportional to the thermal energy stored at T-1.

        The terminal value mechanism in ``ObjectiveBuilder`` multiplies the
        returned expression by ``−avg_import_price / dt``, creating a reward
        for leaving the tank warm at the end of the horizon.

        The expression represents the equivalent electrical kWh stored above the
        minimum temperature at the last step:

            (T_tank[T−1] − min_temp_c) × thermal_cap_kwh_per_k / cop

        Dividing by COP converts thermal kWh to electrical kWh equivalent,
        so the terminal value uses the same units as battery SOC (kWh_electric).
        This ensures an equal average import price coefficient applies to all
        storage devices regardless of their COP.

        Without this terminal value, the solver would drain the tank to
        ``min_temp_c`` at the end of the horizon (free heat storage is treated
        as worthless in a finite-horizon model). With the terminal value, the
        solver preserves tank temperature when the expected refill cost exceeds
        the export price.

        Args:
            ctx: The current solve context. Used to identify the last step.

        Returns:
            Solver expression in kWh_electric, or None if add_variables has
            not been called yet.
        """
        if ctx.T[-1] not in self._T_tank:
            return None
        thermal_cap = self.config.volume_liters * _WATER_THERMAL_CAP_KWH_PER_LITRE_K
        # (T_tank[T-1] - min_temp_c) gives the temperature surplus above the floor,
        # in K. Multiplying by thermal_cap converts to kWh_thermal. Dividing by cop
        # converts to the equivalent kWh_electric needed to produce that heat.
        factor = thermal_cap / self.config.effective_cop()
        return (self._T_tank[ctx.T[-1]] - self.config.min_temp_c) * factor
