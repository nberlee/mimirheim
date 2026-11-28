"""Run the tank calibration and write ``calibration.json`` (benchmark ticket 03).

Steps, on the 1-minute tank data of one installation:

1. Draw-free cooling and heating windows (``calib_windows``), auxiliary
   power from quiet idle minutes, and the cold-water temperature (the raw
   minimum of the lower probe outside legionella windows, used if within
   5-20 C, otherwise 12 C flagged; spec §15.2).
2. For each coil order: the full-data fit from the five fixed starts and
   leave-one-ISO-week-out cross-validation (``calib_cv``). The order with
   the lower pooled held-out top-probe error is central; the other is the
   coil-order sensitivity.
3. The label from the central model's cross-validation and its local
   Jacobian diagnostic (plan amendments AD and AM).
4. The thermostat check (``calib_runs``): the stop rule selected on all
   draw-free complete runs with the central parameters, and per fold on
   training runs only; the legionella rule; the start diagnostic.
5. Appliance limits from complete runs (under commanded targets).

``calibration.json`` also carries everything downstream needs to apply the
model consistently: the domain for clamping (source and supply
temperature), the source-temperature fallback chain (plan amendment I),
the COP cap statistics, the thresholds, the parameter bounds, the
assumptions, and the reduced-scope items marked ``not_assessed``
(amendment AM).

This module orchestrates; the computations live in the ``calib_*``
modules. Run it as
``python -m dhw_benchmark.calibrate --tank tank.csv --out calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, get_args

import numpy as np

from dhw_benchmark.calib_cv import (
    Acceptance,
    CvResult,
    choose_coil_order,
    cross_validate,
    fold_plan,
    label,
    split_invalid,
)
from dhw_benchmark.calib_fit import (
    BOUNDS,
    STARTS,
    STRAINED_SHARE,
    FitResult,
    FitSettings,
    FourNodeParams,
    fit_four_node,
    identification,
)
from dhw_benchmark.calib_heat import A_LOW, COP_MAX, COP_MIN, estimate_aux_kw
from dhw_benchmark.calib_runs import (
    LEGIONELLA_STOP_C,
    STOP_RULES,
    StopKind,
    appliance_limits,
    check_run,
    draw_free,
    find_runs,
    RunCheck,
    off_gaps,
    rule_stats,
    select_rule,
    start_margins,
    thermostat_folds,
)
from dhw_benchmark.calib_windows import DrawRule, cooling_windows, heating_windows, legionella_excluded
from dhw_benchmark.inputs import TankRow, read_rows

COLD_MIN_C = 5.0
COLD_MAX_C = 20.0
COLD_FALLBACK_C = 12.0

_NOT_ASSESSED_REASON = "reduced scope (amendment AM)"


@dataclass(frozen=True)
class CalibrationConfig:
    """Choices of a calibration run.

    Attributes:
        room_temp_c: Room temperature around the tanks (assumption).
        draw_rule: Draw suspicion thresholds for window selection.
        acceptance: Predictive acceptance thresholds.
        lower_node: Node the lower probe observes (central: A_low).
    """

    room_temp_c: float = 18.0
    draw_rule: DrawRule = field(default_factory=DrawRule)
    acceptance: Acceptance = field(default_factory=Acceptance)
    lower_node: int = A_LOW

    def __post_init__(self) -> None:
        """Reject numbers that would make the run meaningless or the output non-finite."""
        numbers = {
            "room_temp_c": self.room_temp_c,
            "acceptance.heating_mae_k": self.acceptance.heating_mae_k,
            "acceptance.cooling_mae_k": self.acceptance.cooling_mae_k,
            "draw_rule.lower_fall_k": self.draw_rule.lower_fall_k,
            "draw_rule.top_fall_k": self.draw_rule.top_fall_k,
        }
        for name, value in numbers.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        positive = [v for k, v in numbers.items() if k != "room_temp_c"]
        counts = [
            self.acceptance.min_heating,
            self.acceptance.min_cooling,
            self.acceptance.min_folds,
            self.draw_rule.window_minutes,
        ]
        if any(v <= 0.0 for v in positive):
            raise ValueError("thresholds must be positive")
        if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in counts):
            raise ValueError("minimum counts and the window length must be positive integers")


def cold_water(rows: Sequence[TankRow]) -> tuple[float, bool, float | None]:
    """The cold-water temperature (spec §15.2, plan item 8).

    Args:
        rows: Minute rows in time order.

    Returns:
        ``(value, flagged, raw_minimum)``: the raw minimum of the lower probe
        outside legionella windows when it lies in 5-20 C, otherwise 12 C
        and flagged.
    """
    flags = legionella_excluded(rows)
    values = [r.t_lower_c for r, excluded in zip(rows, flags, strict=True) if not excluded and r.t_lower_c is not None]
    raw = min(values) if values else None
    if raw is None or not COLD_MIN_C <= raw <= COLD_MAX_C:
        return COLD_FALLBACK_C, True, raw
    return raw, False, raw


def _week_key(row: TankRow) -> str:
    year, week, _ = row.ts.isocalendar()
    return f"{year}-W{week:02d}"


def brine_medians(rows: Sequence[TankRow]) -> tuple[dict[str, float], float | None]:
    """Source-temperature medians of compressor dhw minutes (plan amendment I).

    Args:
        rows: Minute rows.

    Returns:
        ``(per ISO week, overall)``; the overall median is None without
        readings.
    """
    weekly: dict[str, list[float]] = {}
    for row in rows:
        if row.mode == "dhw" and row.compressor_on and row.brine_in_c is not None:
            weekly.setdefault(_week_key(row), []).append(row.brine_in_c)
    every = [v for values in weekly.values() for v in values]
    return {k: float(median(v)) for k, v in sorted(weekly.items())}, (float(median(every)) if every else None)


def _num(value: float | None) -> float | None:
    """A JSON-safe number: None for missing or non-finite values."""
    return None if value is None or not math.isfinite(value) else float(value)


def _mae(values: Sequence[float] | None) -> float | None:
    return _num(float(np.mean(np.abs(values)))) if values else None


def _cv_json(cv: CvResult) -> dict[str, Any]:
    return {
        "usable_folds": cv.usable_folds,
        "skipped_folds": len(cv.folds) - cv.usable_folds,
        "accepted": cv.accepted,
        "heating_top_mae_k": _num(cv.heating_top_mae_k),
        "cooling_top_mae_k": _num(cv.cooling_top_mae_k),
        "heating_lower_mae_k": _num(cv.heating_lower_mae_k),
        "cooling_lower_mae_k": _num(cv.cooling_lower_mae_k),
        "top_mae_k": _num(cv.top_mae_k),
        "heating_minutes": cv.heating_minutes,
        "cooling_minutes": cv.cooling_minutes,
        "folds": [
            {
                "week": f"{f.week[0]}-W{f.week[1]:02d}",
                "skipped_reason": f.skipped_reason,
                "aux_kw": _num(f.aux_kw),
                "aux_flagged": f.aux_flagged,
                "invalid_minutes": f.invalid_minutes,
                "heating_minutes": len(f.heating_top or []),
                "cooling_minutes": len(f.cooling_top or []),
                "heating_top_mae_k": _mae(f.heating_top),
                "cooling_top_mae_k": _mae(f.cooling_top),
                "heating_lower_mae_k": _mae(f.heating_lower),
                "cooling_lower_mae_k": _mae(f.cooling_lower),
                "cap": asdict(f.caps) if f.caps else None,
            }
            for f in cv.folds
        ],
    }


def _fit_json(fit: FitResult) -> dict[str, Any]:
    shifts = np.abs(np.asarray(fit.t_in_shift_k)) if fit.t_in_shift_k else None
    return {
        "parameters": {k: _num(v) for k, v in asdict(fit.params).items()},
        "training_rmse_k": _num(fit.rmse_k),
        "domain": {
            "source_c": [_num(v) for v in fit.source_range_c] if fit.source_range_c else None,
            "supply_c": [_num(v) for v in fit.t_in_range_c] if fit.t_in_range_c else None,
        },
        "cap": {**asdict(fit.caps), "strained": fit.caps.strained},
        "t_in_shift_k": None if shifts is None else {
            "median": _num(float(np.median(shifts))),
            "p95": _num(float(np.percentile(shifts, 95))),
            "max": _num(float(shifts.max())),
        },
    }


def _clip_cold(value: float) -> float:
    return min(COLD_MAX_C, max(COLD_MIN_C, value))


def _shortfalls(checks: Sequence[RunCheck | None]) -> dict[str, Any]:
    """Shortfalls of the runs where a rule did not fire (censored checks, amendments A and T)."""
    values = sorted(c.shortfall_k for c in checks if c is not None and c.shortfall_k is not None)
    return {
        "count": len(values),
        "median": _num(float(median(values))) if values else None,
        "max": _num(values[-1]) if values else None,
        "values": [_num(v) for v in values],
    }


def _not_assessed() -> dict[str, dict[str, str]]:
    items = {
        "parameter_profiles": _NOT_ASSESSED_REASON,
        "parameter_uncertainty": _NOT_ASSESSED_REASON,
        "weak_parameter_scenarios": _NOT_ASSESSED_REASON,
        "lower_probe_placement": _NOT_ASSESSED_REASON,
        "room_15_21_c": _NOT_ASSESSED_REASON,
        "draw_thresholds": _NOT_ASSESSED_REASON,
        "node_splits": _NOT_ASSESSED_REASON,
        "start_timing": "needs draw reconstruction between runs (amendment AN)",
    }
    return {k: {"value": "not_assessed", "reason": v} for k, v in items.items()}


def calibrate(
    rows: Sequence[TankRow],
    config: CalibrationConfig,
    starts: Sequence[FourNodeParams] = STARTS,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Run the calibration of one installation.

    Args:
        rows: The 1-minute tank data in time order.
        config: Choices of the run.
        starts: Fixed starting points for every fit.
        executor: Runs the folds in parallel when given.

    Returns:
        The content of ``calibration.json`` (JSON-safe).
    """
    heat_report = heating_windows(rows, config.draw_rule)
    cool_report = cooling_windows(rows, config.draw_rule)
    heating, cooling = heat_report.accepted, cool_report.accepted
    aux_kw, aux_flagged = estimate_aux_kw(rows)
    cold_c, cold_flagged, cold_raw = cold_water(rows)
    base = FitSettings(config.room_temp_c, cold_c, aux_kw, "A", config.lower_node)

    full_heating, invalid_heating = split_invalid(heating, aux_kw, heating=True)
    full_cooling, invalid_cooling = split_invalid(cooling, aux_kw)
    full_windows = [p for pieces in [*full_heating, *full_cooling] for p in pieces]

    cv: dict[str, CvResult] = {}
    fits: dict[str, FitResult] = {}
    for order in ("A", "B"):
        settings = replace(base, coil_order=order)
        cv[order] = cross_validate(
            rows, heating, cooling, settings, starts=starts, acceptance=config.acceptance, executor=executor
        )
        fits[order] = fit_four_node(full_windows, settings, starts=starts)
    central = choose_coil_order(cv)  # type: ignore[arg-type]
    central_settings = replace(base, coil_order=central)
    central_fit = fits[central]
    diagnosis = identification(central_fit)
    calibration_label = label(cv[central], diagnosis)

    runs = find_runs(rows)
    checked = draw_free(runs, heating)
    natural = [r for r in checked if r.mode == "dhw" and r.natural_stop]
    switched = [r for r in checked if r.mode == "dhw" and not r.natural_stop]  # switched or commanded
    stop_checks = {rule: [check_run(r, rule, central_fit.params, central_settings) for r in natural] for rule in STOP_RULES}
    switched_checks = {
        rule: [check_run(r, rule, central_fit.params, central_settings) for r in switched] for rule in STOP_RULES
    }
    stop_stats = {rule: rule_stats(checks, rule) for rule, checks in stop_checks.items()}
    complete_dhw = [r for r in runs if r.complete and r.mode == "dhw"]
    selected, inconclusive = select_rule(stop_stats)
    held = {f.week: f.test_minutes for f in fold_plan(heating, cooling)}
    fold_models = {
        f.week: (f.params, replace(central_settings, aux_kw=f.aux_kw))
        for f in cv[central].folds
        if f.params is not None
    }
    folds = thermostat_folds(runs, heating, fold_models, held_minutes=held)
    legionella_checks = [
        check_run(r, "a_low", central_fit.params, central_settings, threshold_c=LEGIONELLA_STOP_C)
        for r in runs
        if r.complete and r.mode == "legionella"
    ]
    legionella = rule_stats(legionella_checks, "a_low")
    gaps = off_gaps(rows, runs)
    margins = start_margins(runs, central_settings)
    limits = appliance_limits(runs, aux_kw)
    weekly_brine, overall_brine = brine_medians(rows)

    return {
        "label": calibration_label,
        "central_coil_order": central,
        "models": {
            order: {**_fit_json(fits[order]), "cross_validation": _cv_json(cv[order])} for order in ("A", "B")
        },
        "identification": {
            "kind": "local Jacobian diagnostic (amendment AM); parameter uncertainty not assessed",
            "full_rank": diagnosis.full_rank,
            "condition_number": _num(diagnosis.condition_number),
            "weak_parameters": diagnosis.weak,
            "identified": diagnosis.identified,
            "thresholds": {"condition_number": 1e6, "weak_singular_ratio": 1e-6, "weak_loading": 0.3},
        },
        "acceptance": asdict(config.acceptance),
        "domain": {
            "note": "per model under models.<order>.domain; this is the central model's",
            **_fit_json(central_fit)["domain"],
            "cop_range": [COP_MIN, COP_MAX],
            "cop_strained_share": STRAINED_SHARE,
        },
        "sensitivities": {
            "coil_order": "B" if central == "A" else "A",
            "source_offsets_k": [-2.0, 2.0],
            "source_note": "applied downstream with frozen coefficients (amendment E)",
            "cold_water_c": [_num(_clip_cold(cold_c - 3.0)), _num(_clip_cold(cold_c + 3.0))],
            "cold_water_note": "central +-3 K clipped to 5-20 C (plan item 8)",
        },
        "source_temperature_fallback": {
            "chain": [
                "last pump-on reading up to 24 hours old",
                "median dhw-run source temperature of that ISO week",
                "median dhw-run source temperature over all data (flagged)",
            ],
            "weekly_median_c": weekly_brine,
            "overall_median_c": _num(overall_brine),
        },
        "aux": {"kw": _num(aux_kw), "flagged": aux_flagged},
        "cold_water": {"c": _num(cold_c), "flagged": cold_flagged, "raw_minimum_c": _num(cold_raw)},
        "settings": {
            "room_temp_c": config.room_temp_c,
            "lower_probe_node": "A_low" if config.lower_node == A_LOW else "A_high",
            "node_split": [base.split_a, base.split_b],
            "draw_rule": asdict(config.draw_rule),
        },
        "windows": {
            "heating": len(heating),
            "cooling": len(cooling),
            "heating_after_split": len(full_heating),
            "cooling_after_split": len(full_cooling),
            "invalid_minutes": invalid_heating + invalid_cooling,
            "rejected_heating": heat_report.rejected,
            "rejected_cooling": cool_report.rejected,
        },
        "thermostat": {
            "stop_rule": {
                "selected": selected,
                "inconclusive": inconclusive,
                "rules": {
                    rule: {**asdict(s), "coverage": s.coverage, "shortfalls_k": _shortfalls(stop_checks[rule])}
                    for rule, s in stop_stats.items()
                },
                "folds": [{**asdict(f), "week": f"{f.week[0]}-W{f.week[1]:02d}"} for f in folds],
                "evaluation_set": "draw-free complete dhw runs with a natural stop (amendments AQ, AR)",
                "non_natural_stop_diagnostic": {
                    rule: {**asdict(s), "coverage": s.coverage, "shortfalls_k": _shortfalls(switched_checks[rule])}
                    for rule, s in ((rule, rule_stats(c, rule)) for rule, c in switched_checks.items())
                },
            },
            "stop_kinds": {
                "complete_dhw_runs": len(complete_dhw),
                **{kind: sum(r.stop_kind == kind for r in complete_dhw) for kind in get_args(StopKind)},
                "natural_start": sum(r.natural_start for r in complete_dhw),
            },
            "legionella_rule": {
                **asdict(legionella),
                "coverage": legionella.coverage,
                "shortfalls_k": _shortfalls(legionella_checks),
                "note": "draws not excluded",
            },
            "start_diagnostic": {
                "kind": "estimated tank A mean minus band_low_c at recorded starts of complete runs",
                "runs": len(margins),
                "median_k": _num(float(median(margins))) if margins else None,
                "share_at_or_below": _num(sum(m <= 0.0 for m in margins) / len(margins)) if margins else None,
            },
        },
        "targets": "commanded by the user's automation; the appliance has no default band (amendment AP)",
        "limits": {
            **asdict(limits),
            "minimum_run": "from the appliance settings; to be confirmed (spec §21); the shortest run is an observation",
            "label": "estimates under commanded targets",
            "observed_off_gaps": {
                "count": len(gaps),
                "min_minutes": min((g.minutes for g in gaps), default=None),
                "below_min_off": sum(1 for g in gaps if g.minutes < limits.min_off_minutes),
                "outside_dhw_minutes": sum(g.outside_dhw_minutes for g in gaps),
                "note": "cross-check only; the configured minimum off time is used",
            },
        },
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "assumptions": [
            "four nodes in two tanks (90/90/100/100 L), exchange only within a tank, instant mixing on inversion",
            "coils in series; the order is a variant chosen by cross-validation",
            "COP linear in source and supply temperature, capped to 1-8, held constant within a minute",
            "element heat enters tank A's upper node",
            f"room temperature {config.room_temp_c} C",
            "draw-free windows by the draw suspicion rule",
            "thermostat check on draw-free complete runs with zero draws",
        ],
        "not_assessed": _not_assessed(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Calibrate the tank model and write calibration.json.")
    parser.add_argument("--tank", type=Path, required=True, help="tank.csv of the input set")
    parser.add_argument("--out", type=Path, required=True, help="calibration.json to write")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes for the folds")
    args = parser.parse_args(argv)
    rows = read_rows(args.tank, TankRow)
    config = CalibrationConfig()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        result = calibrate(rows, config, executor=executor)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['label']} (central coil order {result['central_coil_order']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
