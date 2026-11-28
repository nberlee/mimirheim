"""Leave-one-ISO-week-out validation of the tank fit, and the calibration label (ticket 03).

For each ISO week that holds a window (spec §17, plan amendments B, C, AD
and AM):

- the week's own windows are the test set (a window belongs to the week of
  its first minute); the held-out minutes are the week plus every minute
  of a test window, including one that runs into the next week;
- the training set is every window without a held-out minute, and
  auxiliary power is estimated from minutes outside the held-out set;
- windows come from ``calib_windows``, which selects them by mode,
  compressor state, probes and source readings, never by auxiliary power,
  so no aux-dependent filtering happens before the fold;
- a minute whose heat input is invalid under the fold's auxiliary power
  (plan amendment B) splits its window, and those minutes are counted; a
  heating window counts, and is scored, only if a minute with positive
  compressor power survives;
- the model is fitted from the same fixed starting points as the full-data
  fit, so nothing learnt from the held-out week initialises the fold;
- the held-out windows are simulated forward from their start state, with
  no resets and no inferred draws.

A fold needs at least 8 heating and 3 cooling training windows; otherwise
it is skipped and counted. Predictive acceptance pools the held-out minutes
of all usable folds: top-probe MAE at most 2 K in heating windows and at
most 1 K in cooling windows, with at least 3 usable folds.

The central coil order is the one with the lower pooled top-probe MAE over
all held-out minutes (heating and cooling together); a tie goes to order A.

This module validates and labels; it does not select windows, fit the
final parameters or write ``calibration.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

import numpy as np

from dhw_benchmark.calib_fit import (
    STARTS,
    CapStats,
    Diagnosis,
    FitSettings,
    FourNodeParams,
    cap_stats,
    fit_four_node,
    simulate_window,
)
from dhw_benchmark.calib_heat import CoilOrder, compressor_electric_kw, estimate_aux_kw, heat_input_valid
from dhw_benchmark.inputs import TankRow

Week = tuple[int, int]
"""ISO year and week number."""

Label = Literal["conditional on the listed assumptions", "predictive only", "not identified"]

Window = Sequence[TankRow]


@dataclass(frozen=True)
class Acceptance:
    """Predictive acceptance thresholds (plan amendments C and AD).

    Attributes:
        heating_mae_k: Maximum pooled top-probe MAE in heating windows.
        cooling_mae_k: Maximum pooled top-probe MAE in cooling windows.
        min_heating: Training heating windows a fold needs.
        min_cooling: Training cooling windows a fold needs.
        min_folds: Usable folds needed.
    """

    heating_mae_k: float = 2.0
    cooling_mae_k: float = 1.0
    min_heating: int = 8
    min_cooling: int = 3
    min_folds: int = 3


@dataclass(frozen=True)
class Fold:
    """The windows of one fold.

    Attributes:
        week: The held-out ISO week.
        train_heating: Heating windows that do not touch the week.
        train_cooling: Cooling windows that do not touch the week.
        test_heating: Heating windows starting in the week.
        test_cooling: Cooling windows starting in the week.
    """

    week: Week
    train_heating: list[Window]
    train_cooling: list[Window]
    test_heating: list[Window]
    test_cooling: list[Window]

    @property
    def test_minutes(self) -> frozenset[datetime]:
        """Every minute of the test windows."""
        return frozenset(r.ts for w in [*self.test_heating, *self.test_cooling] for r in w)


def _week(ts: datetime) -> Week:
    """The ISO week of a timestamp."""
    year, week, _ = ts.isocalendar()
    return (year, week)


def fold_plan(heating: Sequence[Window], cooling: Sequence[Window]) -> list[Fold]:
    """One fold per ISO week that holds the first minute of a window.

    Args:
        heating: Heating windows.
        cooling: Cooling windows.

    Returns:
        The folds in week order.
    """
    weeks = sorted({_week(w[0].ts) for w in [*heating, *cooling]})
    folds = []
    for week in weeks:
        test_heating = [w for w in heating if _week(w[0].ts) == week]
        test_cooling = [w for w in cooling if _week(w[0].ts) == week]
        held = {r.ts for w in [*test_heating, *test_cooling] for r in w}

        def clean(window: Window, week: Week = week, held: set[datetime] = held) -> bool:
            return not any(_week(r.ts) == week or r.ts in held for r in window)

        folds.append(
            Fold(
                week=week,
                train_heating=[w for w in heating if clean(w)],
                train_cooling=[w for w in cooling if clean(w)],
                test_heating=test_heating,
                test_cooling=test_cooling,
            )
        )
    return folds


def _heats(row: TankRow, aux_kw: float) -> bool:
    """Whether the compressor heats the tanks in this minute under the auxiliary power."""
    electric = compressor_electric_kw(row, aux_kw)
    return isinstance(electric, float) and electric > 0.0


def fold_aux(fold: Fold, rows: Sequence[TankRow]) -> tuple[float, bool]:
    """Auxiliary power from the minutes that are not held out (plan amendment B).

    Args:
        fold: The fold.
        rows: All minute rows.

    Returns:
        ``(aux_kw, flagged)`` per ``estimate_aux_kw``.
    """
    held = fold.test_minutes
    return estimate_aux_kw([r for r in rows if _week(r.ts) != fold.week and r.ts not in held])


def split_invalid(
    windows: Sequence[Window], aux_kw: float, heating: bool = False
) -> tuple[list[list[Window]], int]:
    """Split windows at minutes whose heat input is invalid under an auxiliary power.

    The step from an invalid minute to the next is unknown, so the window
    ends at the invalid minute and resumes at the following one. Pieces with
    fewer than two minutes have no step and are dropped.

    Args:
        windows: Windows from ``calib_windows`` (selected without aux).
        aux_kw: The fold's auxiliary power.
        heating: Heating windows: a window is kept only if one of its
            pieces still has a step with positive compressor power.

    Returns:
        The pieces of every kept window (so window counts stay counts of
        selected windows), and the number of invalid minutes.
    """
    kept: list[list[Window]] = []
    invalid = 0
    for window in windows:
        pieces: list[Window] = []
        current: list[TankRow] = []
        for row in window:
            current.append(row)
            if not heat_input_valid(row, aux_kw):
                invalid += 1
                pieces.append(current)
                current = []
        pieces.append(current)
        pieces = [p for p in pieces if len(p) >= 2]
        if heating and not any(_heats(r, aux_kw) for p in pieces for r in p[:-1]):
            continue  # no heating step left: not evidence of heating
        if pieces:
            kept.append(pieces)
    return kept, invalid


@dataclass
class FoldResult:
    """Outcome of one fold.

    Attributes:
        week: The held-out week.
        skipped_reason: Why the fold was skipped, or None when it was used.
        aux_kw: Auxiliary power from the training minutes.
        aux_flagged: Fewer than 60 quiet idle training minutes.
        invalid_minutes: Minutes invalid under the fold's auxiliary power,
            which split their windows.
        params: Parameters fitted on the training windows.
        heating_top: Held-out top-probe residuals in heating windows.
        cooling_top: Held-out top-probe residuals in cooling windows.
        heating_lower: Held-out lower-probe residuals in heating windows.
        cooling_lower: Held-out lower-probe residuals in cooling windows.
        caps: How often the COP cap binds in the held-out windows.
    """

    week: Week
    skipped_reason: str | None = None
    aux_kw: float = 0.0
    aux_flagged: bool = False
    invalid_minutes: int = 0
    params: FourNodeParams | None = None
    heating_top: list[float] | None = None
    cooling_top: list[float] | None = None
    heating_lower: list[float] | None = None
    cooling_lower: list[float] | None = None
    caps: CapStats | None = None


def _mae(values: Iterable[float]) -> float:
    """Mean absolute value, NaN when empty."""
    arr = np.abs(np.asarray(list(values), dtype=float))
    return float(arr.mean()) if arr.size else float("nan")


@dataclass
class CvResult:
    """Cross-validation of one model variant.

    Attributes:
        folds: Every fold, used or skipped.
        acceptance: The thresholds applied.
    """

    folds: list[FoldResult]
    acceptance: Acceptance

    @property
    def used(self) -> list[FoldResult]:
        """The folds that were not skipped."""
        return [f for f in self.folds if f.skipped_reason is None]

    @property
    def usable_folds(self) -> int:
        """Number of folds used."""
        return len(self.used)

    @property
    def heating_top_mae_k(self) -> float:
        """Pooled held-out top-probe MAE in heating windows."""
        return _mae(r for f in self.used for r in f.heating_top or [])

    @property
    def cooling_top_mae_k(self) -> float:
        """Pooled held-out top-probe MAE in cooling windows."""
        return _mae(r for f in self.used for r in f.cooling_top or [])

    @property
    def heating_lower_mae_k(self) -> float:
        """Pooled held-out lower-probe MAE in heating windows."""
        return _mae(r for f in self.used for r in f.heating_lower or [])

    @property
    def cooling_lower_mae_k(self) -> float:
        """Pooled held-out lower-probe MAE in cooling windows."""
        return _mae(r for f in self.used for r in f.cooling_lower or [])

    @property
    def heating_minutes(self) -> int:
        """Held-out heating-window minutes pooled."""
        return sum(len(f.heating_top or []) for f in self.used)

    @property
    def cooling_minutes(self) -> int:
        """Held-out cooling-window minutes pooled."""
        return sum(len(f.cooling_top or []) for f in self.used)

    @property
    def top_mae_k(self) -> float:
        """Pooled held-out top-probe MAE over all held-out minutes."""
        return _mae(r for f in self.used for r in [*(f.heating_top or []), *(f.cooling_top or [])])

    @property
    def accepted(self) -> bool:
        """Enough usable folds and both pooled MAEs within their thresholds.

        A window type without held-out minutes cannot pass.
        """
        heating, cooling = self.heating_top_mae_k, self.cooling_top_mae_k
        return (
            self.usable_folds >= self.acceptance.min_folds
            and heating <= self.acceptance.heating_mae_k
            and cooling <= self.acceptance.cooling_mae_k
        )


def _split(residuals: Sequence[float]) -> tuple[list[float], list[float]]:
    """Lower and top residuals from the interleaved list."""
    return list(residuals[0::2]), list(residuals[1::2])


def run_fold(
    fold: Fold,
    rows: Sequence[TankRow],
    settings: FitSettings,
    starts: Sequence[FourNodeParams],
    acceptance: Acceptance,
    fixed: Mapping[str, float] | None = None,
) -> FoldResult:
    """Fit on a fold's training windows and simulate its held-out windows.

    Args:
        fold: The fold.
        rows: All minute rows (auxiliary power uses those not held out).
        settings: Fixed values and variants; ``aux_kw`` is replaced by the
            fold's own estimate.
        starts: Fixed starting points.
        acceptance: Window-count requirements.
        fixed: Parameters held fixed.

    Returns:
        The fold result.
    """
    result = FoldResult(week=fold.week)
    result.aux_kw, result.aux_flagged = fold_aux(fold, rows)
    train_heating, n1 = split_invalid(fold.train_heating, result.aux_kw, heating=True)
    train_cooling, n2 = split_invalid(fold.train_cooling, result.aux_kw)
    test_heating, n3 = split_invalid(fold.test_heating, result.aux_kw, heating=True)
    test_cooling, n4 = split_invalid(fold.test_cooling, result.aux_kw)
    result.invalid_minutes = n1 + n2 + n3 + n4
    if len(train_heating) < acceptance.min_heating or len(train_cooling) < acceptance.min_cooling:
        result.skipped_reason = f"{len(train_heating)} heating and {len(train_cooling)} cooling training windows"
        return result
    if not test_heating and not test_cooling:
        result.skipped_reason = "no held-out windows"
        return result
    fold_settings = replace(settings, aux_kw=result.aux_kw)
    training = [p for pieces in [*train_heating, *train_cooling] for p in pieces]
    fit = fit_four_node(training, fold_settings, starts=starts, fixed=fixed)
    result.params = fit.params
    heating_sims = [simulate_window(p, fit.params, fold_settings) for pieces in test_heating for p in pieces]
    cooling_sims = [simulate_window(p, fit.params, fold_settings) for pieces in test_cooling for p in pieces]
    result.heating_lower, result.heating_top = _split([r for s in heating_sims for r in s.residuals])
    result.cooling_lower, result.cooling_top = _split([r for s in cooling_sims for r in s.residuals])
    result.caps = cap_stats(heating_sims + cooling_sims)
    return result


def cross_validate(
    rows: Sequence[TankRow],
    heating: Sequence[Window],
    cooling: Sequence[Window],
    settings: FitSettings,
    starts: Sequence[FourNodeParams] = STARTS,
    acceptance: Acceptance | None = None,
    fixed: Mapping[str, float] | None = None,
    executor: Executor | None = None,
) -> CvResult:
    """Leave-one-ISO-week-out validation of one model variant.

    Args:
        rows: All minute rows.
        heating: Heating windows.
        cooling: Cooling windows.
        settings: Fixed values and variants.
        starts: Fixed starting points, the same for every fold.
        acceptance: Thresholds; the defaults of ``Acceptance`` when None.
        fixed: Parameters held fixed.
        executor: Runs the folds in parallel when given (for example a
            process pool); sequential otherwise.

    Returns:
        The cross-validation result.
    """
    acceptance = acceptance or Acceptance()
    plan = fold_plan(heating, cooling)
    args = [(fold, rows, settings, list(starts), acceptance, dict(fixed or {})) for fold in plan]
    mapper: Callable[..., Iterable[FoldResult]] = executor.map if executor is not None else map
    folds = list(mapper(run_fold, *zip(*args, strict=True))) if args else []
    return CvResult(folds=folds, acceptance=acceptance)


def label(cv: CvResult, diagnosis: Diagnosis) -> Label:
    """The calibration label (plan amendments AD and AM).

    Args:
        cv: Cross-validation of the central variant.
        diagnosis: Local Jacobian diagnostic of its full-data fit.

    Returns:
        "conditional on the listed assumptions" when predictive acceptance
        passes and no parameter is weak; "predictive only" when it passes
        with weak parameters; "not identified" when it fails.
    """
    if not cv.accepted:
        return "not identified"
    return "conditional on the listed assumptions" if diagnosis.identified else "predictive only"


def choose_coil_order(results: Mapping[CoilOrder, CvResult]) -> CoilOrder:
    """The coil order with the lower pooled held-out top-probe MAE; a tie goes to A.

    Args:
        results: Cross-validation per coil order.

    Returns:
        The central coil order.
    """
    def key(order: CoilOrder) -> tuple[float, int]:
        mae = results[order].top_mae_k
        return (mae if mae == mae else float("inf"), 0 if order == "A" else 1)

    return min(results, key=key)
