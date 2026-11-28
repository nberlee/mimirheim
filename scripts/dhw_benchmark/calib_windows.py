"""Select the draw-free windows that calibrate the tank model (benchmark ticket 03).

Two kinds of windows, both assumed free of hot water draws (spec §4, §13.1;
the assumption is reported with the counts of accepted and rejected
windows):

- cooling windows: two hours or more of idle minutes (no hot water,
  legionella or space heating), which show standby loss and mixing;
- heating windows: a hot water run with 30 minutes before and after it,
  which show the heat input. Space heating may run in that padding (the
  heat pump switches straight between the two, and space heating sends no
  heat into the tank); the run itself is hot water only.

A draw is suspected when a probe falls too fast over a rolling 15-minute
difference of the minute readings: the lower probe by more than 0.5 K (cold
water enters at the inlet first) or the top probe by more than 0.3 K. Within
a heating run only a top-probe fall counts, since heating raises the lower
probe. Legionella runs and the six hours after them are excluded.

This module only selects windows; it fits nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from dhw_benchmark.inputs import TankRow

_MINUTE = timedelta(minutes=1)
_LEGIONELLA_TAIL = timedelta(hours=6)
_COOLING_MIN_MINUTES = 120
_PADDING_MINUTES = 30
_TOLERANCE_K = 1e-9  # a fall exactly at a threshold is allowed despite rounding


@dataclass(frozen=True)
class DrawRule:
    """Thresholds of the draw suspicion rule.

    Attributes:
        lower_fall_k: Largest allowed fall of the lower probe over the window.
        top_fall_k: Largest allowed fall of the top probe over the window.
        window_minutes: Length of the rolling difference.
    """

    lower_fall_k: float = 0.5
    top_fall_k: float = 0.3
    window_minutes: int = 15

    def scaled(self, factor: float) -> DrawRule:
        """The same rule with both thresholds multiplied, for sensitivity runs.

        Args:
            factor: Multiplier for both thresholds.

        Returns:
            The scaled rule.
        """
        return DrawRule(self.lower_fall_k * factor, self.top_fall_k * factor, self.window_minutes)


@dataclass
class WindowReport:
    """Selected windows and the reasons others were rejected.

    Attributes:
        accepted: The accepted windows, each a run of consecutive minutes.
        rejected: Count of rejected windows per reason.
    """

    accepted: list[list[TankRow]] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)


def segments(rows: Sequence[TankRow]) -> list[list[TankRow]]:
    """Split minute rows into runs of consecutive minutes with both probes known.

    Args:
        rows: Minute rows in time order.

    Returns:
        The runs; a missing minute or a missing probe reading ends a run.
    """
    runs: list[list[TankRow]] = []
    current: list[TankRow] = []
    for row in rows:
        usable = row.t_top_c is not None and row.t_lower_c is not None
        contiguous = bool(current) and row.ts - current[-1].ts == _MINUTE
        if not usable:
            if current:
                runs.append(current)
            current = []
            continue
        if current and not contiguous:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def legionella_excluded(rows: Sequence[TankRow]) -> list[bool]:
    """Flag legionella minutes and the six hours after each.

    Args:
        rows: Minute rows in time order.

    Returns:
        One flag per row.
    """
    flags: list[bool] = []
    until = None
    for row in rows:
        if row.mode == "legionella":
            until = row.ts + _LEGIONELLA_TAIL
        flags.append(until is not None and row.ts <= until)
    return flags


def _falls(rows: Sequence[TankRow], rule: DrawRule, lower: bool, top: bool) -> list[bool]:
    """Minutes covered by a too-fast fall over the rolling window, within ``rows``."""
    n = rule.window_minutes
    suspect = [False] * len(rows)
    for i in range(n, len(rows)):
        a, b = rows[i - n], rows[i]
        fall_lower = (
            lower and a.t_lower_c is not None and b.t_lower_c is not None
            and a.t_lower_c - b.t_lower_c > rule.lower_fall_k + _TOLERANCE_K
        )
        fall_top = (
            top and a.t_top_c is not None and b.t_top_c is not None
            and a.t_top_c - b.t_top_c > rule.top_fall_k + _TOLERANCE_K
        )
        if fall_lower or fall_top:
            for j in range(i - n, i + 1):
                suspect[j] = True
    return suspect


def cooling_windows(rows: Sequence[TankRow], rule: DrawRule) -> WindowReport:
    """Idle stretches of two hours or more without a suspected draw.

    Args:
        rows: Minute rows in time order.
        rule: Draw suspicion thresholds.

    Returns:
        The accepted windows; rejections are counted as ``draw`` (a stretch
        cut by a suspected draw) and ``too_short`` (pieces under two hours
        after cutting).
    """
    report = WindowReport()
    rejected: Counter[str] = Counter()
    excluded = dict(zip((r.ts for r in rows), legionella_excluded(rows), strict=True))
    for segment in segments(rows):
        stretch: list[TankRow] = []
        for row in [*segment, None]:
            if row is not None and row.mode == "idle" and not excluded[row.ts]:
                stretch.append(row)
                continue
            if stretch:
                suspect = _falls(stretch, rule, lower=True, top=True)
                if any(suspect):
                    rejected["draw"] += 1
                piece: list[TankRow] = []
                for r, bad in zip([*stretch, None], [*suspect, True], strict=True):
                    if not bad and r is not None:
                        piece.append(r)
                        continue
                    if piece:
                        if len(piece) >= _COOLING_MIN_MINUTES:
                            report.accepted.append(piece)
                        else:
                            rejected["too_short"] += 1
                    piece = []
            stretch = []
    report.rejected = dict(rejected)
    return report


def _coil_active(row: TankRow) -> bool:
    """Whether the heat pump heats the tank in this minute."""
    return bool(row.compressor_on) and row.mode in ("dhw", "legionella")


def heating_windows(rows: Sequence[TankRow], rule: DrawRule) -> WindowReport:
    """Hot water runs with 30 quiet minutes before and after, without a suspected draw.

    Args:
        rows: Minute rows in time order.
        rule: Draw suspicion thresholds.

    Returns:
        The accepted windows; rejections are counted as ``incomplete`` (not
        30 minutes of data on both sides), ``other_mode`` (legionella in the
        window, or a run minute that is not hot water), ``missing_mode`` (a
        minute whose mode is unknown, so its tank heat is unknown),
        ``legionella_tail``, ``missing_brine`` (a coil-active minute anywhere
        in the window without a source temperature) and ``draw``.
    """
    report = WindowReport()
    rejected: Counter[str] = Counter()
    excluded = dict(zip((r.ts for r in rows), legionella_excluded(rows), strict=True))
    pad = _PADDING_MINUTES
    for segment in segments(rows):
        i = 0
        while i < len(segment):
            if not (segment[i].mode == "dhw" and segment[i].compressor_on):
                i += 1
                continue
            j = i
            while j < len(segment) and segment[j].mode == "dhw" and segment[j].compressor_on:
                j += 1
            run_start, run_end = i, j  # [run_start, run_end)
            i = j
            if run_start < pad or run_end + pad > len(segment):
                rejected["incomplete"] += 1
                continue
            window = segment[run_start - pad : run_end + pad]
            before, run, after = window[:pad], window[pad : pad + (run_end - run_start)], window[pad + (run_end - run_start) :]
            if any(r.mode is None for r in window):
                rejected["missing_mode"] += 1
                continue
            if any(r.mode == "legionella" for r in window) or any(r.mode != "dhw" for r in run):
                rejected["other_mode"] += 1
                continue
            if any(excluded[r.ts] for r in window):
                rejected["legionella_tail"] += 1
                continue
            if any(_coil_active(r) and r.brine_in_c is None for r in window):
                rejected["missing_brine"] += 1
                continue
            draw = (
                any(_falls(before, rule, lower=True, top=True))
                or any(_falls(after, rule, lower=True, top=True))
                or any(_falls(run, rule, lower=False, top=True))
            )
            if draw:
                rejected["draw"] += 1
                continue
            report.accepted.append(list(window))
    report.rejected = dict(rejected)
    return report
