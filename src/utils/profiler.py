"""Pipeline performance profiler.

Uses ``time.perf_counter()`` for sub-millisecond accuracy.

Typical usage::

    profiler = PipelineProfiler()

    # inside the main loop:
    profiler.start("vehicle_detect")
    detections = model.detect(frame)
    profiler.stop("vehicle_detect")

    profiler.tick()  # once per processed frame

    # on video loop (optional):
    profiler.new_epoch()

    # after the loop:
    profiler.print_report()
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Internal per-stage accumulator
# ---------------------------------------------------------------------------

@dataclass
class _StageStats:
    total_sec: float = 0.0
    calls: int = 0
    _t0: float = field(default=0.0, repr=False)


@dataclass
class _EpochSnapshot:
    """Immutable summary of one completed video-file pass."""
    stage_totals: dict[str, float]   # stage -> total_sec for this epoch
    frames: int
    wall_sec: float


# ---------------------------------------------------------------------------
# Public profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Lightweight named-stage stop-watch for a video processing pipeline.

    Supports multi-epoch mode: call ``new_epoch()`` each time the video loops.
    ``print_report()`` will then show per-stage averages across all epochs so
    that results are not skewed by cumulative sums.

    Thread-safety: **not** thread-safe — intended for single-threaded
    main loops only.
    """

    # Column widths for the report table
    _W_STAGE: int = 22
    _W_TOTAL: int = 10
    _W_AVG:   int = 14
    _W_PCT:   int = 8

    def __init__(self) -> None:
        self._stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self._frames: int = 0
        self._wall_start: float = time.perf_counter()
        self._epochs: list[_EpochSnapshot] = []

    # ------------------------------------------------------------------
    # Measurement API
    # ------------------------------------------------------------------

    def start(self, stage: str) -> None:
        """Begin timing *stage*.  May be called multiple times per frame
        (time accumulates across iterations of inner loops)."""
        self._stats[stage]._t0 = time.perf_counter()

    def stop(self, stage: str) -> None:
        """Stop timing *stage* and add elapsed seconds to its accumulator."""
        elapsed = time.perf_counter() - self._stats[stage]._t0
        stat = self._stats[stage]
        stat.total_sec += elapsed
        stat.calls += 1

    def tick(self) -> None:
        """Increment the frame counter.  Call **once** per processed frame."""
        self._frames += 1

    def new_epoch(self) -> None:
        """Save the current pass as a completed epoch and reset all counters.

        Call this whenever the video source loops so that the final report
        shows per-epoch averages rather than ever-growing cumulative totals.
        No data is lost — every epoch's snapshot is retained for aggregation.
        """
        if self._frames == 0:
            # Nothing recorded yet — skip to avoid a degenerate epoch.
            return
        snapshot = _EpochSnapshot(
            stage_totals={k: v.total_sec for k, v in self._stats.items()},
            frames=self._frames,
            wall_sec=time.perf_counter() - self._wall_start,
        )
        self._epochs.append(snapshot)
        # Reset current-epoch accumulators
        self._stats = defaultdict(_StageStats)
        self._frames = 0
        self._wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _collect_epochs(self) -> list[_EpochSnapshot]:
        """Return all completed epochs plus the currently open one (if any)."""
        current_frames = self._frames
        if current_frames == 0:
            return list(self._epochs)
        current = _EpochSnapshot(
            stage_totals={k: v.total_sec for k, v in self._stats.items()},
            frames=current_frames,
            wall_sec=time.perf_counter() - self._wall_start,
        )
        return list(self._epochs) + [current]

    def print_report(self) -> None:
        """Print a formatted table to *stdout* with per-stage statistics.

        If the video looped at least once (``new_epoch()`` was called), the
        report shows the **weighted average** across all epochs so that each
        pass contributes equally regardless of frame count differences.
        """
        epochs = self._collect_epochs()
        if not epochs:
            print("[PipelineProfiler] No data collected.")
            return

        n_epochs = len(epochs)
        total_frames = sum(e.frames for e in epochs)
        total_wall   = sum(e.wall_sec for e in epochs)

        # Union of all stage names seen across epochs
        all_stages: set[str] = set()
        for e in epochs:
            all_stages.update(e.stage_totals.keys())

        # Per-stage: weighted-average ms/frame across epochs
        # weight = frames in epoch / total_frames
        avg_ms_per_stage: dict[str, float] = {}
        total_sec_per_stage: dict[str, float] = {}
        for stage in all_stages:
            stage_total = sum(e.stage_totals.get(stage, 0.0) for e in epochs)
            total_sec_per_stage[stage] = stage_total
            # Weighted avg/frame in ms
            avg_ms_per_stage[stage] = (stage_total / total_frames) * 1_000

        sum_measured = sum(total_sec_per_stage.values())

        W = self._W_STAGE + self._W_TOTAL + self._W_AVG + self._W_PCT + 3
        sep  = "─" * W
        dsep = "═" * W

        def _row(stage: str, total: float, avg_ms: float, pct: float | None) -> str:
            pct_str = f"{pct:>{self._W_PCT}.1f}%" if pct is not None else " " * (self._W_PCT + 1)
            return (
                f"{stage:<{self._W_STAGE}} "
                f"{total:>{self._W_TOTAL}.3f} "
                f"{avg_ms:>{self._W_AVG}.2f} "
                f"{pct_str}"
            )

        epoch_note = (
            f"  (averaged over {n_epochs} video passes)"
            if n_epochs > 1 else ""
        )

        lines: list[str] = [
            "",
            dsep,
            f"  Pipeline Profiling Report{epoch_note}",
            dsep,
            (
                f"{'Stage':<{self._W_STAGE}} "
                f"{'Total (s)':>{self._W_TOTAL}} "
                f"{'Avg/frame (ms)':>{self._W_AVG}} "
                f"{'% load':>{self._W_PCT + 1}}"
            ),
            sep,
        ]

        sorted_stages = sorted(
            all_stages,
            key=lambda s: total_sec_per_stage[s],
            reverse=True,
        )

        for stage in sorted_stages:
            total = total_sec_per_stage[stage]
            avg_ms = avg_ms_per_stage[stage]
            pct = (total / sum_measured * 100) if sum_measured > 0 else 0.0
            lines.append(_row(stage, total, avg_ms, pct))

        lines += [
            sep,
            _row(
                "TOTAL (measured)",
                sum_measured,
                (sum_measured / total_frames) * 1_000,
                100.0,
            ),
            sep,
            f"{'Wall-clock total':<{self._W_STAGE}} "
            f"{total_wall:>{self._W_TOTAL}.3f}",
            f"{'Frames processed':<{self._W_STAGE}} "
            f"{total_frames:>{self._W_TOTAL}d}",
            f"{'Video passes':<{self._W_STAGE}} "
            f"{n_epochs:>{self._W_TOTAL}d}",
            f"{'Average FPS':<{self._W_STAGE}} "
            f"{total_frames / max(total_wall, 1e-9):>{self._W_TOTAL}.2f}",
            dsep,
            "",
        ]

        print("\n".join(lines))
