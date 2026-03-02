
from __future__ import annotations

import logging
from pathlib import Path

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _StageStats:
    total_sec: float = 0.0
    calls: int = 0
    _t0: float = field(default=0.0, repr=False)


@dataclass
class _EpochSnapshot:
    stage_totals: dict[str, float]
    frames: int
    wall_sec: float


class PipelineProfiler:

    _W_STAGE: int = 22
    _W_TOTAL: int = 10
    _W_AVG: int = 14
    _W_PCT: int = 8

    def __init__(self) -> None:
        self._stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self._frames: int = 0
        self._wall_start: float = time.perf_counter()
        self._epochs: list[_EpochSnapshot] = []

    def start(self, stage: str) -> None:
        self._stats[stage]._t0 = time.perf_counter()

    def stop(self, stage: str) -> None:
        elapsed = time.perf_counter() - self._stats[stage]._t0
        stat = self._stats[stage]
        stat.total_sec += elapsed
        stat.calls += 1

    def tick(self) -> None:
        self._frames += 1

    def new_epoch(self) -> None:
        if self._frames == 0:
            return
        snapshot = _EpochSnapshot(
            stage_totals={k: v.total_sec for k, v in self._stats.items()},
            frames=self._frames,
            wall_sec=time.perf_counter() - self._wall_start,
        )
        self._epochs.append(snapshot)
        self._stats = defaultdict(_StageStats)
        self._frames = 0
        self._wall_start = time.perf_counter()

    def _collect_epochs(self) -> list[_EpochSnapshot]:
        current_frames = self._frames
        if current_frames == 0:
            return list(self._epochs)
        current = _EpochSnapshot(
            stage_totals={k: v.total_sec for k, v in self._stats.items()},
            frames=current_frames,
            wall_sec=time.perf_counter() - self._wall_start,
        )
        return list(self._epochs) + [current]

    def _build_report_text(self) -> str:
        epochs = self._collect_epochs()
        if not epochs:
            return "[PipelineProfiler] No data collected."

        n_epochs = len(epochs)
        total_frames = sum(e.frames for e in epochs)
        total_wall   = sum(e.wall_sec for e in epochs)

        all_stages: set[str] = set()
        for e in epochs:
            all_stages.update(e.stage_totals.keys())

        avg_ms_per_stage: dict[str, float] = {}
        total_sec_per_stage: dict[str, float] = {}
        for stage in all_stages:
            stage_total = sum(e.stage_totals.get(stage, 0.0) for e in epochs)
            total_sec_per_stage[stage] = stage_total
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

        return "\n".join(lines)

    def print_report(self) -> None:
        logging.getLogger(__name__).info(self._build_report_text())

    def save_report(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._build_report_text(), encoding="utf-8")
