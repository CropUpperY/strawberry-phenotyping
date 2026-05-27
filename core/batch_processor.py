"""Batch analysis helpers for grouped strawberry plant samples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.grouping import PlantImageGroup, group_image_files
from core.pipeline import PlantAnalysisResult, analyze_plant_group


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, PlantImageGroup, PlantAnalysisResult], None]


@dataclass(slots=True)
class BatchSampleResult:
    """One analyzed plant sample in a batch run."""

    group: PlantImageGroup
    result: PlantAnalysisResult


@dataclass(slots=True)
class BatchAnalysisReport:
    """Summary payload for one directory-level batch analysis."""

    directory: Path
    sample_results: list[BatchSampleResult]
    started_at: datetime
    finished_at: datetime

    @property
    def total_groups(self) -> int:
        """Return the total number of grouped samples."""

        return len(self.sample_results)

    @property
    def completed_groups(self) -> int:
        """Return the number of successfully analyzed groups."""

        return sum(1 for item in self.sample_results if item.result.status == "analysis_complete")

    @property
    def skipped_groups(self) -> int:
        """Return the number of skipped groups due to incomplete input."""

        return sum(1 for item in self.sample_results if item.result.status == "incomplete_input")

    @property
    def failed_groups(self) -> int:
        """Return the number of failed groups."""

        return self.total_groups - self.completed_groups - self.skipped_groups

    @property
    def duration_seconds(self) -> float:
        """Return the batch runtime in seconds."""

        return (self.finished_at - self.started_at).total_seconds()


def analyze_directory(
    directory_path: str | Path,
    *,
    emit_log: LogCallback | None = None,
    emit_progress: ProgressCallback | None = None,
    debug_output_dir: str | Path | None = None,
    **pipeline_kwargs: Any,
) -> BatchAnalysisReport:
    """Analyze all grouped samples under one directory."""

    directory = Path(directory_path).expanduser().resolve()
    groups = group_image_files(directory)
    return analyze_groups(
        groups,
        directory=directory,
        emit_log=emit_log,
        emit_progress=emit_progress,
        debug_output_dir=debug_output_dir,
        **pipeline_kwargs,
    )


def analyze_groups(
    groups: list[PlantImageGroup],
    *,
    directory: str | Path,
    emit_log: LogCallback | None = None,
    emit_progress: ProgressCallback | None = None,
    debug_output_dir: str | Path | None = None,
    **pipeline_kwargs: Any,
) -> BatchAnalysisReport:
    """Analyze a prepared list of grouped samples."""

    directory_path = Path(directory).expanduser().resolve()
    started_at = datetime.now()
    sample_results: list[BatchSampleResult] = []
    total_groups = len(groups)

    _emit(emit_log, f"开始批量分析目录: {directory_path}")
    _emit(emit_log, f"识别到 {total_groups} 个样本组。")

    for index, group in enumerate(groups, start=1):
        _emit(emit_log, f"[{index}/{total_groups}] 开始处理样本组 {group.sample_id}")
        try:
            result = analyze_plant_group(
                group,
                emit_log=emit_log,
                debug_output_dir=debug_output_dir,
                **pipeline_kwargs,
            )
        except Exception as error:  # noqa: BLE001
            result = _build_unexpected_failure_result(group, error)
            _emit(emit_log, f"[{index}/{total_groups}] 样本组 {group.sample_id} 异常终止: {error}")

        sample_result = BatchSampleResult(group=group, result=result)
        sample_results.append(sample_result)

        if emit_progress is not None:
            emit_progress(index, total_groups, group, result)

        _emit(
            emit_log,
            f"[{index}/{total_groups}] 样本组 {group.sample_id} 完成: status={result.status}",
        )

    report = BatchAnalysisReport(
        directory=directory_path,
        sample_results=sample_results,
        started_at=started_at,
        finished_at=datetime.now(),
    )
    _emit(
        emit_log,
        "批量分析完成。"
        f" 成功={report.completed_groups}, 跳过={report.skipped_groups}, 失败={report.failed_groups},"
        f" 用时={report.duration_seconds:.2f}s。",
    )
    return report


def _build_unexpected_failure_result(group: PlantImageGroup, error: Exception) -> PlantAnalysisResult:
    """Create a pipeline-like failure payload for unexpected batch errors."""

    from core.pipeline import TRAIT_SPECS, TraitResult, ViewLoadResult

    traits = [
        TraitResult(
            key=spec.key,
            label=spec.label,
            source_views=spec.source_views,
            unit=spec.unit,
            status="load_failed",
            message=f"Unexpected batch error: {error}",
        )
        for spec in TRAIT_SPECS
    ]
    return PlantAnalysisResult(
        sample_id=group.sample_id,
        status="load_failed",
        message=f"Batch processing raised an unexpected error: {error}",
        traits=traits,
        view_results={
            "TOP": ViewLoadResult("TOP", group.top_image, "load_failed", message=str(error)),
            "FRONT-1": ViewLoadResult("FRONT-1", group.front_0_image, "load_failed", message=str(error)),
            "FRONT-2": ViewLoadResult("FRONT-2", group.front_180_image, "load_failed", message=str(error)),
        },
        errors=[str(error)],
    )


def _emit(callback: LogCallback | None, message: str) -> None:
    """Emit one batch log line when a callback is available."""

    if callback is not None:
        callback(message)
