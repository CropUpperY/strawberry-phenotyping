"""Tests for CSV and Excel export helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.batch_processor import BatchAnalysisReport, BatchSampleResult
from core.grouping import PlantImageGroup
from core.pipeline import PlantAnalysisResult, TraitResult
from utils.exporter import build_result_record, export_batch_report, export_single_result


def test_build_result_record_flattens_traits_and_statuses() -> None:
    """One analysis result should be flattened into a tabular record."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    result = _build_result()

    record = build_result_record(group, result)

    assert record["sample_id"] == "1AB"
    assert "result_status" not in record
    assert "result_message" not in record
    assert "group_is_complete" not in record
    assert "missing_views" not in record
    assert record["leaf_area"] == 12.34
    assert record["canopy_height"] == 7.89
    assert record["flower_count"] == 3
    assert record["fruit_count"] == 2
    assert "top_calibration_status" not in record
    assert "errors" not in record


def test_build_result_record_includes_debug_fields_when_enabled() -> None:
    """Debug mode export should include detailed status/calibration/error fields."""

    group = PlantImageGroup(
        sample_id="1AB",
        top_image=Path("data/1AB_TOP.png"),
        front_0_image=Path("data/1AB-1.png"),
        front_180_image=Path("data/1AB-2.png"),
    )
    result = _build_result()
    result.errors = ["TOP calibration fallback"]

    record = build_result_record(group, result, include_debug_fields=True)

    assert record["top_view_status"] == "loaded"
    assert record["top_calibration_status"] == "calibrated"
    assert record["error_count"] == 1
    assert "leaf_area_status" in record
    assert "leaf_area_message" in record


def test_export_single_result_writes_csv(tmp_path: Path) -> None:
    """Single-sample export should create a CSV file."""

    group = PlantImageGroup(sample_id="1AB")
    result = _build_result()

    export_path = export_single_result(group, result, tmp_path / "single.csv")

    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8-sig")
    assert "sample_id" in content
    assert "leaf_area" in content
    assert "top_calibration_status" not in content
    assert "errors" not in content


def test_export_batch_report_writes_excel(tmp_path: Path) -> None:
    """Batch export should create an Excel workbook when openpyxl is available."""

    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook

    group = PlantImageGroup(sample_id="1AB")
    result = _build_result()
    report = BatchAnalysisReport(
        directory=tmp_path,
        sample_results=[BatchSampleResult(group=group, result=result)],
        started_at=__import__("datetime").datetime.now(),
        finished_at=__import__("datetime").datetime.now(),
    )

    export_path = export_batch_report(report, tmp_path / "batch.xlsx")

    assert export_path.exists()
    workbook = load_workbook(export_path)
    worksheet = workbook.active
    headers = [cell.value for cell in worksheet[1]]
    assert "样本编号" in headers
    assert "结果状态" not in headers
    assert "叶面积指数(cm^2)" in headers
    assert "冠层高度(cm)" in headers
    assert "植株冠径(cm)" in headers
    assert "花朵数(count)" in headers
    assert "果实数(count)" in headers
    assert "sample_id" not in headers


def _build_result() -> PlantAnalysisResult:
    """Create a compact pipeline-like result payload for exporter tests."""

    traits = [
        TraitResult("leaf_area", "叶面积", ("TOP",), "cm^2", 12.34, "computed", "ok"),
        TraitResult("greenness", "绿色程度", ("TOP",), "ExG", 45.67, "computed", "ok"),
        TraitResult("convex_hull_area", "最小凸包面积", ("TOP",), "cm^2", 14.56, "computed", "ok"),
        TraitResult("canopy_height", "冠层高度", ("FRONT-1", "FRONT-2"), "cm", 7.89, "computed", "ok"),
        TraitResult("canopy_width", "植株冠径", ("TOP",), "cm", 8.9, "computed", "ok"),
        TraitResult("side_projection_area", "侧视投影面积", ("FRONT-1", "FRONT-2"), "cm^2", 9.87, "computed", "ok"),
        TraitResult("flower_count", "花朵数", ("TOP",), "count", 3, "computed", "ok"),
        TraitResult("fruit_count", "果实数", ("TOP",), "count", 2, "computed", "ok"),
    ]
    return PlantAnalysisResult(
        sample_id="1AB",
        status="analysis_complete",
        message="done",
        traits=traits,
        view_results={
            "TOP": SimpleNamespace(status="loaded"),
            "FRONT-1": SimpleNamespace(status="loaded"),
            "FRONT-2": SimpleNamespace(status="loaded"),
        },
        calibration_results={
            "TOP": SimpleNamespace(status="calibrated", mm_per_pixel=0.2),
            "FRONT-1": SimpleNamespace(status="calibrated", mm_per_pixel=0.2),
            "FRONT-2": SimpleNamespace(status="calibrated", mm_per_pixel=0.2),
        },
        errors=[],
    )
