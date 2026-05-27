"""CSV and Excel export helpers for phenotype analysis results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.batch_processor import BatchAnalysisReport, BatchSampleResult
from core.grouping import PlantImageGroup
from core.pipeline import PlantAnalysisResult, TRAIT_SPECS


def build_result_record(
    group: PlantImageGroup,
    result: PlantAnalysisResult,
    *,
    include_debug_fields: bool = False,
) -> dict[str, Any]:
    """Flatten one grouped analysis result into a tabular record."""

    trait_map = result.trait_map()
    if include_debug_fields:
        record: dict[str, Any] = {
            "sample_id": group.sample_id,
            "result_status": result.status,
            "result_message": result.message,
            "group_is_complete": group.is_complete,
            "missing_views": ",".join(group.missing_views),
        }
    else:
        record = {
            "sample_id": group.sample_id,
        }

    if include_debug_fields:
        record.update(
            {
                "top_image": str(group.top_image) if group.top_image else "",
                "front_1_image": str(group.front_0_image) if group.front_0_image else "",
                "front_2_image": str(group.front_180_image) if group.front_180_image else "",
                "top_view_status": _view_status(result, "TOP"),
                "front_1_view_status": _view_status(result, "FRONT-1"),
                "front_2_view_status": _view_status(result, "FRONT-2"),
                "top_calibration_status": _calibration_status(result, "TOP"),
                "front_1_calibration_status": _calibration_status(result, "FRONT-1"),
                "front_2_calibration_status": _calibration_status(result, "FRONT-2"),
                "top_mm_per_pixel": _calibration_value(result, "TOP", "mm_per_pixel"),
                "front_1_mm_per_pixel": _calibration_value(result, "FRONT-1", "mm_per_pixel"),
                "front_2_mm_per_pixel": _calibration_value(result, "FRONT-2", "mm_per_pixel"),
                "error_count": len(result.errors),
                "errors": " | ".join(result.errors),
            }
        )

    for trait_key, trait in trait_map.items():
        if include_debug_fields:
            record[f"{trait_key}_value"] = trait.value
            record[f"{trait_key}_unit"] = trait.unit
            record[f"{trait_key}_status"] = trait.status
            record[f"{trait_key}_message"] = trait.message
        else:
            record[trait_key] = trait.value

    return record


def build_batch_records(report: BatchAnalysisReport, *, include_debug_fields: bool = False) -> list[dict[str, Any]]:
    """Flatten one batch report into exportable tabular rows."""

    return [
        build_result_record(item.group, item.result, include_debug_fields=include_debug_fields)
        for item in report.sample_results
    ]


def export_batch_report(report: BatchAnalysisReport, output_path: str | Path, *, include_debug_fields: bool = False) -> Path:
    """Export a batch report to CSV or Excel based on file suffix."""

    return export_records(build_batch_records(report, include_debug_fields=include_debug_fields), output_path)


def export_single_result(
    group: PlantImageGroup,
    result: PlantAnalysisResult,
    output_path: str | Path,
    *,
    include_debug_fields: bool = False,
) -> Path:
    """Export a single sample result to CSV or Excel."""

    return export_records(
        [build_result_record(group, result, include_debug_fields=include_debug_fields)],
        output_path,
    )


def export_records(records: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """Export generic analysis records to CSV or Excel."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        _write_csv(path, records)
        return path
    if suffix in {".xlsx", ".xlsm"}:
        _write_excel(path, records)
        return path
    raise ValueError(f"Unsupported export format: {path.suffix}")


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records to a UTF-8 CSV file."""

    fieldnames = _collect_fieldnames(records)
    with path.open("w", encoding="utf-8-sig", newline="") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_excel(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records to an Excel workbook."""

    try:
        from openpyxl import Workbook
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("Excel export requires openpyxl to be installed.") from error

    fieldnames = _collect_fieldnames(records)
    chinese_headers = [_to_chinese_header(field_name) for field_name in fieldnames]
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "表型结果"
    worksheet.append(chinese_headers)

    for record in records:
        worksheet.append([record.get(field_name, "") for field_name in fieldnames])

    workbook.save(path)


def _collect_fieldnames(records: Sequence[dict[str, Any]]) -> list[str]:
    """Collect stable fieldnames from all records."""

    include_debug_fields = any("result_status" in record for record in records)
    if include_debug_fields:
        preferred_order = [
            "sample_id",
            "result_status",
            "result_message",
            "group_is_complete",
            "missing_views",
        ]
    else:
        preferred_order = ["sample_id"]

    all_keys: list[str] = []
    seen: set[str] = set()
    for key in preferred_order:
        seen.add(key)
        all_keys.append(key)

    for key in _iter_record_keys(records):
        if key not in seen:
            seen.add(key)
            all_keys.append(key)
    return all_keys


def _iter_record_keys(records: Sequence[dict[str, Any]]) -> Iterable[str]:
    """Iterate keys from the provided records in encounter order."""

    for record in records:
        for key in record:
            yield key


def _view_status(result: PlantAnalysisResult, view_name: str) -> str:
    """Return the load status for a named view."""

    view_result = result.view_results.get(view_name)
    return view_result.status if view_result is not None else ""


def _calibration_status(result: PlantAnalysisResult, view_name: str) -> str:
    """Return the calibration status for a named view."""

    calibration = result.calibration_results.get(view_name)
    return getattr(calibration, "status", "")


def _calibration_value(result: PlantAnalysisResult, view_name: str, attribute_name: str) -> Any:
    """Return one calibration attribute value if available."""

    calibration = result.calibration_results.get(view_name)
    return getattr(calibration, attribute_name, None)


def _to_chinese_header(field_name: str) -> str:
    """Convert internal field names to Chinese Excel headers."""

    base_map = {
        "sample_id": "样本编号",
        "result_status": "结果状态",
        "result_message": "结果说明",
        "group_is_complete": "样本组完整",
        "missing_views": "缺失视角",
        "top_image": "TOP图像路径",
        "front_1_image": "FRONT-1图像路径",
        "front_2_image": "FRONT-2图像路径",
        "top_view_status": "TOP视角状态",
        "front_1_view_status": "FRONT-1视角状态",
        "front_2_view_status": "FRONT-2视角状态",
        "top_calibration_status": "TOP校准状态",
        "front_1_calibration_status": "FRONT-1校准状态",
        "front_2_calibration_status": "FRONT-2校准状态",
        "top_mm_per_pixel": "TOP毫米每像素",
        "front_1_mm_per_pixel": "FRONT-1毫米每像素",
        "front_2_mm_per_pixel": "FRONT-2毫米每像素",
        "error_count": "错误数量",
        "errors": "错误信息",
    }
    if field_name in base_map:
        return base_map[field_name]

    trait_label_map = {spec.key: spec.label for spec in TRAIT_SPECS}
    trait_label_map["leaf_area"] = "叶面积指数"
    trait_unit_map = {spec.key: spec.unit for spec in TRAIT_SPECS}
    for suffix, suffix_name in (
        ("_value", "数值"),
        ("_unit", "单位"),
        ("_status", "状态"),
        ("_message", "说明"),
    ):
        if field_name.endswith(suffix):
            trait_key = field_name[: -len(suffix)]
            trait_label = trait_label_map.get(trait_key, trait_key)
            return f"{trait_label}{suffix_name}"

    if field_name in trait_label_map:
        trait_label = trait_label_map[field_name]
        trait_unit = trait_unit_map.get(field_name, "")
        return f"{trait_label}({trait_unit})" if trait_unit else trait_label

    return field_name
