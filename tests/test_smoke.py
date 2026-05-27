"""Basic smoke tests for the current grouped analysis entry points."""

from pathlib import Path

from core.analyzer import StrawberryAnalyzer
from core.grouping import PlantImageGroup


def test_analyzer_runs_group_pipeline() -> None:
    """The analyzer wrapper should delegate to the grouped pipeline."""
    analyzer = StrawberryAnalyzer()
    group = PlantImageGroup(
        sample_id="demo",
        top_image=Path("data/demo_TOP.png"),
        front_0_image=Path("data/demo-1.png"),
        front_180_image=Path("data/demo-2.png"),
    )

    result = analyzer.analyze_group(group)

    assert result.sample_id == "demo"
    assert result.status in {"dependency_error", "analysis_complete", "segmentation_failed", "load_failed"}
