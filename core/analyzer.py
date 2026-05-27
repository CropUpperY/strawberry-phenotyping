"""Compatibility wrapper around the grouped analysis pipeline."""

from __future__ import annotations

from core.grouping import PlantImageGroup
from core.pipeline import PlantAnalysisResult, analyze_plant_group


class StrawberryAnalyzer:
    """Thin wrapper for the current grouped plant analysis pipeline."""

    def analyze_group(self, group: PlantImageGroup) -> PlantAnalysisResult:
        """Analyze a three-view strawberry plant group."""
        return analyze_plant_group(group)
