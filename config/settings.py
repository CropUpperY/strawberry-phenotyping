"""Application configuration definitions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Basic application settings."""

    app_name: str = "草莓RGB图像表型分析软件"
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")


DEFAULT_CONFIG = AppConfig()
