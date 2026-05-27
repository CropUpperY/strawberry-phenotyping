# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Environment Setup
- Install dependencies: `pip install -r requirements.txt`

### Running the Application
- Start GUI: `python main.py`

### Testing
- Run all tests: `pytest -q tests` (Note: Always specify the `tests` directory to avoid permission issues in `output/`)
- Run a single test file: `pytest tests/test_pipeline.py`
- Run a specific test: `pytest tests/test_segmentation.py::test_segment_top_view`

## Architecture and Code Structure

### Core Logic (`core/`)
- `grouping.py`: Scans directories and groups images by sample name (`<sample>_TOP.png`, `<sample>-1.png`, `<sample>-2.png`).
- `calibration.py`: Color correction and scale estimation (`mm_per_pixel`) using a 24-color checker card.
- `segmentation.py`: Plant segmentation from background; generates masks, contours, and bounding boxes.
- `traits.py`: Calculation of phenotypic traits (leaf area, greenness, convex hull area, plant height/width, side projection area).
- `pipeline.py`: Orchestrates the full analysis flow for a single sample (load -> calibrate -> segment -> calculate -> export).
- `batch_processor.py`: Handles processing of multiple samples in a directory.
- `visualization.py`: Generation of debug/preview images (masks, contours, montages) for the GUI.

### User Interface (`gui/`)
- `main_window.py`: PyQt5 main window. Uses background threads for analysis to maintain responsiveness.
- **Design Pattern**: Visualization logic is separated into `core/visualization.py`; GUI components only handle display and binding.

### Utilities and Configuration
- `config/settings.py`: Thresholds, model parameters, and path configurations.
- `utils/`: Exporting results to Excel/CSV, logging, and debug artifact management.

### Data and Output
- `data/`: Sample input images.
- `output/debug/`: Intermediate visualization results for debugging.
- `output/exports/`: Final analysis result files.
- `logs/gui.log`: Runtime logs for the GUI application.

## Development Guidelines
- **Concurrency**: Always perform heavy analysis in background threads, never on the main GUI thread.
- **Visualization**: When adding new preview features, update `core/visualization.py` first, then bind in `gui/`.
- **Testing**: Add or update tests in `tests/` after modifying core segmentation or calibration logic.
- **File Naming**: Follow the `<sample>_TOP/FRONT` pattern for input data compatibility.