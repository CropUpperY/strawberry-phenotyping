# Source-Sink Ratio Design

Date: 2026-05-29

## Goal

Add a new source-sink relationship metric for strawberry TOP-view analysis by introducing:

- `fruit_area`: visible TOP-view projected fruit area
- `source_sink_ratio`: `leaf_area / fruit_area`

The implementation should follow existing pipeline/result/export patterns and remain compatible with the current GUI and tests.

## Confirmed Decisions

The following decisions were confirmed with the user during design:

1. `fruit_area` means the sum of visible fruit projected area in the TOP view.
2. `fruit_area` should be computed from fruit masks, not YOLO bounding-box area.
3. When no fruit is detected, `source_sink_ratio` should be empty (`None`).
4. If TOP scale calibration is unavailable:
   - `fruit_area` falls back to `px^2`
   - `source_sink_ratio` is still computed as a unitless ratio

## Current Project Context

The current repository already computes:

- `leaf_area` from TOP segmentation in [core/traits.py](/D:/code/pycharm/strawberry/core/traits.py:1)
- `fruit_count` from TOP organ detection in [core/organs.py](/D:/code/pycharm/strawberry/core/organs.py:1) and optionally [core/yolo_counter.py](/D:/code/pycharm/strawberry/core/yolo_counter.py:1)
- result aggregation in [core/pipeline.py](/D:/code/pycharm/strawberry/core/pipeline.py:1)
- CSV/Excel export in [utils/exporter.py](/D:/code/pycharm/strawberry/utils/exporter.py:1)
- table and trait display in [gui/main_window.py](/D:/code/pycharm/strawberry/gui/main_window.py:1)

This feature should fit those existing boundaries instead of introducing a new parallel result path.

## Metric Definitions

### `fruit_area`

Definition:

- The number of non-zero pixels in the visible TOP fruit mask, summed across all detected fruits in the sample.

Source of truth:

- Prefer the classic fruit detector result mask from `detect_top_fruits(...)`.
- The mask already represents visible fruit pixels and aligns with the user's requested definition better than YOLO boxes.

Units:

- `cm^2` when TOP calibration is available
- `px^2` when TOP calibration is unavailable

### `source_sink_ratio`

Definition:

- `leaf_area / fruit_area`

Output behavior:

- Unit is empty string because it is a unitless ratio.
- If `fruit_area` is zero or fruit detection is unavailable, set value to `None`.

Important property:

- The ratio is valid in both calibrated and uncalibrated cases because matching area units cancel out.

## Recommended Approach

### Option A: Mask-based area from the classic fruit detector

Use the existing fruit detector mask to compute projected area while leaving count logic unchanged.

Pros:

- Matches the requested biological definition
- Uses already available mask outputs
- Minimizes architectural change
- Keeps implementation understandable

Cons:

- When YOLO is active, `fruit_count` and `fruit_area` may come from different detectors

### Option B: YOLO bounding-box area

Use summed YOLO fruit-box area as a proxy.

Pros:

- Fastest to implement

Cons:

- Does not match the requested projected-area definition
- Likely overestimates area
- Sensitive to loose boxes and overlap

### Option C: YOLO-guided local segmentation

Detect fruits with YOLO and segment inside each fruit box.

Pros:

- Could align count and area more tightly

Cons:

- Highest complexity
- More tuning and testing burden
- Unnecessary for the current requested scope

### Recommendation

Use Option A.

This is the best fit for the current repository and the user's requested meaning of fruit area.

## Architecture Changes

### 1. Add new trait specs

Update `TRAIT_SPECS` in [core/pipeline.py](/D:/code/pycharm/strawberry/core/pipeline.py:1) to include:

- `fruit_area`
- `source_sink_ratio`

Both should be TOP-view traits.

Suggested labels:

- `fruit_area`: `果实面积`
- `source_sink_ratio`: `源库比`

Suggested default units:

- `fruit_area`: `cm^2`
- `source_sink_ratio`: `""`

The runtime logic will still replace `fruit_area` with `px^2` when calibration is missing.

### 2. Keep area computation inside pipeline aggregation

Do not move `fruit_area` into [core/traits.py](/D:/code/pycharm/strawberry/core/traits.py:1).

Reason:

- `leaf_area` is derived from plant segmentation
- `fruit_area` is derived from organ detection output
- pipeline aggregation is already the place where organ-derived metrics are attached to the final result

This preserves existing responsibilities:

- `traits.py`: segmentation-derived plant measurements
- `pipeline.py`: final result assembly and detector-dependent derived traits

### 3. Extend TOP organ aggregation

The current `_apply_top_organ_counts(...)` logic in [core/pipeline.py](/D:/code/pycharm/strawberry/core/pipeline.py:1) should be extended or refactored so the same aggregation stage updates:

- `flower_count`
- `flower_bud_count`
- `fruit_count`
- `fruit_area`
- `source_sink_ratio`

Recommended refactor:

- rename the helper to something broader such as `_apply_top_organ_traits(...)`
- keep the function responsible for all organ-derived TOP traits

This is optional naming-wise, but the broader responsibility should be explicit in code.

## Detailed Computation Rules

### Fruit area rule

Inputs:

- TOP calibration result
- classic fruit detection result mask, if available

Computation:

1. If classic fruit detection result exists and includes a valid single-channel mask:
   - compute `fruit_area_pixels = count_nonzero(mask > 0)`
2. If TOP calibration is available:
   - convert pixels to `cm^2`
3. Else:
   - keep value in `px^2`

Status/message behavior:

- If fruit detection succeeded:
  - `status = "computed"`
- If fruit detection failed or is missing:
  - `value = None`
  - `status = "segmentation_failed"`
  - message should explain that fruit-area extraction did not complete

### Source-sink ratio rule

Inputs:

- computed `leaf_area`
- computed `fruit_area`

Computation:

1. If `fruit_area` is `None`, set ratio to `None`
2. If `fruit_area` is `0`, set ratio to `None`
3. Else compute `leaf_area / fruit_area`

Formatting:

- store as float rounded in the same style as other numeric traits
- use empty unit string

Status/message behavior:

- `computed` when ratio is available
- `segmentation_failed` when fruit detection failed
- `not_detected` or `computed` with empty value is not recommended here because the current pipeline already uses `segmentation_failed` for failed fruit trait extraction

## Detector Interaction Policy

The repository currently supports two fruit-related paths:

- classic color-based fruit detector
- YOLO combined organ detector

For this feature:

- `fruit_count` should keep its current behavior
- `fruit_area` should come from the classic fruit detector mask

Implementation consequence:

- when YOLO is enabled, the pipeline may still need classic fruit detection to produce `fruit_area`
- this is acceptable for the current scope because the metric definition depends on a pixel mask, not on YOLO boxes

If performance later becomes a concern, a future follow-up can introduce YOLO-guided mask extraction without changing the public trait definition.

## Export Behavior

Update [utils/exporter.py](/D:/code/pycharm/strawberry/utils/exporter.py:1) so the new TOP traits export naturally alongside existing TOP metrics.

Expected behavior:

- TOP rows include `fruit_area` and `source_sink_ratio`
- FRONT rows do not include them because they are TOP-only traits
- both simple export and debug export continue to work without special-case branching beyond existing trait iteration

Excel header mapping should add labels for:

- `fruit_area`
- `source_sink_ratio`

## GUI Behavior

No special GUI layout redesign is required.

Expected behavior after trait-spec updates:

- the result table should automatically show the new TOP traits
- the trait focus panel should display them like other traits
- TOP view filtering in the existing result table should include them automatically because both are TOP-view traits

No new preview pane or visualization card is required for this feature.

## Error Handling

The feature should preserve the repository's current tolerant behavior:

- fruit-related metric failures must not discard already computed canopy metrics
- if fruit detection fails, `leaf_area` and other plant traits still complete
- `fruit_area` and `source_sink_ratio` should remain empty and carry clear status/message text

This mirrors the current behavior already covered by pipeline tests for failed fruit detection.

## Testing Strategy

### Pipeline tests

Update [tests/test_pipeline.py](/D:/code/pycharm/strawberry/tests/test_pipeline.py:1) to cover:

1. calibrated case
   - `fruit_area` in `cm^2`
   - `source_sink_ratio` computed
2. uncalibrated case
   - `fruit_area` in `px^2`
   - `source_sink_ratio` still computed
3. no fruit detected
   - `fruit_area` equals zero or empty according to actual detector payload used
   - `source_sink_ratio is None`
4. fruit detector failure
   - `fruit_area is None`
   - `source_sink_ratio is None`
   - unrelated traits remain computed

Test helpers should provide fruit masks with known pixel counts so assertions remain deterministic.

### Export tests

Update [tests/test_exporter.py](/D:/code/pycharm/strawberry/tests/test_exporter.py:1) to verify:

- flattened records include `fruit_area` and `source_sink_ratio`
- TOP view rows include the new fields
- FRONT rows do not expose TOP-only traits
- Excel headers contain the localized labels

### GUI smoke tests

Update [tests/test_gui_smoke.py](/D:/code/pycharm/strawberry/tests/test_gui_smoke.py:1) only where trait counts or expected TOP-table contents depend on the trait list.

No GUI-specific behavioral tests are required beyond ensuring the new traits render without breaking the current table and focus-panel logic.

## Out of Scope

The following are intentionally not part of this change:

- estimating true 3D fruit surface area or volume
- adding non-red fruit pixel area
- aligning YOLO count and mask area into a unified detector
- adding new phenotype preview overlays specifically for fruit area
- adding physiological interpretation text beyond the raw ratio metric

## Implementation Notes

The least risky implementation order is:

1. add trait specs
2. extend pipeline aggregation logic
3. update exporter headers/tests
4. update GUI smoke expectations
5. run targeted pipeline/exporter/gui tests

This keeps the feature incremental and easy to verify.
