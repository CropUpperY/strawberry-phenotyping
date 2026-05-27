# 色卡色块实验说明

这个目录当前保留了两个用途不同的色卡实验脚本。

## 当前结构

- `detect_patch_grid_from_best_variant.py`
  - 旧基线实验。
  - 输入：已经完成方向选择的 `variant_最佳变体` 图像。
  - 作用：检测可见色块、推断 4×6 网格，并输出网格分配可视化结果。
  - 输出目录：`experiments/patch_detection/output/detect_patch_grid_from_best_variant/`

- `rectify_warped_card_from_patch_grid.py`
  - 当前主实验。
  - 输入：`variant_原始透视校正` 图像。
  - 作用：先选择最佳方向，再拟合色块网格，并执行第二次基于网格的单应校正；同时比较矫正前后的网格质量和颜色拟合质量。
  - 输出目录：`experiments/patch_detection/output/rectify_warped_card_from_patch_grid/`

## 这样命名的原因

- 旧名字 `detect_patch_grid_v2.py` 是按版本命名的，但已经不能直观看出它的职责。
- 旧名字 `rectify_patch_grid_v1.py` 容易误导，因为它实际上是更新、更完整的实验脚本。
- 现在统一改成按“输入是什么、脚本做什么”来命名，避免以后继续出现 v1、v2、v3 这种语义不清的名字。

## 建议使用方式

当前如果要验证主流程里“原始透视校正后再做色块网格二次校正”的效果，优先使用 `rectify_warped_card_from_patch_grid.py`。

`detect_patch_grid_from_best_variant.py` 只保留为早期 patch 网格检测阶段的对照基线，不建议继续作为主实验入口。