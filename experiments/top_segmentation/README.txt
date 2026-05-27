experiments/top_segmentation 目录说明

这个目录用于存放 TOP 视角分割相关实验脚本。它们的目标是“先拆分问题、再验证方案”，
不是直接给正式软件主流程调用。

当前主要脚本说明如下：

1. compare_flower_masks.py
作用：
- 对比“纯轮廓法”和“纯颜色法”两种花器官候选提取方式。
- 主要用于观察：不依赖绿色主掩膜时，结构与颜色各自会抓到什么。

典型输出：
- 02_contour_only_mask.png：纯轮廓掩膜
- 04_color_only_mask.png：纯颜色掩膜
- 06_union_mask.png：二者并集
- 07_montage.png：总览拼图

默认输出目录：
- experiments/top_segmentation/output/flower_mask_compare/<name>


2. debug_flower_preserving_top_mask.py
作用：
- 复现当前正式 TOP 主掩膜结果；
- 单独提取白/黄花器官候选；
- 叠加“救回逻辑”并加入 hard-negative + 离散组件过滤，检查误检/漏检。

典型输出：
- 02_current_top_mask.png：当前正式流程主掩膜
- 13_removed_by_hard_negative.png：被 hard-negative 抑制的区域
- 15_removed_non_green_islands.png：被离散组件规则剔除的区域
- 19_augmented_mask.png：增强后掩膜
- 90_montage.png：总览拼图

默认输出目录：
- experiments/top_segmentation/output/flower_preserving_top_mask/<name>


3. recover_shadow_leaf_mask.py
作用：
- 针对 TOP 图中“阴影叶片漏分”做实验。
- 通过更宽松的绿色条件与边缘/纹理约束，尝试恢复暗部叶片。


4. refine_top_mask_with_grabcut.py
作用：
- 在已有 TOP 主掩膜基础上，用 GrabCut 做前景/背景细化。


5. remove_right_color_card_mask.py
作用：
- 单独验证 TOP 掩膜里“右侧色卡误保留”的剔除逻辑。


6. merge_flower_into_main_mask.py
作用：
- 在“主掩膜阶段”直接并入白/黄花器官候选（不是后处理救回）。
- 与基线结果并排对比新增/减少区域。

默认输出目录：
- experiments/top_segmentation/output/merge_flower_into_main_mask/<image-stem>


7. output/
作用：
- 保存实验脚本生成的中间图和结果图。
- 每个实验子目录通常会包含 `说明.txt` 与 `summary.json`。


建议查看顺序：
1. 先看输出目录中的 `说明.txt`
2. 再看 `90_montage.png`（或类似总览图）
3. 如需细查，再回看中间步骤图
