# TOP 掩膜与花器官提取问题清单（持续跟进）

日期：2026-04-03  
范围：`experiments/top_segmentation/*` 与后续准备回灌到主流程的 TOP 分割/花器官逻辑

## 目标
把当前“花瓣/花蕊被误删、花盆边缘误入、土壤误入、断裂叶片漏分”等问题拆成可逐步实施的任务，避免一次改太多导致回归。

---

## Issue 01：`39_removed_top_pot_band` 被移除后又回来了

### 现象
`39_removed_top_pot_band.png` 显示被判为顶部花盆边缘并移除，但最终增强掩膜又出现这部分区域。

### 当前根因
在 `debug_flower_preserving_top_mask.py` 中：
- `base_mask` 先移除了顶部花盆边缘；
- 但后续 `augmented_mask = base_mask OR flower_near_canopy`；
- `flower_near_canopy` 来自白/黄候选，仅受距离约束，不受“排除花盆边缘”约束；
- 因此被移除区域可能被“救回逻辑”重新加入。

### 方案
在“救回逻辑”增加反向约束：
1. 排除与圆形花盆边缘候选高重叠的 rescue 分量；
2. 或把 `removed_top_pot_band` 当作 hard-negative，不允许被 `flower_near_canopy` 重新并入。

优先级：P0

---

## Issue 02：顶部花盆边缘处理原理不够直观

### 当前原理（正式流程）
`core/segmentation.py::_remove_top_attached_pot_band` 使用“顶部浅层条带 + 列扫描 + 宽度阈值”的启发式规则：
- 候选需靠近图像上缘；
- 垂直深度浅；
- 与主植株之间有间隔；
- 水平连续宽度达到阈值才移除。

### 局限
这是“顶部条带”假设，不是“花盆圆环”几何检测，泛化有限。

优先级：P1（说明已清楚，核心是升级算法）

---

## Issue 03：花盆不一定在顶部，应检测整圈圆环

### 目标
把“顶部条带规则”升级为“全局花盆圆环抑制”。

### 方案草案
1. 在暗背景区域提取边缘；
2. 用圆/椭圆检测（HoughCircle + 椭圆拟合 fallback）提 ring 候选；
3. 结合颜色与纹理过滤（灰绿塑料盆边常见颜色/亮度带）；
4. 输出 `pot_ring_mask`；
5. 在基线掩膜和 rescue 阶段都作为抑制先验。

优先级：P0

---

## Issue 04：花盆外散土被识别进来

### 目标
排除“独立于主植株且非绿色”的土壤/背景碎片。

### 方案草案
对不连通组件做组件级分类：
1. 先找主植株核心连通域（最大绿域）；
2. 对每个离散组件计算：
   - 与主植株距离；
   - 到主植株的细桥接路径可达性；
   - 绿色比例（HSV+ExG）；
   - 细长度/纹理特征；
3. 规则：
   - `非绿色 + 远离 + 无合理桥接` -> 删除；
   - `绿色且距主植株近` -> 保留候选叶片；
   - `花器官颜色 + 靠近已知花梗结构` -> 保留候选花器官。

优先级：P0

---

## Issue 05：`1AB_TOP` 现在反而带入花盆边缘

### 现象
之前好图（如 `1AB_TOP`）在新实验下引入了花盆边缘。

### 可能原因
与 Issue 01 关联显著：被移除区域在 rescue 阶段重新并回；此外白/黄阈值与距离阈值偏宽，可能把盆边高光当作花器官候选。

### 方案
在修复 Issue 01 的同时增加 `1AB_TOP` 回归对照：
- 指标：花盆边缘误入像素下降；
- 不降低对 `111AB_TOP` 红框花器官的召回。

优先级：P0

---

## Issue 06：可利用轮廓边缘补全暗部叶片

### 现象
`flower-mask-compare/02_contour_only_mask.png` 中，暗部/遮挡处有可用边缘信息。

### 方案草案
构建“颜色种子 + 边缘扩展”双通路：
1. 颜色生成高精度绿种子；
2. 在局部边缘图上做 geodesic/region-growing 扩展；
3. 扩展仅在“靠近种子且满足叶片纹理”区域允许。

优先级：P1

---

## Issue 07：不连通叶片区域桥接回主掩膜

### 目标
把被阴影或遮挡切断的同一叶片区域重连。

### 方案草案
对“绿色离散小岛”做桥接：
1. 对每个离散绿色组件找最近主掩膜边界点；
2. 在代价图（暗背景代价高、绿色/边缘代价低）上跑最短路径；
3. 路径总代价低于阈值则桥接；
4. 桥接后做细化与最小宽度约束，防止跨背景乱连。

优先级：P1

---

## Issue 08：输出目录命名与说明 txt 不一致

### 现象
`output/experiments` 同时出现 `111AB` 和 `111AB_TOP`，且部分新目录没有说明 txt。

### 根因
1. `111AB` 目录来自手动指定 `--output-dir` 的历史运行；  
2. `111AB_TOP` 目录来自默认输出（按图片 stem 命名）；  
3. 之前说明 txt 是人工补写，不是脚本自动生成。

### 已落地改动
已把“说明 txt 自动生成”写入以下脚本：
- `experiments/top_segmentation/compare_flower_masks.py`
- `experiments/top_segmentation/debug_flower_preserving_top_mask.py`

每次运行会自动写入 `说明.txt`（以及 `rescued_components/说明.txt`）。

### 后续建议
统一命名策略（可选）：
1. 默认始终用 `image_stem`（当前）；
2. 增加 `--sample-id` 参数；
3. 增加 `--folder-mode {stem,sample}` 做显式控制。

优先级：P0（可视化流程一致性）

---

## 分阶段执行建议

### 阶段 A（先稳住误检）
1. 修复 Issue 01（被移除花盆边缘回流）
2. 修复 Issue 04（散土非绿色离散组件剔除）
3. 回归验证 Issue 05（`1AB_TOP` 不退化）

### 阶段 B（增强召回）
1. Issue 03（花盆整圈检测）
2. Issue 06（边缘信息补全）
3. Issue 07（离散叶片桥接）

### 阶段 C（回灌主流程）
1. 参数冻结与回归测试集固化（`1AB_TOP/2AB_TOP/111AB_TOP`）
2. 接入 `core/segmentation.py` 与 `core/pipeline.py`
3. GUI 结果卡与导出说明同步更新

---

## 当前状态
- 本文档已建立，可作为后续逐步读取与执行的唯一问题清单。
- 下一步建议从“阶段 A 第 1 项（Issue 01）”开始。

### 2026-04-03 实施进展（本轮）
- 已在 `experiments/top_segmentation/debug_flower_preserving_top_mask.py` 落地：
  - Issue 01：增加 hard-negative 抑制，默认使用 `removed_top_pot_band`，避免被移除花盆边缘在 rescue 阶段回流；
  - Issue 04：增加离散组件过滤（邻域绿色上下文 + 面积上限 + 距离约束），用于剔除散土/背景误入；
  - Issue 05：完成 `111AB_TOP` 与 `1AB_TOP` 回归跑图，输出目录为  
    `experiments/top_segmentation/output/flower_preserving_top_mask/{111AB,1AB}`；
  - Issue 08：新增 `--folder-mode {stem,sample}`、`--sample-id`、`--output-root`，并统一自动生成 `说明.txt`。
- 已在 `experiments/top_segmentation/compare_flower_masks.py` 落地：
  - Issue 08：新增同样的目录命名参数与说明文件自动生成；
  - 默认输出改为 `experiments/top_segmentation/output/flower_mask_compare/<name>`。
- `experiments/top_segmentation/README.txt` 已修复乱码并更新为当前脚本行为说明。
