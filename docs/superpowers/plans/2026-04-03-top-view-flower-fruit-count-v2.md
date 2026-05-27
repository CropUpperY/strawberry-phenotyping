# 俯视角花朵与果实计数 V2 实现计划

## 一、目标

将当前基于纯颜色候选区的 `flower_count / fruit_count` 升级为：

- 花朵：`黄心锚点 + 白瓣结构 + 同花合并 + 多花分裂`
- 果实：`颜色候选 + 轮廓评分 + 粘连分裂`

保持以下边界不变：

- 仍然只使用 `TOP` 俯视图
- 仍然输出 `flower_count` 和 `fruit_count`
- 保持与现有 `core/pipeline.py`、`utils/exporter.py`、GUI 结果表兼容

## 二、代码落点

主要修改文件：

- `core/organs.py`
- `tests/test_organs.py`
- `tests/test_pipeline.py`

可能少量调整但原则上不应大改：

- `core/pipeline.py`
- `tests/test_exporter.py`

文档文件：

- `docs/superpowers/specs/2026-04-03-top-view-flower-fruit-count-v2-design.md`

## 三、执行步骤

### Task 1：重构花朵检测主流程

目标：

- 把 `detect_top_flowers()` 从“白色连通域计数”改成“黄心驱动的实例计数”

具体工作：

- 新增黄色花心候选提取函数
- 新增白瓣候选提取函数
- 新增花心与白瓣邻域匹配逻辑
- 新增同花合并逻辑
- 新增基于花心 marker 的粘连分裂逻辑
- 补充花朵调试图输出

完成标准：

- 纯白高光没有黄心支撑时不再计花
- 同一朵花裂成多个白块时不再重复计数

### Task 2：增强果实检测的轮廓级过滤

目标：

- 保留颜色候选，但不再让红色碎片轻易通过

具体工作：

- 拆分“颜色候选”和“实例筛选”两个阶段
- 强化面积、短轴、长宽比、solidity、圆整度过滤
- 优化粘连果实的 marker / watershed 分裂
- 补充果实调试图输出

完成标准：

- 细长红条、小碎片、边缘噪声不过筛
- 两个粘连成熟果在合理情况下能拆开

### Task 3：补全真实失败模式回归测试

目标：

- 不再只测理想二值图，要锁住当前真实误检模式

新增测试建议：

- `test_flower_detection_ignores_leaf_highlight_without_yellow_center`
- `test_flower_detection_merges_split_petals_around_one_center`
- `test_flower_detection_splits_two_flowers_with_two_centers`
- `test_fruit_detection_rejects_thin_red_fragment`
- `test_fruit_detection_splits_touching_red_fruits`

完成标准：

- 所有测试能在不依赖外部真实图片的前提下稳定运行

### Task 4：保持 pipeline 和导出链路稳定

目标：

- 替换算法实现后，不打坏现有结果链路

具体工作：

- 检查 `core/pipeline.py` 中花果识别调用点
- 保持 `TRAIT_SPECS`、`TraitResult`、导出字段不变
- 如有必要，更新 `message` 文案，明确“基于可见目标的实例计数”

完成标准：

- pipeline 仍能正常输出 `flower_count` 和 `fruit_count`
- GUI 结果表和导出文件无需额外适配即可读取新结果

### Task 5：验证与人工抽查

自动验证：

- `pytest tests/test_organs.py tests/test_pipeline.py tests/test_exporter.py -q`
- `pytest -q tests`

人工抽查：

- 用当前已知问题样本看调试图
- 重点确认：
  - 叶片反光不再算花
  - 单花裂片不再双计
  - FRONT / TOP 预览链路未被误伤

## 四、实现顺序建议

严格按以下顺序进行：

1. 先加花朵失败模式测试
2. 再改 `detect_top_flowers()`
3. 再加果实失败模式测试
4. 再改 `detect_top_fruits()`
5. 最后跑 pipeline / exporter / 全量测试

## 五、风险点

- 黄心在遮挡或曝光异常时可能不稳定，需要保留合理回退
- 花心阈值过严会少计，过宽会把幼叶/黄叶边缘误入
- 分裂逻辑过强会把单花拆成两朵，过弱会把双花合成一朵
- 调试图不完整会导致后续难以调参

## 六、完成判据

当满足以下条件时，这轮任务才算完成：

- 花朵检测不再以白色连通域直接计数
- 至少覆盖 3 个花朵失败模式回归测试
- 至少覆盖 2 个果实失败模式回归测试
- 全量测试通过
- 调试图能看出“黄心候选、白瓣候选、实例合并/分裂结果”
