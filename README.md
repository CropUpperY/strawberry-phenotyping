# 草莓RGB图像表型分析软件

用于开发草莓RGB图像表型分析软件的 Python 项目骨架。

## 项目结构

- `config/`：配置文件与参数管理
- `core/`：核心分析逻辑
- `utils/`：通用工具函数
- `gui/`：桌面界面相关代码
- `tests/`：测试代码
- `data/`：示例数据、输入输出数据目录

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

## 后续建议

1. 在 `config/settings.py` 中补充模型参数、路径和阈值配置。
2. 在 `core/analyzer.py` 中实现草莓表型分析流程。
3. 在 `gui/main_window.py` 中接入图像加载、结果展示和批处理功能。
4. 在 `tests/` 中逐步补充单元测试与集成测试。
