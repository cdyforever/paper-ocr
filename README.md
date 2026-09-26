# Canpoint OCR

## 纯 ONNX 方案

本项目为**全链路纯 ONNX 推理**（不依赖 PaddlePaddle / torch / transformers 运行时）：

| 环节 | 模型 | 来源 |
|------|------|------|
| 文本检测 | DBNet | PaddleOCR PP-OCRv4 |
| 文本识别 | PP-OCRv4 Rec | PaddleOCR PP-OCRv4 |
| 公式检测 | YOLOv8 MFD | MinerU / PDF-Extract-Kit |
| 公式识别 | UniMERNet-tiny | opendatalab/UniMERNet |

**推荐入口**：`func/algorithm/onnx_page_pipeline.py` 的 `OnnxPagePipeline` —
页面级「公式检测 → 涂白 → 文本/公式双路径识别 → 阅读顺序合并」流程。实测 DBNet 用文本行的先验去切公式
会结构性失效（公式区域平均只有 43.9% 面积落在行框内），解耦后公式识别
**51.2% → 93.7%**。

推理流程图（可交互 HTML，含深浅主题与导览视图）：[page_inference_pipeline.html](page_inference_pipeline.html)

- 单模块推理: `func/algorithm/onnx_ocr_engine.py`、`formula_detect_onnx.py`、`formula_recognize_onnx.py`
- 导出/验证脚本: `onnx_pipeline/`（见 [onnx_pipeline/README.md](onnx_pipeline/README.md)）
- 详细报告:
  - [ACCURACY_VALIDATION_REPORT.md](ACCURACY_VALIDATION_REPORT.md) — **真实文档精度实测 + 页面级解耦**
  - [PADDLEOCR_TO_ONNX_REPORT.md](PADDLEOCR_TO_ONNX_REPORT.md) — 文本检测/识别转换
  - [FORMULA_DETECT_MODERN_REPORT.md](FORMULA_DETECT_MODERN_REPORT.md) — 公式检测方案调研
  - [UNIMERNET_ONNX_REPORT.md](UNIMERNET_ONNX_REPORT.md) — 公式识别部署

> 模型权重（`weights/`，约 1.1 GB）不入库，用 `onnx_pipeline/export/` 下的脚本重新生成。

## 如何使用

```bash
# 整页推理（默认 test.jpg，可传入图片路径或目录）
python func/algorithm/onnx_page_pipeline.py <图片路径>
```

```python
from func.algorithm.onnx_page_pipeline import OnnxPagePipeline

pipe = OnnxPagePipeline()
items, debug = pipe('test.jpg', return_debug=True)   # items: text + formula 按阅读顺序
```

## future

1. 增加用户账号系统及收费模式
2. 增加后台log
3. 增加输入数据校验
4. Web 服务接口重建（原 tornado 服务依赖已废弃的 paddle 链路，已随清理移除，需要时基于 OnnxPagePipeline 重建）
