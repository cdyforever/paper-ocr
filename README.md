# Canpoint OCR

## 纯 ONNX 方案

本项目已支持**全链路纯 ONNX 推理**（不依赖 PaddlePaddle / torch / transformers 运行时）：

| 环节 | 模型 | 来源 |
|------|------|------|
| 文本检测 | DBNet | PaddleOCR PP-OCRv4 |
| 文本识别 | PP-OCRv4 Rec | PaddleOCR PP-OCRv4 |
| 公式检测 | YOLOv8 MFD | MinerU / PDF-Extract-Kit |
| 公式识别 | UniMERNet-tiny | opendatalab/UniMERNet |

**推荐入口**：`func/algorithm/onnx_page_pipeline.py` 的 `OnnxPagePipeline` —
页面级「公式检测 → 涂白 → 文本检测」解耦流程。实测 DBNet 用文本行的先验去切公式
会结构性失效（公式区域平均只有 43.9% 面积落在行框内），解耦后公式识别
**51.2% → 93.7%**。

- 单模块推理: `func/algorithm/onnx_ocr_engine.py`、`formula_detect_onnx.py`、`formula_recognize_onnx.py`
- 导出/验证脚本: `onnx_pipeline/`（见 [onnx_pipeline/README.md](onnx_pipeline/README.md)）
- 详细报告:
  - [ACCURACY_VALIDATION_REPORT.md](ACCURACY_VALIDATION_REPORT.md) — **真实文档精度实测 + 页面级解耦**
  - [PADDLEOCR_TO_ONNX_REPORT.md](PADDLEOCR_TO_ONNX_REPORT.md) — 文本检测/识别转换
  - [FORMULA_DETECT_MODERN_REPORT.md](FORMULA_DETECT_MODERN_REPORT.md) — 公式检测方案调研
  - [UNIMERNET_ONNX_REPORT.md](UNIMERNET_ONNX_REPORT.md) — 公式识别部署

> 模型权重（`weights/`，约 1.1 GB）与导出用隔离环境（`venv_legacy_tf/`）不入库，
> 用 `onnx_pipeline/export/` 下的脚本重新生成。

## future

1. 增加用户账号系统及收费模式
2. 增加后台log
3. 增加输入数据校验

## 修改

### 修改 8.06

1. 增加“照片”文本行弯曲图像修复
2. 增加Photo/PDF文本整合为题目功能，提供相应接口
2. 修复mathpix结果格式bug

### 重构 7.29

1. 去除弃用代码及模块
2. 算法模块整合，去除不同进程分别调用算法模块的运作方式
3. 缩小try范围，只包含在mathpix调用以及服务请求中
4. 重做数据持久化模块，方便采集mathpix结果
5. 添加config全局配置，包括模型路径，mathpix开关，服务url端口等
6. weight路径整合，提供下载脚本，方便部署
7. 重做web接口

## 如何使用

### 启动服务
python server.py

### 发送请求
import requests  
url = 'http://<服务器ip>:<服务端口>/photo_ocr_test'  
files = {'image': open('<图片路径>', 'rb')}  
response = requests.post(url, files=files)  

