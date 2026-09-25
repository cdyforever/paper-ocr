# 公式检测方案调研与实施报告

## 一、结论先行

原启发式公式检测算法（2021 年方案）**已被现代深度学习方案替代**，并已跑通纯 ONNX 实现。

| 项目 | 原方案 | 新方案 |
|------|--------|--------|
| 算法 | 轮廓分析 + 高度/位置启发式 | **YOLOv8 目标检测** |
| 来源 | 自研（2021） | **MinerU / PDF-Extract-Kit** |
| 类别 | 仅"公式 / 文本" | **行内公式 + 独立公式**（更细） |
| 模型体积 | `latex-detect-v2.onnx`（已丢失） | 83.5 MB (FP16) |
| 依赖 | 需 OpenCV 3.x（有 bug） | **纯 ONNX Runtime** |
| 状态 | ❌ 模型缺失 | ✅ **已验证可用** |

---

## 二、调研：主流开源方案

### 方案对比

| 方案 | 检测模型 | 识别模型 | 体积 | 许可 | 评价 |
|------|---------|---------|------|------|------|
| **MinerU / PDF-Extract-Kit** | YOLOv8 (2类) | UniMERNet / PP-FormulaNet | 334MB / 1.2GB | AGPL | ⭐ **采用** |
| **CnSTD** (breezedeus) | YOLOv7-tiny MFD | — | 11.6MB | 会员制 | MFD 模型需付费 |
| **MinerU2.5-2509-1.2B** | 端到端 VLM | 同一模型 | 2.2GB | AGPL | 太重 |
| **pix2tex** (LaTeX-OCR) | 无（仅识别） | ResNet+Transformer | 116MB | MIT | 已装，质量一般 |
| **PaddleOCR PP-FormulaNet** | — | PP-FormulaNet_plus-M | 589MB | Apache | 识别备选 |

### 关键发现

1. **MinerU 的 MFD 模型类别设计精准**：
   ```
   {0: 'embedding',   # 行内公式  f(x) = x² + 2x - 3
    1: 'isolated'}    # 独立公式  x = (-b ± √(b²-4ac)) / 2a
   ```
   正好覆盖试卷/论文的核心需求。

2. **CnSTD 的 MFD 模型需付费会员**，但普通检测模型是公开 ONNX。

3. **原始权重服务器已下线**（`http://123.60.217.149/weights.zip` → 502），这是原 PDF 公式检测模型缺失的根因。

---

## 三、实施：纯 ONNX 公式检测

### 3.1 模型获取

```
来源: HuggingFace opendatalab/PDF-Extract-Kit-1.0
文件: models/MFD/YOLO/yolo_v8_ft.pt  (333.7 MB)
架构: YOLOv8l, nc=2, 43.6M 参数, 421.9 GFLOPs
```

### 3.2 ONNX 导出

```python
from ultralytics import YOLO
model = YOLO('yolo_v8_ft.pt')
model.export(format='onnx', opset=12, simplify=True, imgsz=1024)
model.export(format='onnx', opset=12, simplify=True, imgsz=1024, half=True)  # FP16
```

产物：

| 文件 | 体积 |
|------|------|
| `weights/formula_detect/mfd_yolov8.onnx` | 166.9 MB |
| `weights/formula_detect/mfd_yolov8_fp16.onnx` | **83.5 MB** |

模型结构：
```
Input : images  [1, 3, 1024, 1024]  float32
Output: output0 [1, 6, 21504]       (4 bbox + 2 class)
```

### 3.3 纯 ONNX 检测器

已实现 `func/algorithm/formula_detect_onnx.py`，**不依赖 ultralytics / torch / PaddlePaddle**：

```python
from func.algorithm.formula_detect_onnx import FormulaDetectorONNX

det = FormulaDetectorONNX()          # 默认加载 FP16 模型
results = det(image)
# [{'box': [x1,y1,x2,y2], 'type': 'embedding'|'isolated', 'score': 0.92}, ...]

vis = det.draw(image, results)
```

内部实现要点：
- **Letterbox 预处理**：等比缩放 + 灰边填充到 1024×1024
- **YOLOv8 后处理**：`[1, 6, N]` → 转置 → 按类别分别 NMS
- **坐标还原**：去除 padding、除以缩放比、裁剪到原图
- **阅读顺序排序**：先上后下、再左到右
- **中文可视化**：PIL 渲染标签（`cv2.putText` 不支持中文）

---

## 四、验证结果

### 4.1 检测效果

测试图（含行内公式 + 独立公式）：

![公式检测结果](weights/onnx_ocr/formula_detect_pure_onnx.jpg)

检测结果：
```
embedding 0.917  [177, 112, 387, 144]   f(x) = x² + 2x - 3
embedding 0.908  [360, 161, 562, 194]   (a+b)/2 ≥ √(ab)
embedding 0.895  [205, 162, 278, 193]   a > 0
embedding 0.879  [99,  162, 170, 193]   b > 0
isolated  0.799  [323, 235, 653, 270]   x = (-b ± √(b²-4ac)) / 2a
isolated  0.651  [296, 310, 493, 347]   ∫₀¹ x² dx = 1/3
```

✅ **准确识别行内公式和独立公式**  
✅ **未误检中文正文**（第 3、4 题纯文字未被框选）

### 4.2 与 ultralytics 一致性

| 指标 | 结果 |
|------|------|
| 检测数量 | 6 vs 6 |
| **匹配率** | **6/6 (100%)** |
| **IoU** | **全部 > 0.95** |
| 速度 | ONNX 2.03s vs ultralytics 1.37s |

### 4.3 依赖验证

```
[OK] torch 已拦截
[OK] ultralytics 已拦截
已加载的 torch/paddle 模块: 无
[SUCCESS] 纯 ONNX 公式检测器运行成功，无 torch/ultralytics/paddle 依赖
```

---

## 五、公式识别（MFR）现状

检测已解决，**识别仍是缺口**。可用方案：

| 方案 | 体积 | 状态 | 质量 |
|------|------|------|------|
| **pix2tex** | 116 MB | ✅ 已安装可用 | 一般 |
| **UniMERNet-tiny** | 410 MB | ⬇️ 可下载 | 好 |
| **PP-FormulaNet_plus-M** | 589 MB | ⬇️ 可下载 | 好 |
| **MinerU2.5 VLM** | 2.2 GB | ⬇️ 可下载 | 最好 |

### pix2tex 实测

| 输入 | 输出 | 评价 |
|------|------|------|
| `f(x) = x² + 2x - 3` | `\,\mathbf{f}(\mathbf{x})=\mathbf{x}^{2}+2\mathbf{x}-3` | ✅ 正确 |
| `(a+b)/2 ≥ √(ab)` | `(\mathbf{a}+\mathbf{b})/2\,\geq\,\bigvee(\mathbf{ab})` | ⚠️ `√`→`\bigvee` 错 |
| `x = (-b ± √(b² - 4ac)) / 2a` | `\chi=(-\mathbf{b}\pm\sqrt{(\mathbf{b}^{2}\cdot4\mathbf{a}c)})\left/2\mathbf{a}\right.` | ⚠️ `x`→`\chi` 错 |
| `∫₀¹ x² dx = 1/3` | `\textstyle\bigcap\Pi^{1}\times^{2}\operatorname{d}\!\chi=\ 1/3` | ❌ 积分符号识别错 |

**结论**：pix2tex 对简单公式可用，复杂公式（积分、根号）错误较多。建议换 UniMERNet。

---

## 六、产出文件

### 核心模块

| 文件 | 说明 |
|------|------|
| `func/algorithm/formula_detect_onnx.py` | **纯 ONNX 公式检测器** |

### 模型文件

```
weights/formula_detect/
├── mfd_yolov8_fp16.onnx    83.5 MB   ← 推荐使用
├── mfd_yolov8.onnx        166.9 MB   ← FP32
└── yolo_v8_ft.pt          333.7 MB   ← 原始 PyTorch
```

### 工具脚本

| 文件 | 说明 |
|------|------|
| `research_formula_solutions.py` | 开源方案调研 |
| `research_mineru_detail.py` | MinerU/CnSTD 深入调研 |
| `check_formula_models_availability.py` | 模型可用性与体积核查 |
| `download_mfd_models.py` | MFD 模型下载 |
| `test_mfd_models.py` | 模型加载测试 |
| `test_and_export_mfd.py` | 测试 + ONNX 导出 |
| `export_mfd_fp16.py` | FP16 导出与体积优化 |
| `test_formula_detect_onnx.py` | 纯 ONNX 验证（含一致性对比） |
| `test_formula_recognition.py` | 公式识别测试 |

---

## 七、下一步建议

### 优先级 1：接入公式识别（补齐闭环）

下载 **UniMERNet-tiny**（410 MB）替代 pix2tex，显著提升复杂公式识别率：

```python
# HuggingFace: wanderkid/unimernet_tiny
# 或 PDF-Extract-Kit: models/MFR/unimernet_tiny/pytorch_model.pth
```

### 优先级 2：集成到 PaperOCR

将 `FormulaDetectorONNX` 接入 `TextRecognizer`，替换旧的 `detectFormula_PDF` / `detectFormula_Photo`：

```python
# 旧: 启发式 + 缺失模型
segments_pos = detectFormula_PDF(ench_formula, model, median_height)

# 新: 纯 ONNX YOLOv8
formulas = self.formula_detector(ench_formula)
```

### 优先级 3：模型体积优化

83.5 MB 的 FP16 模型可进一步 INT8 量化到约 25 MB（精度损失约 1-2%）。

---

## 八、总结

| 问题 | 答案 |
|------|------|
| 原公式检测算法是否适合现在？ | **不适合**。2021 年启发式方案，且 PDF 路径模型已丢失 |
| 找到更好的开源方案了吗？ | **是**。MinerU / PDF-Extract-Kit 的 YOLOv8 MFD |
| 能用纯 ONNX 跑吗？ | **能**。已验证，无 torch/ultralytics/paddle 依赖 |
| 效果如何？ | **很好**。6/6 匹配，IoU > 0.95，能区分行内/独立公式 |
| 还缺什么？ | **公式识别（MFR）**。pix2tex 可用但质量一般，建议换 UniMERNet |
