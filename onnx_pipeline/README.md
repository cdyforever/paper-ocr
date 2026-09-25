# ONNX Pipeline

纯 ONNX OCR 全链路的**模型导出**与**验证**脚本。

推理代码在 `func/algorithm/` 下，本目录只负责「把模型变成 ONNX」和「验证结果对不对」。

---

## 目录结构

```
onnx_pipeline/
├── export/                                  # 模型导出链路
│   ├── 1_text_ocr_paddleocr_to_onnx.py          # PaddleOCR PP-OCRv4 -> ONNX
│   ├── 2_formula_detect_download.py             # 下载 MinerU YOLOv8 MFD 权重
│   ├── 2_formula_detect_export_onnx.py          # 导出 MFD ONNX
│   ├── 2_formula_detect_export_fp16.py          # MFD FP16
│   ├── 3_formula_recognize_download.py          # 下载 UniMERNet 配置
│   ├── 3_formula_recognize_download_weights.py  # 下载 UniMERNet 权重
│   ├── 3_formula_recognize_make_minimal.py      # 提取最小代码副本
│   ├── 3_formula_recognize_export_onnx.py       # 导出 encoder/decoder ONNX
│   └── 3_formula_recognize_quantize_fp16.py     # UniMERNet FP16
└── verify/                                  # 验证脚本
    ├── verify_text_ocr.py                       # 文本检测 + 识别
    ├── verify_formula_detect.py                 # 公式检测 (对比 ultralytics)
    ├── verify_formula_recognize.py              # 公式识别 (对比 PyTorch)
    ├── verify_no_framework_deps.py              # 全链路无框架依赖
    ├── verify_real_accuracy.py                  # 真实文档精度量化
    ├── verify_semantic_eval.py                  # 语义级评估 + 渲染对比
    ├── verify_e2e_split.py                      # ★ 端到端: 切分修复前后对比
    ├── verify_box_splitting.py                  # 检测框切分实验
    ├── verify_clean_vs_photo.py                 # 干净渲染 vs 真实照片对比
    ├── verify_page_decoupling.py                # ★ 页面级解耦: 公式/文本流程对比
    └── dump_intermediates.py                    # ★ 中间结果可视化
```

> 所有脚本已注入项目根定位 shim，**可从任意目录运行**。

---

## 中间结果可视化

```bash
python onnx_pipeline/verify/dump_intermediates.py [image_path]
```

默认用 `test.jpg`，产物输出到 `weights/onnx_ocr/intermediates/`：

| 文件 | 内容 |
|------|------|
| `01_det_input.jpg` | DBNet 缩放后输入 |
| `02_det_prob_map.jpg` | **DBNet 概率图**（JET 热力图 + 统计） |
| `03_det_bitmap.jpg` | 二值化 bitmap |
| `04_det_boxes_raw.jpg` | 膨胀前原始四边形框 |
| `05_det_boxes_final.jpg` | 排序后最终文本框 |
| `06_rec_crops/` | 每个文本框的裁剪图 |
| `10_mfd_input.jpg` | MFD letterbox 后输入 |
| `11_mfd_boxes_raw.jpg` | NMS 后未去重框 |
| `12_mfd_after_dedup.jpg` | 跨类去重后 |
| `13_mfd_final.jpg` | **切分修复后最终框** |
| `14_mfd_split_detail.jpg` | **被切分框的 原框 vs 子框 对比** |
| `20_formula_crops/` | 每个公式区域裁剪图 |
| `intermediates_report.json` | 全部中间量（概率统计、切分记录、坐标） |

---

## 验证

```bash
# 基础功能
python onnx_pipeline/verify/verify_text_ocr.py            # 文本检测 + 识别
python onnx_pipeline/verify/verify_formula_detect.py      # 公式检测 (匹配率 100%)
python onnx_pipeline/verify/verify_formula_recognize.py   # 公式识别 (一致率 100%)
python onnx_pipeline/verify/verify_no_framework_deps.py   # 无 PaddlePaddle/torch 依赖

# 真实文档精度 (用 test.jpg 教材习题页)
python onnx_pipeline/verify/verify_real_accuracy.py       # 文本 93.6% / 公式 89.8%
python onnx_pipeline/verify/verify_e2e_split.py           # 切分修复前后对比
python onnx_pipeline/verify/verify_semantic_eval.py       # 语义级 + 渲染对比
python onnx_pipeline/verify/verify_box_splitting.py       # 检测框切分实验
python onnx_pipeline/verify/verify_clean_vs_photo.py      # 干净 vs 照片对比
```

验证结果详见 [ACCURACY_VALIDATION_REPORT.md](../ACCURACY_VALIDATION_REPORT.md)。

---

## 已修复: 公式检测跨行合并

**问题**：YOLOv8 MFD 在真实文档上会把多行公式合并成一个高框
（实测出现过 757px 高、含 4 道题的框），导致 UniMERNet 输出垃圾。

**修复**（已实现在 `func/algorithm/formula_detect_onnx.py`）：

1. **跨类去重** `_dedup_cross_class()` — 同位置 IoU ≥ 0.9 的
   `embedding`/`isolated` 框只保留高置信者
2. **过高框切分** `_split_tall_boxes()` — 框高 > 中位高 × 2.5 时触发
3. **连通域 + 行聚类** `_split_box_by_components()` — 保留分式完整性
4. **分子分母合并** `_merge_fraction_groups()` — 把被分数线拆开的
   分子/分母重新合并（判定：紧邻 + 一方明显更窄 + x 有重叠）

**实测效果**（757px 合并框 → 4 个正确子框）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 平均最佳相似度 | 36.8% | **62.4%** |
| 达标数 (≥80%) | 1 | **4** |
| 4 道题正确识别 | 0/4 | **3/4** |

可配置参数：

```python
det = FormulaDetectorONNX(
    split_tall_boxes=True,   # 开关切分
    tall_box_ratio=2.5,      # 框高 > 中位高 * 该值 触发切分
    dedup_iou=0.9,           # 跨类去重 IoU 阈值
)
```

---

## 已解决: DBNet 对公式区域的结构性失效

### 现象

在公式密集的页面上, DBNet 经常**只框住公式的一小部分, 或者干脆截断**。

### 根因（实测，非推测）

DBNet 是为「横向文本行」训练的, 分式/根式/上下标/积分号**不是它的目标形态**。
用 3000×4000 教材页 (30 个公式区域) 量化:

| 指标 | 数值 |
|------|------|
| 公式区域被 DBNet 行框覆盖的**平均宽度** | **60.3%** |
| 公式区域被 DBNet 行框覆盖的**平均高度** | **58.2%** |
| 完整覆盖 (宽高均 ≥90%) | **10/30 (33%)** |
| 公式面积落在行框内的**平均可用率** | **44.8%** |
| 可用率 < 50% 的区域 | **16/30 (53%)** |

失效归因: 无 DBNet 框 7/30、框太小 7/30、碎片未覆盖 6/30、正常 10/30。

> 注意: 这不是「模型太弱」。把公式单独裁剪出来喂给 DBNet, **17/20 能检出** ——
> 说明 DBNet 认识公式的笔画, 但它按「文本行」的先验去分割, 于是把公式切碎。

### 关键结论: 公式检测/识别**不依赖** DBNet

`formula_detect_onnx.py` 与 `formula_recognize_onnx.py` 只吃**原图**,
代码中没有任何 `dbnet` / `det_boxes` / `OnnxOCREngine` 引用 —— 两者架构上完全独立。

原来的耦合来自流程顺序:

```
旧: DBNet 切行  ->  在每一行内部做公式检测  ->  行内公式识别
```

行内 MFD 因此受 DBNet 拖累: 页面上 30 个公式, 行内 MFD **只找到 6 个 (73% 丢失)**。

### 解决方案: 页面级解耦流水线

把顺序**反过来** —— 公式检测先行, 再在「涂掉公式的图」上做文本检测:

```
新: 页面级 MFD 检测公式  ->  公式区域涂白  ->  DBNet 只切纯文本行
                                    |
                                    +-> 公式区域整体送 UniMERNet
```

实现在 **`func/algorithm/onnx_page_pipeline.py`** (`OnnxPagePipeline`)。

### 实测对比

| 流程 | 公式区域宽度覆盖 | 高度覆盖 | 完整 | 公式识别 (10 题平均) | ≥80% |
|------|----------------|---------|------|-------------------|------|
| A 旧流程 (DBNet 切行) | 60.3% | 58.2% | 10/30 | 51.2% | 2/10 |
| B 仅放大 DBNet 输入 (960→1920) | 74.2% | 66.6% | 10/30 | — | — |
| **C 新流程 (页面级解耦)** | **100%** | **100%** | **30/30** | **93.7%** | **9/10** |

逐题（旧 → 新）:

```
y = x3 + 7/x4 - 2/x + 12        26.9%  ->  100.0%
y = x2 ln x                     66.7%  ->  100.0%
y = ln x / x                    25.0%  ->   91.7%
y = 5x3 - 2^x + 3e^x            80.0%  ->  100.0%
y = sin x cos x                 69.2%  ->   92.3%
y = 3ex cos x                   27.3%  ->   90.9%
y = ex/x2 + ln3                 33.3%  ->   94.4%
y = x2 ln x cos x               64.3%  ->   92.9%
s = (1+sin t)/(1+cos t)         38.1%  ->  100.0%
y = 2tan x + sec x - 1          81.2%  ->   75.0%   <- 识别问题(字距), 非检测
```

顺带的好处: DBNet 不再被公式碎片干扰, 涂白后**没有任何文本行残留在公式区域内**
(>30% 压在公式上的行: 0)。

### 用法

```python
from func.algorithm.onnx_page_pipeline import OnnxPagePipeline

pipe = OnnxPagePipeline(mask_mode='mask')       # 'mask' | 'filter' | 'none'
items, debug = pipe('test.jpg', return_debug=True)

for it in items:
    if it['type'] == 'text':
        print(it['text'])
    else:
        print(it['formula_type'], it['latex'])   # embedding / isolated

print(pipe.to_markdown(items))                   # 拼成 Markdown ($...$ / $$...$$)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `mask_mode` | `'mask'` | `'mask'` 涂白后再跑 DBNet（推荐）；`'filter'` DBNet 照常跑、事后丢弃公式内的行；`'none'` 对照组 |
| `mask_margin` | 6 | 涂白外扩像素，防止笔画残留 |
| `drop_line_formula_ratio` | 0.5 | `'filter'` 模式下丢弃阈值 |

> `debug` 里带 `det_boxes_original`（原图上的 DBNet 框）、`text_lines`、
> `dropped_text_lines`、`formula_items`，便于逐环节核对。

### 附带修复: 切分子框与独立检测框重复

`_split_tall_boxes()` 切分「合并框」时, 切出的某个子框可能**已经被单独检出**
（实测: 757px 合并框切出 4 份, 其中 1 份与另一条 score 更高的独立框 IoU=0.85），
导致同一公式被识别两次。

现增加 `split_child_iou` 参数（默认 0.6）: 子框与任何独立检测框的 IoU
达到该值即丢弃, 并在 `debug['split_records']` 的 `dropped_children` 中留痕。

```python
det = FormulaDetectorONNX(split_child_iou=0.6)
```

实测效果: 页面级近重复框 **1 对 → 0 对**。

### 性能

页面级解耦在 3000×4000 教材页上的耗时构成 (30 个公式区域):

| 阶段 | 耗时 |
|------|------|
| MFD 公式检测 | 5.2 s |
| DBNet + PP-OCRv4 (涂白图) | 6.4 s |
| UniMERNet 公式识别 (30 个) | **96.4 s** |
| 合计 | 109.1 s |

瓶颈是公式识别的自回归解码（每步重算全部历史 KV）。
如需提速见下方「可选优化」。

---

## 验证

```bash
# 基础功能
python onnx_pipeline/verify/verify_text_ocr.py            # 文本检测 + 识别
python onnx_pipeline/verify/verify_formula_detect.py      # 公式检测 (匹配率 100%)
python onnx_pipeline/verify/verify_formula_recognize.py   # 公式识别 (一致率 100%)
python onnx_pipeline/verify/verify_no_framework_deps.py   # 无 PaddlePaddle/torch 依赖

# 真实文档精度 (用 test.jpg 教材习题页)
python onnx_pipeline/verify/verify_real_accuracy.py       # 文本 93.6% / 公式 89.8%
python onnx_pipeline/verify/verify_e2e_split.py           # 切分修复前后对比
python onnx_pipeline/verify/verify_page_decoupling.py     # ★ 页面级解耦 A/B/C 对比
python onnx_pipeline/verify/verify_semantic_eval.py       # 语义级 + 渲染对比
python onnx_pipeline/verify/verify_box_splitting.py       # 检测框切分实验
python onnx_pipeline/verify/verify_clean_vs_photo.py      # 干净 vs 照片对比
```

验证结果详见 [ACCURACY_VALIDATION_REPORT.md](../ACCURACY_VALIDATION_REPORT.md)。

### 诊断脚本

分析 DBNet 在公式区域的行为（用于支撑上面的结论）:

| 脚本 | 用途 |
|------|------|
| `verify/analyze_dbnet_on_formula.py` | DBNet 框 vs MFD 框的覆盖度对比 |
| `verify/analyze_dbnet_rootcause.py` | 失效归因（无框 / 框太小 / 碎片未覆盖） |
| `verify/analyze_dbnet_fix.py` | 四种解决方案的量化对比 |
| `verify/verify_pipeline_coupling.py` | 行内 MFD 的丢失率 |
| `verify/verify_coupling_precise.py` | 行内 MFD 三分类（完整/部分/完全丢失） |
| `verify/verify_info_availability.py` | 公式面积落在行框内的比例（排除 resize 干扰） |

---

## 重新导出

### 1. 文本检测 + 识别

```bash
# 需要先安装: pip install paddlepaddle==2.6.2 paddleocr==2.9.0 paddle2onnx==1.3.1
python onnx_pipeline/export/1_text_ocr_paddleocr_to_onnx.py
```

> 关键: `paddle2onnx` 必须用 **1.3.1**（2.x 要求 paddle 3.0+，与 2.6.2 冲突）。

### 2. 公式检测

```bash
python onnx_pipeline/export/2_formula_detect_download.py       # 下载 yolo_v8_ft.pt (334MB)
python onnx_pipeline/export/2_formula_detect_export_onnx.py    # 导出 FP32 ONNX
python onnx_pipeline/export/2_formula_detect_export_fp16.py    # 导出 FP16 (83.5MB)
```

> 需要 `ultralytics`。

### 3. 公式识别

```bash
python onnx_pipeline/export/3_formula_recognize_download.py          # 配置 + tokenizer
python onnx_pipeline/export/3_formula_recognize_download_weights.py  # 权重 (430MB)
python onnx_pipeline/export/3_formula_recognize_make_minimal.py      # 提取最小代码
python onnx_pipeline/export/3_formula_recognize_export_onnx.py       # 导出 ONNX
python onnx_pipeline/export/3_formula_recognize_quantize_fp16.py     # FP16
```

> **重要**: UniMERNet 官方代码为 `transformers 4.36` 编写，
> 需用隔离环境 `venv_legacy_tf/`（已随项目保留）。
> 导出脚本已自动把该目录加入 `sys.path`。

---

## 模型产物

| 模型 | 路径 | 体积 |
|------|------|------|
| 文本检测 DBNet | `weights/text_line_detect/dbnet_det.onnx` | 4.5 MB |
| 文本识别 PP-OCRv4 | `weights/text_recognition/crnn_rec.onnx` | 10.3 MB |
| 方向分类 | `weights/text_recognition/cls.onnx` | 558 KB |
| 字符字典 | `weights/text_recognition/ppocr_keys_v1.txt` | 26 KB |
| **公式检测 MFD** | `weights/formula_detect/mfd_yolov8_fp16.onnx` | **83.5 MB** |
| **公式识别 encoder** | `weights/formula_recognize/onnx_fp16/unimernet_encoder.onnx` | **54.4 MB** |
| **公式识别 decoder** | `weights/formula_recognize/onnx_fp16/unimernet_decoder.onnx` | **156.5 MB** |

原始 PyTorch 权重（重新导出时用）：
- `weights/formula_detect/yolo_v8_ft.pt` (334 MB)
- `weights/formula_recognize/unimernet_tiny/unimernet_tiny_pdfkit.pth` (430 MB)

---

## 推理接口

```python
# 文本检测 + 识别
from func.algorithm.onnx_ocr_engine import OnnxOCREngine
engine = OnnxOCREngine()
for r in engine('image.jpg'):
    print(r['text'], r['score'])

# ★ 推荐: 页面级解耦全流程 (文本 + 公式)
from func.algorithm.onnx_page_pipeline import OnnxPagePipeline
pipe = OnnxPagePipeline()
items, debug = pipe('page.jpg', return_debug=True)
print(pipe.to_markdown(items))

# 公式检测
from func.algorithm.formula_detect_onnx import FormulaDetectorONNX
det = FormulaDetectorONNX()
for f in det(image):
    print(f['type'], f['box'])      # embedding / isolated

# 公式识别
from func.algorithm.formula_recognize_onnx import FormulaRecognizerONNX
rec = FormulaRecognizerONNX()
latex = rec(formula_crop)
```

---

## 依赖

推理时**只需要**：
```
onnxruntime
opencv-python
numpy
pyclipper   # DBNet 后处理
shapely     # DBNet 后处理
tokenizers  # UniMERNet BPE 解码
```

**不需要** PaddlePaddle / torch / transformers。
