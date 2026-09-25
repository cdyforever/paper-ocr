# 精度验证报告 — 真实文档实测

## 结论

**能，但精度高度依赖前置环节。** 分环节实测结果（真实教材照片 3000×4000，非合成图）：

| 环节 | 平均精度 | 达标率 | 评价 |
|------|---------|--------|------|
| **文本识别** (PP-OCRv4) | **93.6%** | 5/6 完全正确 | ✅ 优秀 |
| **公式识别** (UniMERNet-tiny) | **89.8%** | 8/9 ≥80% | ✅ 良好 |
| **公式检测** (YOLOv8 MFD) | 有缺陷 | — | ✅ 已修复 |
| **流程编排** (DBNet 与公式的耦合) | 结构性失效 | — | ✅ 已解耦 |

> **最重要的一条**：DBNet 在公式区域上是**结构性失效**的（公式区域平均只有
> 44.8% 的面积落在它的行框内），所以**公式检测与识别必须与 DBNet 解耦**。
> 解耦后公式识别从 **51.2% → 93.7%**。详见 [第八节](#八dbnet-对公式区域的结构性失效--页面级解耦修复)。

---

## 一、文本识别：93.6%

测试对象：教材习题页中的纯文本行（不含数学公式）。

| GT 文本 | 识别结果 | 准确率 |
|---------|---------|--------|
| 习题2-2 | 习题2-2 | 100% |
| 1. 推导余切函数及余割函数的导数公式： | 1.推导余切函数及余割函数的导数公式： | 100% |
| 2. 求下列函数的导数： | 2.求下列函数的导数： | 100% |
| 3. 求下列函数在给定点处的导数： | 3.求下列函数在给定点处的导数： | 100% |
| (1) 该物体的速度 v(t); | （1）该物体的速度v（t）； | 61.5% |
| 6. 求下列函数的导数： | 6.求下列函数的导数： | 100% |

**完全正确 5/6**。唯一的 61.5% 是标点全半角差异（`()` vs `（）`、`;` vs `；`），**语义无损**。

> 结论：**中文/英文文本识别质量优秀，可直接用于生产。**

---

## 二、公式识别：89.8%

用**精确裁剪**的公式区域（绕过检测环节）测试 UniMERNet-tiny 本身的能力：

| GT 公式 | UniMERNet 输出 | 相似度 |
|---------|---------------|--------|
| y = x³ + 7/x⁴ - 2/x + 12 | `y = x^{3} + \frac{7}{x^{4}} - \frac{2}{x} + 12` | **100%** |
| y = 5x³ - 2ˣ + 3eˣ | `y = 5x^{3} - 2^{x} + 3e^{x}` | **100%** |
| s = (1+sin t)/(1+cos t) | `s = \frac{1+\sin t}{1+\cos t}` | **100%** |
| y = x²ln x | `y = x^{2}\ln x` | **100%** |
| y = eˣ/x² + ln 3 | `y = \frac{e^{x}}{x^{2}} + \ln 3` | 94.4% |
| y = x²ln xcos x | `y = x^{2}\ln x\cos x` | 92.9% |
| y = sin x · cos x | `y = \sin x \cdot \cos` | 92.3% |
| y = 3eˣcos x | `y = 3e^{x}\cos x` | 90.9% |
| y = 2tan x + sec x - 1 | `y = 2\ln 1 1 x + 8\rho(x-1)` | 37.5% |

**4/9 完全正确，8/9 达标（≥80%）**。

### 渲染对比验证

用 matplotlib 渲染 GT 与预测结果做视觉核对：

- `y = x³ + 7/x⁴ - 2/x + 12` → 渲染完全一致 ✅
- `y = eˣ/x² + ln 3` → 渲染完全一致 ✅（`\mathbf{e}` 与 `e` 语义相同）
- `y = (1+sin t)/(1+cos t)` → 渲染完全一致 ✅

> 结论：**公式识别模型本身质量良好**，剩余 1/9 失败是 `sec` 被误识为 `\ln 1 1`（字距过大导致）。

---

## 三、公式检测：缺陷已修复 ✅

MFD 在真实照片上检出 30 个区域，原先有 **2 类问题**，现已全部修复。

### 问题 1：跨行合并（最严重）

一个 757px 高的框把 **4 道不同的题**合并成一个区域：

```
检测框 [340, 1008, 1077, 1765]  高=757px
  ├─ 2(1) y = x³ + 7/x⁴ - 2/x + 12
  ├─ 2(3) y = 2tan x + sec x - 1
  ├─ 2(5) y = x²ln x
  └─ 2(7) y = ln x / x
```

送进 UniMERNet 后输出 226 字符的 `\begin{array}...` 垃圾。

### 问题 2：跨类重复

3 对框 IoU > 0.98（同一位置同时被标为 `embedding` 和 `isolated`）。

---

### 修复实现

已在 `func/algorithm/formula_detect_onnx.py` 实现 4 步后处理：

| 步骤 | 方法 | 作用 |
|------|------|------|
| 1 | `_dedup_cross_class()` | 跨类去重：同位置 IoU ≥ 0.9 保留高置信 |
| 2 | `_split_tall_boxes()` | 框高 > 中位高 × 2.5 触发切分 |
| 3 | `_split_box_by_components()` | 连通域 + 行聚类，保留分式完整性 |
| 4 | `_merge_fraction_groups()` | 合并被分数线拆开的分子/分母 |

**第 4 步是关键的迭代**：仅做行聚类会把 `y = ln x / x` 从分数线处
切成「分子」和「分母」两段。判定条件为：

```
紧邻 (间距 < 1.8 × 中位字高)
且 一方宽度 < 另一方 0.8 倍   (分子/分母比整行窄)
且 x 范围有重叠
```

### 修复效果实测

**切分质量**（757px 合并框 → 4 个正确子框）：

```
#1.1 h=210px  y = x³ + 7/x⁴ - 2/x + 12      ← 分式完整
#1.2 h=132px  y = 2tan x + sec x - 1
#1.3 h=116px  y = x²ln x
#1.4 h=142px  y = ln x / x                  ← 分子分母已合并
```

**识别效果**：

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| 平均最佳相似度 | 36.8% | **62.4%** | +25.6pp |
| 达标数 (≥80%) | 1 | **4** | +3 |
| 4 道题正确识别 | 0/4 | **3/4** | +3 |

逐题核对（修复后）：

```
OK  [2(1)] y = x3 + 7/x4 - 2/x + 12   100.0%
OK  [2(5)] y = x2 ln x                100.0%
OK  [2(7)] y = ln x / x                91.7%
BAD [2(3)] y = 2tan x + sec x - 1      75.0%   ← sec 被误识为 8∈
```

唯一未达标的 `2(3)` 是识别问题（`sec` 字距过大被读成 `8∈`），
非检测问题。

### 配置接口

```python
from func.algorithm.formula_detect_onnx import FormulaDetectorONNX

det = FormulaDetectorONNX(
    split_tall_boxes=True,   # 开关切分修复
    tall_box_ratio=2.5,      # 框高 > 中位高 * 该值 触发切分
    dedup_iou=0.9,           # 跨类去重 IoU 阈值
)

# 调试模式: 拿到全部中间量
results, debug = det(img, return_debug=True)
# debug['raw_boxes']       NMS 后未去重
# debug['after_dedup']     跨类去重后
# debug['split_records']   切分记录 (原框 + 子框)
# debug['final_boxes']     最终结果
```

---

## 四、中间结果可视化

新增 `dump_intermediates.py`，保存完整推理链路的所有中间产物：

```bash
python onnx_pipeline/verify/dump_intermediates.py test.jpg
```

输出到 `weights/onnx_ocr/intermediates/`：

| 文件 | 内容 |
|------|------|
| `01_det_input.jpg` | DBNet 缩放后输入 (704×960) |
| `02_det_prob_map.jpg` | **DBNet 概率图** JET 热力图 |
| `03_det_bitmap.jpg` | 二值化 bitmap (前景 7.90%) |
| `04_det_boxes_raw.jpg` | 膨胀前原始四边形框 (30) |
| `05_det_boxes_final.jpg` | 排序后最终文本框 (28) |
| `06_rec_crops/` | 28 张文本行裁剪图 |
| `10_mfd_input.jpg` | MFD letterbox 输入 (1024×1024) |
| `11_mfd_boxes_raw.jpg` | NMS 后 (31) |
| `12_mfd_after_dedup.jpg` | 跨类去重后 (28) |
| `13_mfd_final.jpg` | 切分修复后 (30) |
| `14_mfd_split_detail.jpg` | **原框 vs 子框对比** |
| `20_formula_crops/` | 30 张公式区域裁剪图 |
| `intermediates_report.json` | 全部中间量 |

### DBNet 概率图

![DBNet 概率图](weights/onnx_ocr/intermediates/02_det_prob_map.jpg)

文本行定位清晰，概率值 `min=0.000 max=1.000 mean=0.078`，
二值化后前景占 7.90%。

### 切分修复对比

![切分对比](weights/onnx_ocr/intermediates/14_mfd_split_detail.jpg)

上图：757px 原框含 4 道题。下图：切分后 4 个子框，分式完整保留。

---

## 五、DBNet 对公式区域的结构性失效 → 页面级解耦修复

### 5.1 问题

DBNet 在公式区域**经常只框住一小部分，或者完全截断**。

### 5.2 根因（实测，非推测）

DBNet 是为「横向文本行」训练的，分式 / 根式 / 上下标 / 积分号**不是它的目标形态**。
在 3000×4000 教材页上对 30 个公式区域量化：

| 指标 | 数值 |
|------|------|
| 公式区域被 DBNet 行框覆盖的**平均宽度** | **60.3%** |
| 公式区域被 DBNet 行框覆盖的**平均高度** | **58.2%** |
| 完整覆盖（宽高均 ≥90%） | **10/30 (33%)** |
| 公式面积落在行框内的**平均可用率** | **44.8%** |
| 可用率 < 50% 的区域 | **16/30 (53%)** |

失效归因：

| 原因 | 数量 |
|------|------|
| 完全没有 DBNet 框 | 7/30 (23%) |
| 有框但太小 | 7/30 (23%) |
| 框碎了、覆盖不全 | 6/30 (20%) |
| 正常 | 10/30 (33%) |

> **不是「模型太弱」**：把公式单独裁剪出来喂给 DBNet，**17/20 能检出** ——
> DBNet 认识公式的笔画，但它按「文本行」的先验去分割，于是把公式切碎。

### 5.3 关键结论：公式检测/识别**不依赖** DBNet

`formula_detect_onnx.py` 与 `formula_recognize_onnx.py` 只吃**原图**，
代码中没有任何 `dbnet` / `det_boxes` / `OnnxOCREngine` 引用 —— 两者架构上完全独立。

失效来自**流程顺序**，不是模型依赖：

```
旧: DBNet 切行  ->  在每一行内部做公式检测  ->  行内公式识别
```

行内 MFD 因此被 DBNet 拖累：页面上 30 个公式，行内 MFD **只找到 6 个（73% 丢失）**。

### 5.4 解决方案：页面级解耦

把顺序**反过来** —— 公式检测先行，再在「涂掉公式的图」上做文本检测：

```
新: 页面级 MFD 检测公式  ->  公式区域涂白  ->  DBNet 只切纯文本行
                                    |
                                    +-> 公式区域整体送 UniMERNet
```

实现：`func/algorithm/onnx_page_pipeline.py` 的 `OnnxPagePipeline`。

### 5.5 实测对比

| 流程 | 公式区域宽度覆盖 | 高度覆盖 | 完整 | 公式识别（10 题平均） | ≥80% |
|------|----------------|---------|------|-------------------|------|
| A 旧流程（DBNet 切行） | 60.3% | 58.2% | 10/30 | 51.2% | 2/10 |
| B 仅放大 DBNet 输入（960→1920） | 74.2% | 66.6% | 10/30 | — | — |
| **C 新流程（页面级解耦）** | **100%** | **100%** | **30/30** | **93.7%** | **9/10** |

逐题（旧 → 新）：

| 题目 | 旧流程 | 新流程 |
|------|--------|--------|
| y = x³ + 7/x⁴ - 2/x + 12 | 26.9% | **100.0%** |
| y = x²ln x | 66.7% | **100.0%** |
| y = ln x / x | 25.0% | **91.7%** |
| y = 5x³ - 2ˣ + 3eˣ | 80.0% | **100.0%** |
| y = sin x · cos x | 69.2% | **92.3%** |
| y = 3eˣcos x | 27.3% | **90.9%** |
| y = eˣ/x² + ln 3 | 33.3% | **94.4%** |
| y = x²ln xcos x | 64.3% | **92.9%** |
| s = (1+sin t)/(1+cos t) | 38.1% | **100.0%** |
| y = 2tan x + sec x - 1 | 81.2% | 75.0% ← 识别问题（字距），非检测 |

> 注意「放大输入」这条捷径（B）**不解决问题**：宽度覆盖从 60.3% 涨到 74.2%，
> 但「完整覆盖」仍是 10/30 —— 因为瓶颈是 DBNet 的分割先验，不是分辨率。

**顺带的好处**：DBNet 不再被公式碎片干扰，涂白后**没有任何文本行残留在公式区域内**
（>30% 面积压在公式上的行：0）。

### 5.6 用法

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
| `drop_line_formula_ratio` | 0.5 | `'filter'` 模式下的丢弃阈值 |

三种模式实测（3000×4000 教材页）：

| mask_mode | 文本行 | 丢弃行 | 残留在公式上的行 | 公式 |
|-----------|--------|--------|-----------------|------|
| `mask` | **25** | 0 | **0** | 30 |
| `filter` | 7 | 21 | 1 | 30 |
| `none` | 7 | 21 | 1 | 30 |

> `filter`/`none` 只剩 7 行，是因为 DBNet 把整个公式段落当成了一个大框、
> 然后被整体丢弃 —— 这正是耦合流程「要么切碎、要么整块丢」的写照。

### 5.7 性能

| 阶段 | 耗时 |
|------|------|
| MFD 公式检测 | 5.2 s |
| DBNet + PP-OCRv4（涂白图） | 6.4 s |
| UniMERNet 公式识别（30 个） | **96.4 s** |
| 合计 | 109.1 s |

瓶颈是公式识别的自回归解码（每步重算全部历史 KV），优化见第八节。

### 5.8 可视化

![页面级解耦结果](weights/onnx_ocr/decoupling/C_page_decoupled.jpg)

绿=文本行，橙=行内公式，蓝=独立公式。

![涂白后的纯文本图](weights/onnx_ocr/decoupling/C_text_only_masked.jpg)

公式区域被涂白后，DBNet 只需处理纯文本，行切分干净。

---

## 六、关键发现

### 影响精度因素排序

| 排序 | 因素 | 影响 |
|------|------|------|
| **1** | **流程编排（DBNet 与公式耦合）** | 公式识别 51.2% → 解耦后 93.7% |
| **2** | **公式检测框质量** | 合并 → 识别率 0% |
| **3** | **裁剪边界** | 切掉分式分子 → 33% |
| **4** | 字距/字体 | `sec` → `\ln 1 1` |
| **5** | 图像清晰度 | 影响上标识别 |
| **6** | 模型容量 | tiny 版对超复杂公式有限 |

### 重要认知

**公式识别模型不是瓶颈，公式检测与流程编排才是。**

- 只要给 UniMERNet 正确的裁剪区域，**8/9 达 80% 以上，4/9 完全正确**
- 一旦检测框合并了多行，**识别率跌到 0%**
- 一旦裁剪切掉分式的一部分，**识别率跌到 33%**
- 修复检测后处理，合并区域的识别率从 **36.8% → 62.4%**
- 把公式从 DBNet 的行切分里解放出来，公式识别从 **51.2% → 93.7%**
- DBNet 本身没问题 —— **它的失效只发生在「用文本行的先验去切公式」时**

---

## 七、评估方法说明

### 为什么不用字符级 LaTeX 比对

同一公式的不同 LaTeX 写法语义相同但字符差异巨大：

```
GT  : y=\frac{e^{x}}{x^{2}}+\ln 3
Pred: y={\frac{\mathbf{e}^{x}}{{x}^{2}}}+\mathbf{l}\,\mathbf{n}\ 3
```

字符级相似度：**0%**（误判）  
语义级相似度：**94.4%**（正确）

本报告采用**语义归一化**：剥离 `\mathbf`、`\,`、`{}` 等排版差异后比对，并辅以 matplotlib 渲染视觉核对。

---

## 八、建议

### 已完成

- ✅ **公式检测跨行合并修复**（62.4%，4 道题 3/4 正确）
- ✅ **中间结果可视化**（DBNet 概率图、各级检测框、切分对比）
- ✅ **页面级解耦**（公式识别 51.2% → 93.7%，公式覆盖度 10/30 → 30/30）
- ✅ **切分子框去重**（合并框切出的子框若已被独立检测覆盖则丢弃，重复框 1 → 0）

### 可选优化

- **KV Cache**：公式识别当前每步重算历史，加 cache 可提速 3-5 倍（当前 30 个公式耗时 96 s）
- **beam search**：当前贪婪解码，beam=3 可提升复杂公式准确率
- **换 base 版**：UniMERNet-base（1.2GB）对复杂公式更强
- **字距预处理**：对 `sec` 这类大间距 token 做形态学闭运算，可修复 2(3)

---

## 附：验证脚本

| 脚本 | 用途 |
|------|------|
| `onnx_pipeline/verify/dump_intermediates.py` | **中间结果可视化** |
| `onnx_pipeline/verify/verify_e2e_split.py` | **端到端切分修复对比** |
| `onnx_pipeline/verify/verify_page_decoupling.py` | **★ 页面级解耦 A/B/C 对比** |
| `onnx_pipeline/verify/analyze_dbnet_on_formula.py` | DBNet 框 vs MFD 框覆盖度 |
| `onnx_pipeline/verify/analyze_dbnet_rootcause.py` | DBNet 失效归因 |
| `onnx_pipeline/verify/analyze_dbnet_fix.py` | 四种解决方案量化对比 |
| `onnx_pipeline/verify/verify_pipeline_coupling.py` | 行内 MFD 丢失率 |
| `onnx_pipeline/verify/verify_coupling_precise.py` | 行内 MFD 三分类 |
| `onnx_pipeline/verify/verify_info_availability.py` | 公式面积落在行框内的比例 |
| `onnx_pipeline/verify/verify_real_accuracy.py` | 真实文档精度汇总 |
| `onnx_pipeline/verify/verify_semantic_eval.py` | 语义级评估 + 渲染对比 |
| `onnx_pipeline/verify/verify_box_splitting.py` | 检测框切分实验 |
| `onnx_pipeline/verify/verify_clean_vs_photo.py` | 干净渲染 vs 照片对比 |
| `onnx_pipeline/verify/verify_text_ocr.py` | 文本检测 + 识别 |
| `onnx_pipeline/verify/verify_formula_detect.py` | 公式检测 |
| `onnx_pipeline/verify/verify_formula_recognize.py` | 公式识别 |
| `onnx_pipeline/verify/verify_no_framework_deps.py` | 无框架依赖 |
