# -*- coding: utf-8 -*-
"""算法模块 (纯 ONNX 实现)。

- OnnxOCREngine          文本检测 + 方向分类 + 文本识别 (PP-OCRv4 ONNX)
- FormulaDetectorONNX    公式区域检测 (YOLOv8l MFD ONNX)
- FormulaRecognizerONNX  公式识别 (UniMERNet-tiny ONNX, KV cache 解码)
- OnnxPagePipeline       页面级组合管线 (公式检测 -> 涂白 -> 双路识别 -> 合并)
"""
