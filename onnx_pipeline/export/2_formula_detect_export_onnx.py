#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinerU MFD (YOLOv8) 实际测试 + ONNX 导出
=========================================

1. 用真实含公式的图片测试检测效果
2. 导出为 ONNX (供纯 ONNX 运行时使用)
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os
from pathlib import Path as _Path

# --- 自动定位项目根, 使 Path.cwd() 指向项目根 (shim) ---
def _chdir_project_root():
    _p = _Path(__file__).resolve()
    for _parent in [_p] + list(_p.parents):
        if (_parent / 'weights').is_dir() and (_parent / 'func').is_dir():
            _os.chdir(_parent)
            return _parent
    return _Path.cwd()

_chdir_project_root()
# --- shim end ---

ROOT = Path.cwd()
FD = ROOT / 'weights' / 'formula_detect'
OUT = ROOT / 'weights' / 'onnx_ocr'
OUT.mkdir(parents=True, exist_ok=True)


def make_test_image():
    """生成含行内公式和独立公式的测试图。"""
    font_path = r'C:\Windows\Fonts\msyh.ttc'
    if not os.path.exists(font_path):
        font_path = r'C:\Windows\Fonts\simhei.ttf'

    W, H = 1000, 560
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    f_big = ImageFont.truetype(font_path, 46)
    f_txt = ImageFont.truetype(font_path, 26)

    # 标题
    d.text((40, 25), '数学试卷示例', fill='black', font=f_big)

    # 正文 + 行内公式 (embedding)
    d.text((40, 110), '1. 已知函数 f(x) = x² + 2x - 3，求 f(1) 的值。', fill='black', font=f_txt)
    d.text((40, 160), '2. 设 a > 0 且 b > 0，证明 (a+b)/2 ≥ √(ab)。', fill='black', font=f_txt)

    # 独立公式 (isolated) — 居中单独成行
    d.text((330, 235), 'x = (-b ± √(b² - 4ac)) / 2a', fill='black', font=f_txt)
    d.text((300, 310), '∫₀¹ x² dx = 1/3', fill='black', font=f_txt)

    # 更多正文
    d.text((40, 385), '3. 计算下列定积分，并写出完整的解题步骤。', fill='black', font=f_txt)
    d.text((40, 435), '4. 求矩阵 A 的特征值与特征向量。', fill='black', font=f_txt)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    p = OUT / 'formula_test_input.jpg'
    cv2.imwrite(str(p), bgr)
    return bgr, p


def main():
    print('=' * 74)
    print('MinerU MFD (YOLOv8) 测试 + ONNX 导出')
    print('=' * 74)

    from ultralytics import YOLO

    model_path = FD / 'yolo_v8_ft.pt'
    model = YOLO(str(model_path))
    print(f'\n模型: {model_path.name}')
    print(f'类别: {model.names}')

    # ---- 1. 生成测试图 ----
    print('\n【1】生成测试图')
    print('-' * 74)
    bgr, img_path = make_test_image()
    print(f'  [OK] {img_path}  {bgr.shape[1]}x{bgr.shape[0]}')

    # ---- 2. 推理 ----
    print('\n【2】公式检测推理')
    print('-' * 74)
    t0 = time.time()
    results = model.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)
    dt = time.time() - t0
    print(f'  耗时: {dt:.3f}s')

    r = results[0]
    boxes = r.boxes
    print(f'  检测到 {len(boxes)} 个公式区域:')

    vis = bgr.copy()
    colors = {0: (0, 165, 255), 1: (0, 200, 0)}   # embedding=橙, isolated=绿
    counts = {}
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        name = model.names[cls]
        counts[name] = counts.get(name, 0) + 1
        print(f'    [{i}] {name:10} conf={conf:.3f}  box={xyxy.tolist()}')

        c = colors.get(cls, (255, 0, 0))
        cv2.rectangle(vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 2)
        label = f'{name} {conf:.2f}'
        cv2.putText(vis, label, (xyxy[0], max(xyxy[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

    print(f'\n  类别统计: {counts}')
    vis_path = OUT / 'formula_detect_result.jpg'
    cv2.imwrite(str(vis_path), vis)
    print(f'  可视化: {vis_path}')

    # ---- 3. 导出 ONNX ----
    print('\n【3】导出 ONNX')
    print('-' * 74)
    onnx_path = FD / 'mfd_yolov8.onnx'
    try:
        exported = model.export(format='onnx', opset=12, simplify=True,
                                imgsz=1024, dynamic=False)
        print(f'  [OK] 导出成功: {exported}')
        # 移动到目标位置
        exp = Path(exported)
        if exp.exists() and exp != onnx_path:
            import shutil
            shutil.move(str(exp), str(onnx_path))
            print(f'  已移动到: {onnx_path}  ({onnx_path.stat().st_size/1024/1024:.1f} MB)')
    except Exception as e:
        import traceback
        print(f'  [ERROR] {type(e).__name__}: {e}')
        traceback.print_exc(limit=3)

    # ---- 4. 验证 ONNX ----
    if onnx_path.exists():
        print('\n【4】验证导出的 ONNX')
        print('-' * 74)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        for i in sess.get_inputs():
            print(f'  Input : {i.name} {i.shape} {i.type}')
        for o in sess.get_outputs():
            print(f'  Output: {o.name} {o.shape} {o.type}')

        # 用 ONNX 跑一遍并对比
        inp_name = sess.get_inputs()[0].name
        h, w = sess.get_inputs()[0].shape[2], sess.get_inputs()[0].shape[3]
        img = cv2.resize(bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        t0 = time.time()
        outs = sess.run(None, {inp_name: blob})
        print(f'  ONNX 推理耗时: {time.time()-t0:.3f}s')
        print(f'  输出 shape: {[o.shape for o in outs]}')


if __name__ == '__main__':
    main()
