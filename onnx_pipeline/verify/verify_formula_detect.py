#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试纯 ONNX 公式检测器
======================

1. 与 ultralytics 结果对比
2. 验证无 torch/ultralytics 依赖
3. 可视化输出
"""

import sys
import os
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np
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
OUT = ROOT / 'weights' / 'onnx_ocr'


class TorchBlocker:
    """拦截 torch / ultralytics，证明纯 ONNX 路径无这些依赖。"""
    BLOCKED = ('torch', 'torchvision', 'ultralytics', 'paddle', 'paddleocr')

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.BLOCKED:
            raise ImportError(f'[BLOCKED] {fullname}')
        return None


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def make_test_image():
    """生成含行内公式和独立公式的测试图（自给自足，不依赖外部文件）。"""
    from PIL import Image, ImageDraw, ImageFont
    font_path = r'C:\Windows\Fonts\msyh.ttc'
    if not os.path.exists(font_path):
        font_path = r'C:\Windows\Fonts\simhei.ttf'

    W, H = 1000, 560
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    f_big = ImageFont.truetype(font_path, 46)
    f_txt = ImageFont.truetype(font_path, 26)

    d.text((40, 25), '数学试卷示例', fill='black', font=f_big)
    d.text((40, 110), '1. 已知函数 f(x) = x² + 2x - 3，求 f(1) 的值。', fill='black', font=f_txt)
    d.text((40, 160), '2. 设 a > 0 且 b > 0，证明 (a+b)/2 ≥ √(ab)。', fill='black', font=f_txt)
    d.text((330, 235), 'x = (-b ± √(b² - 4ac)) / 2a', fill='black', font=f_txt)
    d.text((300, 310), '∫₀¹ x² dx = 1/3', fill='black', font=f_txt)
    d.text((40, 385), '3. 计算下列定积分，并写出完整的解题步骤。', fill='black', font=f_txt)
    d.text((40, 435), '4. 求矩阵 A 的特征值与特征向量。', fill='black', font=f_txt)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'formula_test_input.jpg'
    cv2.imwrite(str(p), bgr)
    return bgr, p


def main():
    print('=' * 74)
    print('纯 ONNX 公式检测器测试')
    print('=' * 74)

    img, img_path = make_test_image()
    print(f'\n测试图: {img_path.name}  {img.shape[1]}x{img.shape[0]}')

    # ---------- 1. 参考: ultralytics ----------
    print('\n【1】参考结果 (ultralytics YOLO)')
    print('-' * 74)
    ref = []
    try:
        from ultralytics import YOLO
        m = YOLO(str(ROOT / 'weights/formula_detect/yolo_v8_ft.pt'))
        t0 = time.time()
        r = m.predict(str(img_path), conf=0.25, iou=0.45, verbose=False)[0]
        dt_ref = time.time() - t0
        for b in r.boxes:
            xyxy = [float(v) for v in b.xyxy[0].cpu().numpy()]
            cls = int(b.cls[0].cpu().numpy())
            ref.append({
                'box': xyxy,
                'type': m.names[cls],
                'score': float(b.conf[0].cpu().numpy()),
            })
        print(f'  检测 {len(ref)} 个, 耗时 {dt_ref:.3f}s')
        for x in ref:
            print(f"    {x['type']:10} {x['score']:.3f}  {[int(v) for v in x['box']]}")
    except Exception as e:
        print(f'  [WARN] 无法运行 ultralytics: {e}')

    # ---------- 2. 纯 ONNX (拦截 torch) ----------
    print('\n【2】纯 ONNX 结果 (拦截 torch/ultralytics)')
    print('-' * 74)

    # 清除第一步已缓存的模块，否则拦截器不会生效
    for name in list(sys.modules):
        if name.split('.')[0] in TorchBlocker.BLOCKED:
            del sys.modules[name]
    import gc
    gc.collect()

    sys.meta_path.insert(0, TorchBlocker())
    for mod in ('torch', 'ultralytics'):
        try:
            __import__(mod)
            print(f'  [FAIL] {mod} 未被拦截')
            return 1
        except ImportError:
            print(f'  [OK] {mod} 已拦截')

    mod = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py')
    det = mod.FormulaDetectorONNX(use_gpu=False, verbose=True)

    t0 = time.time()
    got = det(img)
    dt_onnx = time.time() - t0
    print(f'\n  检测 {len(got)} 个, 耗时 {dt_onnx:.3f}s')
    for x in got:
        print(f"    {x['type']:10} {x['score']:.3f}  {[int(v) for v in x['box']]}")

    # ---------- 3. 对比 ----------
    print('\n【3】结果一致性对比')
    print('-' * 74)
    if ref:
        matched = 0
        for g in got:
            best = max((iou_xyxy(g['box'], r['box']) for r in ref), default=0)
            ok = best > 0.5
            matched += ok
            print(f"    ONNX {g['type']:10} {[int(v) for v in g['box']]}  "
                  f"-> 最佳IoU={best:.3f}  {'OK' if ok else 'MISS'}")
        print(f'\n  匹配: {matched}/{len(got)}  '
              f"({matched/max(len(got),1)*100:.0f}%)")
        print(f'  速度: ONNX {dt_onnx:.3f}s  vs  ultralytics {dt_ref:.3f}s')
    else:
        print('  (无参考结果)')

    # ---------- 4. 可视化 ----------
    vis = det.draw(img, got)
    vis_path = OUT / 'formula_detect_pure_onnx.jpg'
    cv2.imwrite(str(vis_path), vis)
    print(f'\n可视化: {vis_path}')

    # 检查是否有 torch 被加载
    loaded = [m for m in sys.modules if m.split('.')[0] in
              ('torch', 'ultralytics', 'paddle', 'paddleocr')]
    print(f'已加载的 torch/paddle 模块: {loaded if loaded else "无"}')

    print('\n' + '=' * 74)
    if not loaded:
        print('[SUCCESS] 纯 ONNX 公式检测器运行成功，无 torch/ultralytics/paddle 依赖')
    return 0


if __name__ == '__main__':
    sys.exit(main())
