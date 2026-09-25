#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 FP16 / 量化版 MFD ONNX 模型以减小体积
"""

import os
import shutil
from pathlib import Path
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


def fmt(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'


def main():
    print('=' * 74)
    print('MFD ONNX 体积优化')
    print('=' * 74)

    from ultralytics import YOLO

    pt = FD / 'yolo_v8_ft.pt'
    model = YOLO(str(pt))

    # ---------- 1. FP16 导出 ----------
    print('\n【1】导出 FP16 版本')
    print('-' * 74)
    try:
        out = model.export(format='onnx', opset=12, simplify=True,
                           imgsz=1024, half=True, dynamic=False)
        src = Path(out)
        dst = FD / 'mfd_yolov8_fp16.onnx'
        if src.exists():
            shutil.move(str(src), str(dst))
        print(f'  [OK] {dst.name}  ({fmt(dst.stat().st_size)})')
    except Exception as e:
        print(f'  [ERROR] {type(e).__name__}: {e}')

    # ---------- 2. 对比 ----------
    print('\n【2】体积对比')
    print('-' * 74)
    for p in sorted(FD.glob('*.onnx')):
        print(f'  {p.name:28} {fmt(p.stat().st_size)}')

    # ---------- 3. 验证 FP16 能否推理 ----------
    print('\n【3】验证 FP16 ONNX 推理')
    print('-' * 74)
    fp16 = FD / 'mfd_yolov8_fp16.onnx'
    if not fp16.exists():
        print('  [SKIP] FP16 文件不存在')
        return

    import onnxruntime as ort
    import numpy as np
    import cv2
    import time
    for name in ['mfd_yolov8.onnx', 'mfd_yolov8_fp16.onnx']:
        p = FD / name
        if not p.exists():
            continue
        try:
            sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
            inp = sess.get_inputs()[0]
            print(f'\n  {name}:')
            print(f'    Input : {inp.name} {inp.shape} {inp.type}')

            h, w = inp.shape[2], inp.shape[3]
            blob = np.random.rand(1, 3, h, w).astype(np.float32)
            t0 = time.time()
            outs = sess.run(None, {inp.name: blob})
            dt = time.time() - t0
            print(f'    推理: {dt:.3f}s, 输出 {[o.shape for o in outs]}')
            print(f'    [OK] 可正常推理')
        except Exception as e:
            print(f'\n  {name}: [ERROR] {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
