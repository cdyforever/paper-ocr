#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证纯 ONNX 引擎不依赖 PaddlePaddle
====================================

通过安装一个 import hook 彻底封杀 paddle / paddleocr 及其所有子模块，
然后运行完整 OCR 流程。如果依然成功，即证明推理阶段零 Paddle 依赖。
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


class PaddleBlocker:
    """Meta path finder that refuses any 'paddle*' / 'paddleocr*' import."""

    BLOCKED = ('paddle', 'paddleocr', 'paddlex', 'paddle2onnx')

    def find_module(self, fullname, path=None):
        root = fullname.split('.')[0]
        if root in self.BLOCKED:
            return self
        return None

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in self.BLOCKED:
            raise ImportError(
                f'[BLOCKED] 导入被测试脚本拦截: {fullname} '
                f'(证明推理过程未使用 PaddlePaddle)')
        return None


def main():
    print('=' * 70)
    print('验证: 纯 ONNX 引擎无 PaddlePaddle 依赖')
    print('=' * 70)

    # 安装拦截器
    sys.meta_path.insert(0, PaddleBlocker())
    print('\n[已启用] PaddlePaddle 导入拦截器')

    # 确认拦截生效
    for mod in ('paddle', 'paddleocr'):
        try:
            __import__(mod)
            print(f'[FAIL] {mod} 仍可导入，拦截失败')
            return 1
        except ImportError:
            print(f'[OK] {mod} 已被成功拦截')

    # 加载引擎 (本脚本位于 onnx_pipeline/verify/，项目根在上两级)
    _here = Path(__file__).resolve().parent
    _root = _here.parent.parent if _here.name == 'verify' else _here
    engine_path = _root / 'func' / 'algorithm' / 'onnx_ocr_engine.py'
    spec = importlib.util.spec_from_file_location('onnx_ocr_engine', engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['onnx_ocr_engine'] = mod
    spec.loader.exec_module(mod)

    print('\n[OK] onnx_ocr_engine 模块加载成功')

    # 生成测试图片
    from PIL import Image, ImageDraw, ImageFont
    font_path = r'C:\Windows\Fonts\msyh.ttc'
    if not os.path.exists(font_path):
        font_path = r'C:\Windows\Fonts\simhei.ttf'

    lines = [
        '无 PaddlePaddle 环境测试',
        'Pure ONNX Runtime Only',
        '发票号码：87654321',
    ]
    font = ImageFont.truetype(font_path, 40)
    img = Image.new('RGB', (900, 320), 'white')
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((40, 30 + i * 90), ln, fill='black', font=font)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 运行引擎
    print('\n初始化引擎...')
    engine = mod.OnnxOCREngine(use_gpu=False, verbose=True)

    print('\n运行 OCR...')
    t0 = time.time()
    results = engine(bgr)
    dt = time.time() - t0

    print(f'\n耗时: {dt:.3f}s')
    print(f'识别到 {len(results)} 行:')
    for r in results:
        print(f"  {r['text']!r}  score={r['score']:.4f}")

    # 验证
    out_dir = Path.cwd() / 'weights' / 'onnx_ocr'
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = engine.draw(bgr, results)
    vis_path = out_dir / 'result_no_paddle.jpg'
    cv2.imwrite(str(vis_path), vis)
    print(f'\n可视化: {vis_path}')

    # 检查是否有 paddle 模块被加载
    loaded_paddle = [m for m in sys.modules if m.split('.')[0] in
                     ('paddle', 'paddleocr', 'paddlex')]
    print(f'\n已加载的 Paddle 相关模块: {loaded_paddle if loaded_paddle else "无"}')

    if loaded_paddle:
        print('[FAIL] 检测到 Paddle 模块被加载')
        return 1

    print('\n' + '=' * 70)
    print('[SUCCESS] 纯 ONNX 引擎在无 PaddlePaddle 环境下运行成功!')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
