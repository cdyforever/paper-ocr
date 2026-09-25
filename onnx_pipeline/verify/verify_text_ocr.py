#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯 ONNX OCR 引擎完整测试
========================

测试内容:
1. 英文 + 数字识别
2. 中文识别 (使用 PIL 正确渲染中文字体)
3. 中英文混合
4. 性能统计

输出:
- weights/onnx_ocr/test_english.jpg
- weights/onnx_ocr/test_chinese.jpg
- weights/onnx_ocr/test_mixed.jpg
- weights/onnx_ocr/results.json
"""

import os
import sys
import json
import time
import importlib.util
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

# Load the engine module directly by file path so that importing it does not
# trigger func/algorithm/__init__.py (which still imports legacy Paddle code).
# 本脚本位于 onnx_pipeline/verify/，项目根在上两级。
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent if _HERE.name == 'verify' else _HERE
_ENGINE_PATH = _ROOT / 'func' / 'algorithm' / 'onnx_ocr_engine.py'
_spec = importlib.util.spec_from_file_location('onnx_ocr_engine', _ENGINE_PATH)
_engine_mod = importlib.util.module_from_spec(_spec)
sys.modules['onnx_ocr_engine'] = _engine_mod
_spec.loader.exec_module(_engine_mod)
OnnxOCREngine = _engine_mod.OnnxOCREngine


OUT_DIR = Path.cwd() / 'weights' / 'onnx_ocr'


def find_chinese_font():
    """Locate a usable Chinese font on Windows."""
    candidates = [
        r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
        r'C:\Windows\Fonts\msyhbd.ttc',
        r'C:\Windows\Fonts\simhei.ttf',    # 黑体
        r'C:\Windows\Fonts\simsun.ttc',    # 宋体
        r'C:\Windows\Fonts\Deng.ttf',      # 等线
        r'C:\Windows\Fonts\arial.ttf',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def make_image(lines, font_path, font_size=40, width=1000, margin=40, line_gap=30):
    """Render text lines into a white BGR image using PIL (UTF-8 safe)."""
    font = ImageFont.truetype(font_path, font_size)

    # Measure
    tmp = Image.new('RGB', (10, 10), 'white')
    td = ImageDraw.Draw(tmp)
    heights = []
    for ln in lines:
        bbox = td.textbbox((0, 0), ln, font=font)
        heights.append(bbox[3] - bbox[1])

    line_h = max(heights) + line_gap
    height = margin * 2 + line_h * len(lines)

    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        y = margin + i * line_h
        draw.text((margin, y), ln, fill='black', font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run_case(engine, name, image, lines_expected):
    """Run OCR on one image and report results."""
    print('\n' + '=' * 70)
    print(f'用例: {name}')
    print('=' * 70)

    img_path = OUT_DIR / f'test_{name}.jpg'
    cv2.imwrite(str(img_path), image)
    print(f'图片: {img_path}')

    t0 = time.time()
    results = engine(image)
    elapsed = time.time() - t0

    print(f'\n检测到 {len(results)} 行文本 (耗时 {elapsed:.3f}s):')
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['text']!r}  score={r['score']:.4f}")

    # 可视化
    vis = engine.draw(image, results)
    vis_path = OUT_DIR / f'result_{name}.jpg'
    cv2.imwrite(str(vis_path), vis)
    print(f'可视化: {vis_path}')

    # 简单匹配率
    recognized = [r['text'] for r in results]
    print(f'\n期望: {lines_expected}')
    print(f'实际: {recognized}')

    return {
        'case': name,
        'image': str(img_path),
        'visualization': str(vis_path),
        'elapsed_s': round(elapsed, 3),
        'num_lines': len(results),
        'expected': lines_expected,
        'results': results,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('纯 ONNX OCR 引擎完整测试')
    print('=' * 70)

    font_path = find_chinese_font()
    print(f'\n使用字体: {font_path}')

    # 初始化引擎
    engine = OnnxOCREngine(use_gpu=False, verbose=True)

    all_reports = []

    # ---- 用例 1: 英文 + 数字 ----
    lines_en = [
        'Hello World!',
        'Numbers: 1234567890',
        'Email: test@example.com',
        'Price: $19.99 (50% off)',
    ]
    img_en = make_image(lines_en, font_path or r'C:\Windows\Fonts\arial.ttf')
    all_reports.append(run_case(engine, 'english', img_en, lines_en))

    # ---- 用例 2: 中文 ----
    lines_cn = [
        '这是一段中文测试文本',
        '人工智能改变世界',
        '发票号码：12345678',
        '金额：人民币壹万元整',
    ]
    if font_path:
        img_cn = make_image(lines_cn, font_path)
        all_reports.append(run_case(engine, 'chinese', img_cn, lines_cn))
    else:
        print('\n[WARN] 未找到中文字体，跳过中文用例')

    # ---- 用例 3: 中英混合 ----
    lines_mix = [
        '产品名称: iPhone 15 Pro',
        '数量 Qty: 3',
        '总价 Total: ¥8,997.00',
        '备注 Remark: 加急处理',
    ]
    if font_path:
        img_mix = make_image(lines_mix, font_path)
        all_reports.append(run_case(engine, 'mixed', img_mix, lines_mix))

    # ---- 汇总 ----
    print('\n' + '=' * 70)
    print('汇总')
    print('=' * 70)
    total_lines = sum(r['num_lines'] for r in all_reports)
    total_time = sum(r['elapsed_s'] for r in all_reports)
    print(f'用例数: {len(all_reports)}')
    print(f'总文本行: {total_lines}')
    print(f'总耗时: {total_time:.3f}s')
    if total_lines:
        print(f'平均每行: {total_time / total_lines * 1000:.1f}ms')

    # 保存 JSON
    report_path = OUT_DIR / 'results.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {report_path}')

    print('\n[OK] 纯 ONNX OCR 测试完成（全程未使用 PaddlePaddle）')


if __name__ == '__main__':
    main()
