#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
干净测试: 信息可用性 (排除 resize 干扰)
=======================================

上一版把行框裁剪喂给 MFD, 小裁剪被放大 9 倍可能本身就会失败。
这里只测【信息可用性】: 公式区域有多少面积落在 DBNet 行框内。

若公式面积大部分不在任何 DBNet 框内, 则行内 MFD 从信息上就不可能成功
—— 与 resize 无关。
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'pipeline_coupling'


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def quad_to_xyxy(q):
    q = np.array(q, dtype=np.float32)
    return [float(q[:, 0].min()), float(q[:, 1].min()),
            float(q[:, 0].max()), float(q[:, 1].max())]


def inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2-x1) * max(0.0, y2-y1)


def main():
    print('=' * 78)
    print('信息可用性测试: 公式区域有多少落在 DBNet 行框内')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    det = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py'
                   ).FormulaDetectorONNX(use_gpu=False, verbose=False)

    page = det(img)
    lines = eng(img)
    line_boxes = [quad_to_xyxy(r['box']) for r in lines]

    print(f'\n  公式区域: {len(page)}')
    print(f'  DBNet 行: {len(lines)}')

    # 行框的并集面积
    line_area = sum((b[2]-b[0])*(b[3]-b[1]) for b in line_boxes)
    print(f'  DBNet 行框总面积: {line_area/1e6:.2f} M px²')
    print(f'  页面面积        : {W*H/1e6:.2f} M px²')
    print(f'  覆盖率          : {line_area/(W*H)*100:.1f}%')

    # 每个公式区域: 落在行框内的面积比例
    print(f'\n{"#":<4} {"类型":<10} {"公式尺寸":<12} {"落在行框内":>12} '
          f'{"判定":<14}')
    print('-' * 60)

    ratios = []
    cats = {'完全可用': 0, '部分可用': 0, '不可用': 0}
    detail = []
    for i, f in enumerate(page, 1):
        fb = f['box']
        fa = (fb[2]-fb[0]) * (fb[3]-fb[1])
        if fa <= 0:
            continue
        covered = sum(inter_area(fb, b) for b in line_boxes)
        ratio = min(1.0, covered / fa)
        ratios.append(ratio)

        if ratio >= 0.95:
            v = '完全可用'
        elif ratio >= 0.5:
            v = '部分可用'
        else:
            v = '不可用'
        cats[v] += 1

        fw = int(fb[2]-fb[0]); fh = int(fb[3]-fb[1])
        print(f'{i:<4} {f["type"]:<10} {fw}x{fh:<7} {ratio*100:>11.0f}% {v:<14}')
        detail.append({'idx': i, 'type': f['type'], 'box': fb,
                       'avail_ratio': round(ratio, 3), 'verdict': v})

    print('-' * 60)
    print(f'\n  平均可用率: {np.mean(ratios)*100:.1f}%')
    print(f'\n  --- 统计 ---')
    for k in ['完全可用', '部分可用', '不可用']:
        v = cats[k]
        print(f'    {k:<8}: {v:2d}/{len(page)} ({v/len(page)*100:.0f}%)')

    print(f"""
  结论:
    公式区域平均只有 {np.mean(ratios)*100:.0f}% 的面积落在 DBNet 行框内。
    这意味着「DBNet 切行 -> 行内公式检测」的信息基础本身就不完整,
    与 MFD 的 resize 处理无关。

    {cats['不可用']}/{len(page)} 的区域可用率 < 50%, 属于结构性丢失。
""")

    # 可视化: 行框并集 vs 公式区域
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in line_boxes:
        cv2.rectangle(mask, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), 255, -1)

    vis = img.copy()
    # 行框内区域染蓝
    vis[mask > 0] = (vis[mask > 0] * 0.6 +
                     np.array([180, 120, 60], dtype=np.float32) * 0.4).astype(np.uint8)
    # 公式框
    for d in detail:
        b = d['box']
        c = {'完全可用': (0, 200, 0), '部分可用': (0, 165, 255),
             '不可用': (0, 0, 255)}[d['verdict']]
        cv2.rectangle(vis, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), c, 5)
    cv2.putText(vis, f'Blue = DBNet line boxes   '
                     f'Green=OK Orange=partial Red=unavailable',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.imwrite(str(OUT / 'info_availability.jpg'), vis)
    print(f'  可视化: {OUT / "info_availability.jpg"}')

    rep = {'avg_availability': float(np.mean(ratios)),
           'categories': cats, 'detail': detail}
    rp = OUT / 'info_availability.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'  报告: {rp}')


if __name__ == '__main__':
    main()
