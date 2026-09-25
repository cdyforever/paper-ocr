#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确度量: 行内 MFD 的实际损失
=============================

上一版用 IoU>=0.5 判定, 可能把「部分检出」误判为「完全丢失」。
这里改用两种度量:
  - 完全丢失: 行内 MFD 在该区域完全无检出
  - 部分检出: 有检出但覆盖 < 90%
  - 完整检出: 覆盖 >= 90%
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
OUT.mkdir(parents=True, exist_ok=True)


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


def coverage(fb, boxes):
    hits = [b for b in boxes if inter_area(fb, b) > 0]
    if not hits:
        return 0.0, 0.0
    xs1 = min(b[0] for b in hits); xs2 = max(b[2] for b in hits)
    ys1 = min(b[1] for b in hits); ys2 = max(b[3] for b in hits)
    cx1, cx2 = max(xs1, fb[0]), min(xs2, fb[2])
    cy1, cy2 = max(ys1, fb[1]), min(ys2, fb[3])
    fw, fh = fb[2]-fb[0], fb[3]-fb[1]
    return (max(0.0, cx2-cx1)/fw if fw else 0,
            max(0.0, cy2-cy1)/fh if fh else 0)


def main():
    print('=' * 78)
    print('精确度量: 行内 MFD vs 整页 MFD')
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

    print(f'\n  整页 MFD : {len(page)} 个区域')
    print(f'  DBNet 行 : {len(lines)} 行')

    # 逐行跑 MFD
    inline = []
    for i, r in enumerate(lines, 1):
        lb = quad_to_xyxy(r['box'])
        x1, y1, x2, y2 = [int(v) for v in lb]
        pad = 10
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(W, x2+pad), min(H, y2+pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue
        for s in det(crop):
            sb = s['box']
            inline.append([sb[0]+cx1, sb[1]+cy1, sb[2]+cx1, sb[3]+cy1])

    print(f'  行内 MFD : {len(inline)} 个区域')

    # ---------- 三类判定 ----------
    print('\n' + '=' * 78)
    print('整页区域在行内 MFD 中的覆盖情况')
    print('=' * 78)
    print(f'\n{"#":<4} {"类型":<10} {"尺寸":<12} {"宽覆盖":>8} {"高覆盖":>8} {"判定":<12}')
    print('-' * 64)

    cats = {'完整检出': 0, '部分检出': 0, '完全丢失': 0}
    detail = []
    for i, f in enumerate(page, 1):
        wc, hc = coverage(f['box'], inline)
        if wc == 0 and hc == 0:
            verdict = '完全丢失'
        elif wc >= 0.9 and hc >= 0.9:
            verdict = '完整检出'
        else:
            verdict = '部分检出'
        cats[verdict] += 1
        fw = int(f['box'][2]-f['box'][0]); fh = int(f['box'][3]-f['box'][1])
        print(f'{i:<4} {f["type"]:<10} {fw}x{fh:<7} {wc*100:>7.0f}% '
              f'{hc*100:>7.0f}% {verdict:<12}')
        detail.append({'idx': i, 'type': f['type'], 'box': f['box'],
                       'w_cov': round(wc, 3), 'h_cov': round(hc, 3),
                       'verdict': verdict})

    print('-' * 64)
    print('\n  --- 统计 ---')
    for k in ['完整检出', '部分检出', '完全丢失']:
        v = cats[k]
        print(f'    {k:<8}: {v:2d}/{len(page)} ({v/len(page)*100:.0f}%)')

    # ---------- 关键: 是否完全丢失 ----------
    n_lost = cats['完全丢失']
    print(f'\n  => 行内 MFD 完全丢失 {n_lost}/{len(page)} '
          f'({n_lost/len(page)*100:.0f}%) 的公式区域')

    # ---------- 可视化 ----------
    vis = img.copy()
    cmap = {'完整检出': (0, 200, 0), '部分检出': (0, 165, 255),
            '完全丢失': (0, 0, 255)}
    for d in detail:
        b = d['box']
        c = cmap[d['verdict']]
        cv2.rectangle(vis, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), c, 5)
    cv2.putText(vis, f'In-line MFD coverage: '
                     f'{cats["完整检出"]} full / {cats["部分检出"]} partial / '
                     f'{cats["完全丢失"]} lost',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4)
    cv2.imwrite(str(OUT / 'inline_coverage.jpg'), vis)
    print(f'\n  可视化: {OUT / "inline_coverage.jpg"}')
    print('    绿=完整检出  橙=部分检出  红=完全丢失')

    rep = {'page_total': len(page), 'inline_total': len(inline),
           'categories': cats, 'detail': detail}
    rp = OUT / 'precise_coverage.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'  报告: {rp}')


if __name__ == '__main__':
    main()
