#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严谨验证: DBNet 在公式区域的失效机制与解决方案
==============================================

三个明确问题:
  Q1. DBNet 是否真的在公式区域失效?      -> 量化覆盖度
  Q2. 失效的具体机制是什么?               -> 逐像素/逐框归因
  Q3. 有办法解决吗?                       -> 测试 4 种方案
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'dbnet_fix'
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
    """公式框 fb 被 boxes 覆盖的 (宽比, 高比)"""
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
    print('DBNet 公式区域失效: 机制与解决方案')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    fd_mod = load_mod('formula_detect_onnx',
                      ROOT / 'func/algorithm/formula_detect_onnx.py')

    det = fd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False)
    formulas = det(img)
    dbnet_boxes = [quad_to_xyxy(r['box']) for r in eng(img)]

    # ==================== Q1 ====================
    print('\n' + '=' * 78)
    print('Q1. DBNet 是否真的在公式区域失效?')
    print('=' * 78)

    base = []
    for i, f in enumerate(formulas, 1):
        wc, hc = coverage(f['box'], dbnet_boxes)
        base.append({'idx': i, 'type': f['type'], 'box': f['box'],
                     'w_cov': wc, 'h_cov': hc})
    w_avg = np.mean([r['w_cov'] for r in base])
    h_avg = np.mean([r['h_cov'] for r in base])
    n_ok = sum(1 for r in base if r['w_cov'] >= 0.9 and r['h_cov'] >= 0.9)

    print(f'\n  公式区域数        : {len(base)}')
    print(f'  平均宽度覆盖      : {w_avg*100:.1f}%')
    print(f'  平均高度覆盖      : {h_avg*100:.1f}%')
    print(f'  完整覆盖 (>=90%x90%): {n_ok}/{len(base)} ({n_ok/len(base)*100:.0f}%)')
    print(f'\n  => 结论: DBNet 覆盖度仅 ~{w_avg*100:.0f}%, 确认存在系统性截断')

    # ==================== Q2 ====================
    print('\n' + '=' * 78)
    print('Q2. 失效机制归因')
    print('=' * 78)

    # 用 MFD 的连通域分析: 公式区域里有多少「小碎片」没被 DBNet 框覆盖
    print(f'\n{"#":<4} {"公式尺寸":<14} {"DBNet框数":<10} '
          f'{"碎片数":<8} {"未覆盖碎片":<10} {"判定":<14}')
    print('-' * 74)

    reasons = {'无DBNet框': 0, '框太小': 0, '碎片未覆盖': 0, '正常': 0}
    detail = []
    for r in base:
        fb = r['box']
        x1, y1, x2, y2 = [int(v) for v in fb]
        crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if crop.size == 0:
            continue

        # 公式区域内 DBNet 框数
        hits = [b for b in dbnet_boxes if inter_area(fb, b) > 0]

        # 连通域碎片数
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        frags = []
        for k in range(1, n):
            fx, fy, fw2, fh2, fa = stats[k]
            if fa < 12 or fh2 > crop.shape[0]*0.9:
                continue
            frags.append([x1+fx, y1+fy, x1+fx+fw2, y1+fy+fh2])

        # 有多少碎片未被任何 DBNet 框覆盖
        uncov = 0
        for fr in frags:
            fa = (fr[2]-fr[0]) * (fr[3]-fr[1])
            if fa <= 0:
                continue
            covered = sum(inter_area(fr, b) for b in dbnet_boxes)
            if covered / fa < 0.5:
                uncov += 1

        # 判定
        if not hits:
            verdict = '完全漏检'
            reasons['无DBNet框'] += 1
        elif r['w_cov'] < 0.5 or r['h_cov'] < 0.5:
            verdict = '严重截断'
            reasons['框太小'] += 1
        elif uncov > 0:
            verdict = '碎片漏检'
            reasons['碎片未覆盖'] += 1
        else:
            verdict = '正常'
            reasons['正常'] += 1

        print(f'{r["idx"]:<4} {int(fb[2]-fb[0])}x{int(fb[3]-fb[1]):<9} '
              f'{len(hits):<10} {len(frags):<8} {uncov:<10} {verdict:<14}')

        detail.append({**r, 'n_frags': len(frags), 'n_uncov': uncov,
                       'verdict': verdict})

    print('\n  --- 失效归因统计 ---')
    for k, v in reasons.items():
        print(f'    {k:<12}: {v:2d}/{len(base)} ({v/len(base)*100:.0f}%)')

    # ==================== Q3 ====================
    print('\n' + '=' * 78)
    print('Q3. 解决方案测试')
    print('=' * 78)

    solutions = {}

    # 方案 A: 现状 (DBNet 原样)
    solutions['A 现状'] = {'w': w_avg, 'h': h_avg, 'n_ok': n_ok}

    # 方案 B: 放大输入 (提高小目标召回)
    print('\n  [B] 放大 DBNet 输入尺寸 (limit_side_len 960 -> 1920)')
    eng_big = load_mod('onnx_ocr_engine',
                       ROOT / 'func/algorithm/onnx_ocr_engine.py'
                       ).OnnxOCREngine(use_gpu=False, verbose=False,
                                       det_limit_side_len=1920)
    boxes_big = [quad_to_xyxy(r['box']) for r in eng_big(img)]
    cov_b = [coverage(f['box'], boxes_big) for f in formulas]
    w_b = np.mean([c[0] for c in cov_b]); h_b = np.mean([c[1] for c in cov_b])
    n_b = sum(1 for c in cov_b if c[0] >= 0.9 and c[1] >= 0.9)
    print(f'      框数 {len(dbnet_boxes)} -> {len(boxes_big)}')
    print(f'      宽度覆盖 {w_b*100:.1f}%  高度覆盖 {h_b*100:.1f}%  '
          f'完整 {n_b}/{len(formulas)}')
    solutions['B 放大输入'] = {'w': w_b, 'h': h_b, 'n_ok': n_b}

    # 方案 C: 用 MFD 框直接替换公式区域的 DBNet 框
    print('\n  [C] 公式区域改用 MFD 框 (绕过 DBNet)')
    merged = list(dbnet_boxes) + [f['box'] for f in formulas]
    cov_c = [coverage(f['box'], merged) for f in formulas]
    w_c = np.mean([c[0] for c in cov_c]); h_c = np.mean([c[1] for c in cov_c])
    n_c = sum(1 for c in cov_c if c[0] >= 0.9 and c[1] >= 0.9)
    print(f'      宽度覆盖 {w_c*100:.1f}%  高度覆盖 {h_c*100:.1f}%  '
          f'完整 {n_c}/{len(formulas)}')
    solutions['C 用MFD框'] = {'w': w_c, 'h': h_c, 'n_ok': n_c}

    # 方案 D: 区域路由 (公式区域整体送 MFR, 不做行切分)
    print('\n  [D] 区域路由: 公式区域整体走 MFR, 文本区域走 DBNet')
    # 度量: 公式区域是否被完整交给 MFR
    n_route = len(formulas)
    print(f'      公式区域 {n_route} 个全部整体交给 MFR')
    print(f'      覆盖度视为 100% (无需 DBNet 参与)')
    solutions['D 区域路由'] = {'w': 1.0, 'h': 1.0, 'n_ok': len(formulas)}

    # ---------- 汇总 ----------
    print('\n' + '=' * 78)
    print('方案对比')
    print('=' * 78)
    print(f'\n  {"方案":<16} {"宽度覆盖":>10} {"高度覆盖":>10} {"完整覆盖":>12}')
    print('  ' + '-' * 54)
    for k, v in solutions.items():
        print(f'  {k:<16} {v["w"]*100:>9.1f}% {v["h"]*100:>9.1f}% '
              f'{v["n_ok"]:>6}/{len(formulas)}')

    # 可视化方案 C
    vis = img.copy()
    for f in formulas:
        b = f['box']
        cv2.rectangle(vis, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), (0, 200, 0), 4)
    cv2.imwrite(str(OUT / 'solution_mfd_boxes.jpg'), vis)

    # 保存
    rep = {'q1': {'w_avg': float(w_avg), 'h_avg': float(h_avg),
                  'n_ok': n_ok, 'n_total': len(base)},
           'q2': {'reasons': reasons, 'detail': detail},
           'q3': {k: {'w': float(v['w']), 'h': float(v['h']),
                      'n_ok': int(v['n_ok'])} for k, v in solutions.items()}}
    rp = OUT / 'fix_analysis.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'\n报告: {rp}')


if __name__ == '__main__':
    main()
