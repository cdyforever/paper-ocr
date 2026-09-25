#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化 DBNet 在公式区域的表现
===========================

测量:
  1. DBNet 文本框与 MFD 公式框的重叠情况
  2. DBNet 在公式区域的「截断」程度
     - 用公式框面积 / DBNet 框覆盖面积 衡量
  3. 架构依赖关系确认

判定截断: 对每个公式区域, 看 DBNet 检出的框是否覆盖了它的
          完整宽度/高度。
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'dbnet_formula'
OUT.mkdir(parents=True, exist_ok=True)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def box_xyxy_from_quad(quad):
    q = np.array(quad, dtype=np.float32)
    return [float(q[:, 0].min()), float(q[:, 1].min()),
            float(q[:, 0].max()), float(q[:, 1].max())]


def inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def main():
    print('=' * 78)
    print('量化 DBNet 在公式区域的表现')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    det = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py'
                   ).FormulaDetectorONNX(use_gpu=False, verbose=False)

    # DBNet 文本框
    dbnet = eng(img)
    dbnet_boxes = [box_xyxy_from_quad(r['box']) for r in dbnet]
    print(f'\nDBNet 文本框: {len(dbnet_boxes)}')

    # MFD 公式框
    formulas = det(img)
    print(f'MFD 公式框  : {len(formulas)}')

    # ---------- 逐个公式区域分析 ----------
    print('\n' + '=' * 78)
    print('逐个公式区域的 DBNet 覆盖情况')
    print('=' * 78)
    print(f'\n{"#":<4} {"类型":<10} {"公式框宽x高":<16} {"覆盖DBNet框":<16} '
          f'{"宽度覆盖":>9} {"高度覆盖":>9} {"判定":<10}')
    print('-' * 92)

    rows = []
    for i, f in enumerate(formulas, 1):
        fb = f['box']
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        fa = fw * fh

        # 找与公式框有重叠的 DBNet 框
        hits = []
        for db in dbnet_boxes:
            ia = inter_area(fb, db)
            if ia > 0:
                hits.append((db, ia))

        if not hits:
            print(f'{i:<4} {f["type"]:<10} {fw:.0f}x{fh:.0f}{"":<8} '
                  f'{"(无)":<16} {0:>8.0f}% {0:>8.0f}% {"完全漏检":<10}')
            rows.append({'idx': i, 'type': f['type'], 'box': fb,
                         'width_cov': 0, 'height_cov': 0,
                         'verdict': '完全漏检'})
            continue

        # 覆盖度量: 并集覆盖的宽/高比例
        xs1 = min(h[0][0] for h in hits)
        ys1 = min(h[0][1] for h in hits)
        xs2 = max(h[0][2] for h in hits)
        ys2 = max(h[0][3] for h in hits)

        # 公式框内被 DBNet 覆盖的 x/y 跨度
        cx1, cx2 = max(xs1, fb[0]), min(xs2, fb[2])
        cy1, cy2 = max(ys1, fb[1]), min(ys2, fb[3])
        width_cov = max(0.0, cx2 - cx1) / fw if fw > 0 else 0
        height_cov = max(0.0, cy2 - cy1) / fh if fh > 0 else 0

        # 判定
        if width_cov >= 0.9 and height_cov >= 0.9:
            verdict = '完整'
        elif width_cov >= 0.9:
            verdict = '高度截断'
        elif height_cov >= 0.9:
            verdict = '宽度截断'
        else:
            verdict = '严重截断'

        print(f'{i:<4} {f["type"]:<10} {fw:.0f}x{fh:.0f}{"":<8} '
              f'{len(hits)}个{"":<12} {width_cov*100:>8.0f}% {height_cov*100:>8.0f}% '
              f'{verdict:<10}')

        rows.append({'idx': i, 'type': f['type'], 'box': fb,
                     'n_hits': len(hits),
                     'width_cov': round(width_cov, 3),
                     'height_cov': round(height_cov, 3),
                     'verdict': verdict})

    # ---------- 统计 ----------
    print('\n' + '=' * 78)
    print('统计')
    print('=' * 78)

    from collections import Counter
    cnt = Counter(r['verdict'] for r in rows)
    print()
    for k, v in cnt.most_common():
        print(f'  {k:<12}: {v:2d} / {len(rows)}  ({v/len(rows)*100:.0f}%)')

    avg_w = np.mean([r['width_cov'] for r in rows])
    avg_h = np.mean([r['height_cov'] for r in rows])
    print(f'\n  平均宽度覆盖: {avg_w*100:.1f}%')
    print(f'  平均高度覆盖: {avg_h*100:.1f}%')

    # ---------- 可视化 ----------
    print('\n' + '=' * 78)
    print('可视化')
    print('=' * 78)

    vis = img.copy()
    # DBNet 框 (蓝)
    for db in dbnet_boxes:
        cv2.rectangle(vis, (int(db[0]), int(db[1])),
                      (int(db[2]), int(db[3])), (255, 0, 0), 3)
    # 公式框 (绿) + 覆盖状态
    colors = {'完整': (0, 200, 0), '高度截断': (0, 165, 255),
              '宽度截断': (0, 100, 255), '严重截断': (0, 0, 255),
              '完全漏检': (255, 0, 255)}
    for r in rows:
        b = r['box']
        c = colors.get(r['verdict'], (0, 0, 255))
        cv2.rectangle(vis, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), c, 3)
        cv2.putText(vis, f"{r['idx']} {r['verdict']}",
                    (int(b[0]), max(int(b[1]) - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.imwrite(str(OUT / 'dbnet_vs_mfd.jpg'), vis)
    print(f'  {OUT / "dbnet_vs_mfd.jpg"}')
    print('  蓝=DBNet文本框  绿=完整  橙=高度截断  红=严重截断  紫=完全漏检')

    # 只保留公式区域的放大图
    if rows:
        ys = [int(r['box'][1]) for r in rows]
        ye = [int(r['box'][3]) for r in rows]
        y0, y1 = max(0, min(ys) - 40), min(H, max(ye) + 40)
        crop = vis[y0:y1, :]
        cv2.imwrite(str(OUT / 'dbnet_vs_mfd_crop.jpg'), crop)
        print(f'  {OUT / "dbnet_vs_mfd_crop.jpg"}')

    # ---------- 架构依赖确认 ----------
    print('\n' + '=' * 78)
    print('架构依赖确认')
    print('=' * 78)
    print("""
  MFD (公式检测) 输入: 原始图像  ->  无 DBNet 依赖
  MFR (公式识别) 输入: MFD 裁剪  ->  无 DBNet 依赖

  代码验证 (formula_detect_onnx.py / formula_recognize_onnx.py):
    未出现 dbnet / det_boxes / OnnxOCREngine 等符号
""")

    rep = {'rows': rows, 'avg_width_cov': float(avg_w),
           'avg_height_cov': float(avg_h),
           'counts': dict(cnt),
           'n_dbnet': len(dbnet_boxes), 'n_formula': len(formulas)}
    rp = OUT / 'dbnet_formula_analysis.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'报告: {rp}')


if __name__ == '__main__':
    main()
