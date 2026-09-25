#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 pipeline 耦合问题
======================

发现: 原 paper_ocr.py 的流程是
    DBNet 切行 (text_line_detector.detect)
        -> 逐行做公式检测 (predict_photo 内 detectFormula)
        -> 逐行识别

问题: 公式检测被限制在 DBNet 的**行框内**。若 DBNet 在公式区域
      截断/漏检, 公式检测也随之丢失。

对比:
    A. 行内 MFD (受 DBNet 限制)   — 模拟原 pipeline
    B. 整页 MFD (独立于 DBNet)    — 推荐方案
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


def main():
    print('=' * 78)
    print('验证 pipeline 耦合: 行内 MFD vs 整页 MFD')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    fd_mod = load_mod('formula_detect_onnx',
                      ROOT / 'func/algorithm/formula_detect_onnx.py')
    det = fd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False)

    # ---------- B. 整页 MFD (推荐) ----------
    print('\n' + '=' * 78)
    print('【B】整页 MFD (独立于 DBNet)')
    print('=' * 78)
    page_formulas = det(img)
    print(f'\n  整页检出 {len(page_formulas)} 个公式区域')
    n_emb = sum(1 for f in page_formulas if f['type'] == 'embedding')
    n_iso = sum(1 for f in page_formulas if f['type'] == 'isolated')
    print(f'  embedding={n_emb}  isolated={n_iso}')

    # ---------- A. 行内 MFD (模拟原 pipeline) ----------
    print('\n' + '=' * 78)
    print('【A】行内 MFD (受 DBNet 行框限制, 模拟原 pipeline)')
    print('=' * 78)

    text_lines = eng(img)
    print(f'\n  DBNet 切出 {len(text_lines)} 行')

    # 逐行跑 MFD
    line_formulas = []
    for i, r in enumerate(text_lines, 1):
        lb = quad_to_xyxy(r['box'])
        x1, y1, x2, y2 = [int(v) for v in lb]
        # 加一点边距, 模拟真实用法
        pad = 10
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(W, x2+pad), min(H, y2+pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue

        subs = det(crop)
        for s in subs:
            # 映射回整页坐标
            sb = s['box']
            line_formulas.append({
                'box': [sb[0]+cx1, sb[1]+cy1, sb[2]+cx1, sb[3]+cy1],
                'type': s['type'], 'score': s['score'],
                'from_line': i,
            })

    print(f'  逐行 MFD 共检出 {len(line_formulas)} 个公式区域')

    # ---------- 对比 ----------
    print('\n' + '=' * 78)
    print('对比: 整页 MFD 检出的区域, 有多少能在行内 MFD 中找到')
    print('=' * 78)

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, x2-x1), max(0, y2-y1)
        inter = iw*ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0

    print(f'\n{"#":<4} {"类型":<10} {"整页框":<24} {"行内最佳IoU":>12} {"状态":<10}')
    print('-' * 70)

    found = 0
    missed = []
    for i, f in enumerate(page_formulas, 1):
        best = 0.0
        for lf in line_formulas:
            v = iou(f['box'], lf['box'])
            if v > best:
                best = v
        ok = best >= 0.5
        found += ok
        if not ok:
            missed.append({'idx': i, 'type': f['type'], 'box': f['box'],
                           'best_iou': best})
        bs = f'[{int(f["box"][0])},{int(f["box"][1])},{int(f["box"][2])},{int(f["box"][3])}]'
        print(f'{i:<4} {f["type"]:<10} {bs:<24} {best:>12.3f} '
              f'{"找到" if ok else "丢失":<10}')

    print('-' * 70)
    print(f'\n  整页 MFD 检出        : {len(page_formulas)}')
    print(f'  行内 MFD 能找到      : {found} ({found/len(page_formulas)*100:.0f}%)')
    print(f'  行内 MFD 丢失        : {len(missed)} ({len(missed)/len(page_formulas)*100:.0f}%)')

    # ---------- 可视化 ----------
    vis_page = img.copy()
    for f in page_formulas:
        b = f['box']
        cv2.rectangle(vis_page, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), (0, 200, 0), 4)
    cv2.putText(vis_page, f'B: Page-level MFD  ({len(page_formulas)} regions)',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
    cv2.imwrite(str(OUT / 'B_page_level.jpg'), vis_page)

    vis_line = img.copy()
    for lf in line_formulas:
        b = lf['box']
        cv2.rectangle(vis_line, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), (0, 165, 255), 4)
    for m in missed:
        b = m['box']
        cv2.rectangle(vis_line, (int(b[0]), int(b[1])),
                      (int(b[2]), int(b[3])), (0, 0, 255), 5)
    cv2.putText(vis_line, f'A: In-line MFD  ({len(line_formulas)} regions, '
                          f'{len(missed)} missed)',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    cv2.imwrite(str(OUT / 'A_line_level.jpg'), vis_line)

    print(f'\n  可视化:')
    print(f'    {OUT / "B_page_level.jpg"}   (绿=整页检出)')
    print(f'    {OUT / "A_line_level.jpg"}   (橙=行内检出, 红=丢失)')

    # 结论
    print('\n' + '=' * 78)
    print('结论')
    print('=' * 78)
    loss = len(missed)/len(page_formulas)*100
    print(f"""
  1. 整页 MFD 检出 {len(page_formulas)} 个公式区域
  2. 行内 MFD 只能找到 {found} 个, 丢失 {len(missed)} 个 ({loss:.0f}%)
  3. 丢失原因: DBNet 行框截断了公式区域, MFD 在残缺的裁剪里
     无法正确检测

  => 原 pipeline「DBNet 切行 -> 行内公式检测」存在级联失效。
     公式检测/识别应当【在整页上独立运行】, 与 DBNet 解耦。
""")

    rep = {'page_mfd': len(page_formulas), 'line_mfd': len(line_formulas),
           'found': found, 'missed': missed,
           'loss_pct': loss}
    rp = OUT / 'coupling.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'报告: {rp}')


if __name__ == '__main__':
    main()
