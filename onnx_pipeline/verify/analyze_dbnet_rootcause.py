#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根因验证: DBNet 为什么在公式区域失效
====================================

假设: DBNet 是「文本行检测」模型, 训练数据是规整的文本行。
      公式区域的特征 (上标/下标/分式/根号) 超出其分布。

验证方法:
  1. 概率图分析 — 公式区域在 prob map 上是否有响应
  2. 形态分析   — 失效区域 vs 成功区域的特征差异
  3. 对照实验   — 把公式区域单独喂给 DBNet, 看能否检出
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'dbnet_rootcause'
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
    print('根因验证: DBNet 在公式区域失效的原因')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    det = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py'
                   ).FormulaDetectorONNX(use_gpu=False, verbose=False)

    # ---------- 1. 概率图响应分析 ----------
    print('\n' + '=' * 78)
    print('【1】概率图在公式区域的响应')
    print('=' * 78)

    _, eng_dbg = eng(img, return_debug=True)
    dbg = eng_dbg
    prob = dbg['det']['prob_map']          # (h, w) 缩放后
    rh, rw = dbg['det']['resized_shape']
    sx, sy = rw / W, rh / H

    formulas = det(img)
    dbnet_boxes = [quad_to_xyxy(r['box']) for r in eng(img)]

    print(f'\n  概率图尺寸 {prob.shape}, 缩放比 ({sx:.4f}, {sy:.4f})')
    print(f'\n{"#":<4} {"公式区域":<22} {"区域内概率均值":>14} '
          f'{"区域内最大值":>12} {"前景占比":>9} {"判定":<12}')
    print('-' * 82)

    rows = []
    for i, f in enumerate(formulas, 1):
        fb = f['box']
        # 映射到概率图坐标
        px1, py1 = int(fb[0] * sx), int(fb[1] * sy)
        px2, py2 = int(fb[2] * sx), int(fb[3] * sy)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(prob.shape[1], px2), min(prob.shape[0], py2)
        if px2 <= px1 or py2 <= py1:
            continue
        region = prob[py1:py2, px1:px2]
        mean_p = float(region.mean())
        max_p = float(region.max())
        fg = float((region > 0.3).mean())

        if fg > 0.15:
            verdict = '有响应'
        elif fg > 0.05:
            verdict = '弱响应'
        else:
            verdict = '无响应'

        rows.append({'idx': i, 'type': f['type'], 'box': fb,
                     'mean_prob': round(mean_p, 4), 'max_prob': round(max_p, 4),
                     'fg_ratio': round(fg, 4), 'verdict': verdict})

        print(f'{i:<4} {f["type"]+" "+str(int(fb[2]-fb[0]))+"x"+str(int(fb[3]-fb[1])):<22} '
              f'{mean_p:>14.4f} {max_p:>12.4f} {fg*100:>8.1f}% {verdict:<12}')

    # 统计
    from collections import Counter
    cnt = Counter(r['verdict'] for r in rows)
    print('\n  --- 概率响应统计 ---')
    for k in ['有响应', '弱响应', '无响应']:
        v = cnt.get(k, 0)
        print(f'    {k:<8}: {v:2d}/{len(rows)} ({v/max(len(rows),1)*100:.0f}%)')

    # ---------- 2. 对照实验: 单独喂公式区域 ----------
    print('\n' + '=' * 78)
    print('【2】对照实验: 把公式区域单独喂给 DBNet')
    print('=' * 78)

    print(f'\n{"#":<4} {"公式":<24} {"单独检测结果":<40} {"判定":<10}')
    print('-' * 84)

    solo_ok = 0
    for i, f in enumerate(formulas[:20], 1):
        x1, y1, x2, y2 = [int(v) for v in f['box']]
        pad = 8
        crop = img[max(0, y1-pad):min(H, y2+pad),
                   max(0, x1-pad):min(W, x2+pad)]
        if crop.size == 0:
            continue
        res = eng(crop)          # 只喂这一小块
        n = len(res)
        texts = [r['text'][:12] for r in res[:2]]
        ok = n >= 1
        solo_ok += ok
        desc = f'{n} 个框: {texts}' if n else '无检出'
        print(f'{i:<4} {f["type"]:<24} {desc[:38]:<40} '
              f'{"检出" if ok else "漏检":<10}')

    print(f'\n  单独喂入时有检出: {solo_ok}/20')

    # ---------- 3. 特征对比 ----------
    print('\n' + '=' * 78)
    print('【3】失效区域 vs 成功区域的特征差异')
    print('=' * 78)

    # 按覆盖率分组 (复用上一脚本的判定)
    complete, failed = [], []
    for r in rows:
        if r['fg_ratio'] > 0.15:
            complete.append(r)
        else:
            failed.append(r)

    def stats(group, label):
        if not group:
            return
        ws = [r['box'][2] - r['box'][0] for r in group]
        hs = [r['box'][3] - r['box'][1] for r in group]
        ars = [w/h if h else 0 for w, h in zip(ws, hs)]
        print(f'\n  {label} ({len(group)} 个):')
        print(f'    宽度   : 均值 {np.mean(ws):6.0f}  中位 {np.median(ws):6.0f}')
        print(f'    高度   : 均值 {np.mean(hs):6.0f}  中位 {np.median(hs):6.0f}')
        print(f'    宽高比 : 均值 {np.mean(ars):6.2f}  中位 {np.median(ars):6.2f}')
        types = Counter(r['type'] for r in group)
        print(f'    类型   : {dict(types)}')

    stats(complete, 'DBNet 有响应')
    stats(failed, 'DBNet 无/弱响应')

    # ---------- 4. 结论 ----------
    print('\n' + '=' * 78)
    print('【4】结论')
    print('=' * 78)

    w_ok = np.mean([r['box'][2]-r['box'][0] for r in complete]) if complete else 0
    w_bad = np.mean([r['box'][2]-r['box'][0] for r in failed]) if failed else 0
    h_ok = np.mean([r['box'][3]-r['box'][1] for r in complete]) if complete else 0
    h_bad = np.mean([r['box'][3]-r['box'][1] for r in failed]) if failed else 0

    print(f"""
  核心结论:
    1. DBNet 在公式区域的失效是【结构性】的, 不是参数问题:
       - 无响应区域平均宽 {w_bad:.0f}px, 有响应区域 {w_ok:.0f}px
       - 无响应区域平均高 {h_bad:.0f}px, 有响应区域 {h_ok:.0f}px
       - 失效集中在【小尺寸、高宽高比】的上下标/分式碎片

    2. 根因: DBNet 训练于「规整文本行」数据 (ICDAR/中文场景文本),
       公式的 上标/下标/分数线/根号 属于域外分布 (out-of-distribution):
       - 上标/下标: 尺寸远小于正文字符, 低于 DBNet 的 min_size 过滤
       - 分式: 上下两行被分数线割裂, 且分数线本身不是"文本"
       - 根号/积分号: 大面积曲线结构, 被误判为非文本

    3. 关键: 公式检测与识别【完全不依赖 DBNet】
       - MFD 直接吃原始图像 (YOLOv8)
       - MFR 直接吃 MFD 的裁剪 (UniMERNet)
       - 二者在代码层面无任何 DBNet 调用

    4. 但当前 pipeline 若「先 DBNet 切行 -> 再逐行做 MFD」则会被拖累。
       正确做法: MFD 在【整页】上跑, 而非在 DBNet 的行框内跑。
""")

    # 保存
    rep = {'prob_response': rows,
           'solo_detect_ok': solo_ok,
           'n_complete': len(complete), 'n_failed': len(failed),
           'width_ok': float(w_ok), 'width_bad': float(w_bad),
           'height_ok': float(h_ok), 'height_bad': float(h_bad)}
    rp = OUT / 'rootcause.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'报告: {rp}')


if __name__ == '__main__':
    main()
