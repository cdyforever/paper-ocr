#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进切分: 连通域 + 行聚类
=========================

问题: 简单水平投影会把分式的分子/分母切开。
改进: 基于连通域 + 行聚类, 保留分式完整性。
"""

import os
import re
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'accuracy'
OUT.mkdir(parents=True, exist_ok=True)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_STRIP = [r'\mathbf', r'\mathrm', r'\mathit', r'\mathsf', r'\mathtt',
          r'\mathcal', r'\boldsymbol', r'\text', r'\operatorname',
          r'\displaystyle', r'\textstyle', r'\left', r'\right', r'\smash',
          r'\overline', r'\Bigm', r'\big', r'\Big']
_SP = [r'\,', r'\;', r'\:', r'\!', r'\ ', r'\quad', r'\qquad']


def latex_norm(s):
    if not s:
        return ''
    s = s.strip()
    s = re.sub(r'\\begin\{[^}]*\}', '', s)
    s = re.sub(r'\\end\{[^}]*\}', '', s)
    for c in _STRIP:
        s = s.replace(c, '')
    for sp in _SP:
        s = s.replace(sp, '')
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\s+', '', s)
    s = s.replace(r'\cdot', '*').replace('·', '*').replace(r'\times', '*')
    s = s.replace(r"^{\prime}", "'").replace(r'\prime', "'")
    s = s.rstrip(';,.').rstrip(';')
    return s.lower()


def sim(a, b):
    a, b = latex_norm(a), latex_norm(b)
    if not b:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (0 if a[i-1] == b[j-1] else 1))
            prev = cur
    return max(0.0, 1.0 - dp[n] / len(b))


def split_by_components(img, box, gap_ratio=0.45):
    """
    基于连通域 + 行聚类切分。
    gap_ratio: 行间距 > gap_ratio * 中位字符高 则分行
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return [box]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 12:
            continue
        if ch > roi.shape[0] * 0.9:
            continue
        comps.append((int(cx), int(cy), int(cw), int(ch), int(area), float(cents[i][1])))

    if not comps:
        return [box]

    heights = sorted(c[3] for c in comps)
    med_h = heights[len(heights)//2]

    comps.sort(key=lambda c: c[5])

    lines = []
    cur = [comps[0]]
    for c in comps[1:]:
        cur_top = min(x[1] for x in cur)
        cur_bot = max(x[1] + x[3] for x in cur)
        c_top, c_bot = c[1], c[1] + c[3]
        overlap = min(cur_bot, c_bot) - max(cur_top, c_top)
        gap = max(c_top - cur_bot, cur_top - c_bot)
        if overlap > 0 or gap < med_h * gap_ratio:
            cur.append(c)
        else:
            lines.append(cur)
            cur = [c]
    lines.append(cur)

    out = []
    for ln in lines:
        top = min(c[1] for c in ln)
        bot = max(c[1] + c[3] for c in ln)
        if bot - top < 15:
            continue
        pad = 5
        out.append([x1, max(0, y1 + top - pad), x2, min(H, y1 + bot + pad)])
    return out if out else [box]


GT4 = [
    ('2(1)', r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12', 'y = x3 + 7/x4 - 2/x + 12'),
    ('2(3)', r'y=2\tan x+\sec x-1', 'y = 2tan x + sec x - 1'),
    ('2(5)', r'y=x^{2}\ln x', 'y = x2ln x'),
    ('2(7)', r'y=\frac{\ln x}{x}', 'y = ln x / x'),
]


def main():
    print('=' * 78)
    print('Improved splitting (connected components + line clustering)')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    rec = load_mod('formula_recognize_onnx',
                   ROOT / 'func/algorithm/formula_recognize_onnx.py'
                   ).FormulaRecognizerONNX(use_gpu=False, verbose=False)

    big = [340, 1008, 1077, 1765]
    print(f'\nOversized box: {big}  height={big[3]-big[1]}px')

    subs = split_by_components(img, big)
    print(f'\nSplit into {len(subs)} sub-regions:')
    for i, s in enumerate(subs, 1):
        print(f'  #{i} {s}  h={s[3]-s[1]}px')

    print(f'\n{"GT":<24} {"Prediction":<44} {"Sim":>7}')
    print('-' * 80)

    results = []
    for i, (tag, gt, disp) in enumerate(GT4):
        if i >= len(subs):
            print(f'{disp:<24} {"(no match)":<44} {"--":>7}')
            continue
        s = subs[i]
        crop = img[s[1]:s[3], s[0]:s[2]]
        cv2.imwrite(str(OUT / f'{tag}.jpg'), crop)
        latex = rec(crop)
        a = sim(latex, gt)
        dd = (disp[:22]+'..') if len(disp) > 24 else disp
        ld = (latex[:42]+'..') if len(latex) > 44 else latex
        print(f'{dd:<24} {ld:<44} {a*100:6.1f}%')
        results.append({'tag': tag, 'gt': gt, 'disp': disp,
                        'latex': latex, 'sim': a,
                        'box': [int(v) for v in s]})

    if results:
        accs = [r['sim'] for r in results]
        print('-' * 80)
        print(f'\nMean similarity: {np.mean(accs)*100:.1f}%')
        print(f'  >=90%: {sum(1 for a in accs if a>=0.9)}/{len(accs)}')
        print(f'  >=70%: {sum(1 for a in accs if a>=0.7)}/{len(accs)}')

    with open(OUT / 'box_splitting.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nOutput: {OUT}')


if __name__ == '__main__':
    main()
