#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修正坐标后重跑能力实验
======================

上一版 A6/A7 坐标标错 (框到别的公式)。这里改用
「检测框 + 按识别内容自动匹配」的方式, 确保每个 GT 拿到正确区域。
"""

import os
import re
import sys
import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def render_clean(text, size=(900, 150)):
    fp = r'C:\Windows\Fonts\msyh.ttc'
    if not os.path.exists(fp):
        fp = r'C:\Windows\Fonts\simhei.ttf'
    font = ImageFont.truetype(fp, 58)
    img = Image.new('RGB', size, 'white')
    ImageDraw.Draw(img).text((25, 32), text, fill='black', font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def enhance(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# (标签, 显示文本, GT latex)
ITEMS = [
    ('A1', 'y = x³ + 7/x⁴ - 2/x + 12', r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12'),
    ('A2', 'y = 5x³ - 2ˣ + 3eˣ', r'y=5x^{3}-2^{x}+3e^{x}'),
    ('A3', 'y = sin x · cos x', r'y=\sin x\cdot\cos x'),
    ('A4', 'y = 3eˣcos x', r'y=3e^{x}\cos x'),
    ('A5', 'y = eˣ/x² + ln 3', r'y=\frac{e^{x}}{x^{2}}+\ln 3'),
    ('A6', 's = (1+sin t)/(1+cos t)', r's=\frac{1+\sin t}{1+\cos t}'),
    ('A7', 'y = x²ln xcos x', r'y=x^{2}\ln x\cos x'),
    ('A8', 'y = 2tan x + sec x - 1', r'y=2\tan x+\sec x-1'),
    ('A9', 'y = x²ln x', r'y=x^{2}\ln x'),
    ('A10', 'y = ln x / x', r'y=\frac{\ln x}{x}'),
]


def main():
    print('=' * 78)
    print('能力实验 v2 — 自动匹配正确区域')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    det = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py'
                   ).FormulaDetectorONNX(use_gpu=False, verbose=False)
    rec = load_mod('formula_recognize_onnx',
                   ROOT / 'func/algorithm/formula_recognize_onnx.py'
                   ).FormulaRecognizerONNX(use_gpu=False, verbose=False)

    formulas = det(img)
    heights = sorted(f['box'][3]-f['box'][1] for f in formulas)
    med = heights[len(heights)//2]
    normal = [f for f in formulas if (f['box'][3]-f['box'][1]) <= med*2.5]
    print(f'\n检测 {len(formulas)} 个, 合理 {len(normal)} 个')

    # 预先识别所有合理框
    print('预识别所有区域...')
    crops = {}
    for f in normal:
        x1, y1, x2, y2 = [int(v) for v in f['box']]
        m = 4
        crop = img[max(0, y1-m):min(H, y2+m), max(0, x1-m):min(W, x2+m)]
        if crop.size == 0:
            continue
        key = (x1, y1, x2, y2)
        crops[key] = crop

    rec_cache = {}
    for key, crop in crops.items():
        rec_cache[key] = rec(crop)

    print(f'\n{"标签":<5} {"公式":<24} {"A 干净":>8} {"B 照片":>8} {"C 增强":>8}  {"匹配区域":<22}')
    print('-' * 100)

    rows = []
    for tag, disp, gt in ITEMS:
        # 找语义最接近的区域
        best_key, best_s = None, -1
        for key, latex in rec_cache.items():
            s = sim(latex, gt)
            if s > best_s:
                best_s, best_key = s, key
        # 匹配度太低则认为检测不到
        if best_s < 0.25:
            best_key = None

        # A. 干净渲染
        clean = render_clean(disp)
        cv2.imwrite(str(OUT / f'{tag}_A.jpg'), clean)
        la = rec(clean)
        sa = sim(la, gt)

        if best_key is not None:
            crop = crops[best_key]
            cv2.imwrite(str(OUT / f'{tag}_B.jpg'), crop)
            lb = rec_cache[best_key]
            sb = sim(lb, gt)

            enh = enhance(crop)
            cv2.imwrite(str(OUT / f'{tag}_C.jpg'), enh)
            lc = rec(enh)
            sc = sim(lc, gt)
            bs = f'[{best_key[0]},{best_key[1]},{best_key[2]},{best_key[3]}]'
        else:
            lb = lc = '(未检测到)'
            sb = sc = 0.0
            bs = '(未检测到)'

        dd = (disp[:22]+'..') if len(disp) > 24 else disp
        print(f'{tag:<5} {dd:<24} {sa*100:7.1f}% {sb*100:7.1f}% {sc*100:7.1f}%  {bs:<22}')

        rows.append({'tag': tag, 'disp': disp, 'gt': gt,
                     'A': {'latex': la, 'sim': sa},
                     'B': {'latex': lb, 'sim': sb},
                     'C': {'latex': lc, 'sim': sc},
                     'box': list(best_key) if best_key else None})

    sa_all = [r['A']['sim'] for r in rows]
    sb_all = [r['B']['sim'] for r in rows]
    sc_all = [r['C']['sim'] for r in rows]

    print('-' * 100)
    print(f'{"平均":<5} {"":<24} {np.mean(sa_all)*100:7.1f}% '
          f'{np.mean(sb_all)*100:7.1f}% {np.mean(sc_all)*100:7.1f}%')

    print(f'\n{"指标":<18} {"A 干净":>10} {"B 照片":>10} {"C 增强":>10}')
    print('-' * 52)
    for label, thr in [('>=95% (完全正确)', 0.95), ('>=80% (达标)', 0.8),
                       ('>=60% (可用)', 0.6), ('<40% (失败)', 0.4)]:
        cells = []
        for arr in (sa_all, sb_all, sc_all):
            n = sum(1 for a in arr if (a < 0.4 if thr == 0.4 else a >= thr))
            cells.append(f'{n}/{len(arr)}')
        print(f'{label:<18} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}')

    with open(OUT / 'clean_vs_photo.json', 'w', encoding='utf-8') as f:
        json.dump({'avg': {'A': float(np.mean(sa_all)),
                           'B': float(np.mean(sb_all)),
                           'C': float(np.mean(sc_all))},
                   'details': rows}, f, ensure_ascii=False, indent=2)
    print(f'\n输出: {OUT}')


if __name__ == '__main__':
    main()
