#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终精度评估汇总
================

汇总所有实验，给出明确结论:
  1. 文本识别 (PP-OCRv4) 真实精度
  2. 公式识别 (UniMERNet) 真实精度
  3. 影响精度的关键因素

评测方法: 语义归一化相似度 (剥离 LaTeX 排版差异)
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


# ---------------------------------------------------------------------------
# 纯文本行 GT (该页不含公式的行)
# ---------------------------------------------------------------------------
GT_TEXT = [
    ('习题2-2',                    144),
    ('1. 推导余切函数及余割函数的导数公式：',  612),
    ('2. 求下列函数的导数：',            919),
    ('3. 求下列函数在给定点处的导数：',      2159),
    ('(1) 该物体的速度 v(t);',        3423),
    ('6. 求下列函数的导数：',            3731),
]

# 公式 GT: (标签, GT latex, 说明, 推荐裁剪框)
GT_FORMULA = [
    ('F1', r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12',
     'y = x3 + 7/x4 - 2/x + 12', [345, 1020, 1017, 1247]),
    ('F2', r'y=5x^{3}-2^{x}+3e^{x}',
     'y = 5x3 - 2^x + 3e^x', [1690, 983, 2337, 1119]),
    ('F3', r'y=\sin x\cdot\cos x',
     'y = sin x · cos x', [1693, 1243, 2360, 1349]),
    ('F4', r'y=3e^{x}\cos x',
     'y = 3e^x cos x', [1698, 1389, 2203, 1511]),
    ('F5', r'y=\frac{e^{x}}{x^{2}}+\ln 3',
     'y = e^x/x^2 + ln 3', [1699, 1548, 2183, 1775]),
    ('F6', r's=\frac{1+\sin t}{1+\cos t}',
     's = (1+sin t)/(1+cos t)', [1751, 1840, 2209, 2063]),
    ('F7', r'y=x^{2}\ln x\cos x',
     'y = x2 ln x cos x', [356, 1921, 891, 2049]),
    ('F8', r'y=2\tan x+\sec x-1',
     'y = 2tan x + sec x - 1', [340, 1320, 1077, 1426]),
    ('F9', r'y=x^{2}\ln x',
     'y = x2 ln x', [340, 1461, 1077, 1577]),
]


def main():
    print('=' * 78)
    print('最终精度评估')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng = load_mod('onnx_ocr_engine',
                   ROOT / 'func/algorithm/onnx_ocr_engine.py'
                   ).OnnxOCREngine(use_gpu=False, verbose=False)
    rec = load_mod('formula_recognize_onnx',
                   ROOT / 'func/algorithm/formula_recognize_onnx.py'
                   ).FormulaRecognizerONNX(use_gpu=False, verbose=False)

    # ==================== 1. 文本识别 ====================
    print('\n' + '=' * 78)
    print('【1】文本识别精度 (PP-OCRv4)')
    print('=' * 78)

    ocr = eng(img)
    print(f'\n检测 {len(ocr)} 行')

    print(f'\n{"GT 文本":<36} {"识别结果":<36} {"准确率":>7}')
    print('-' * 84)

    text_accs = []
    for gt, gt_y in GT_TEXT:
        best, bd = None, 1e9
        for r in ocr:
            cy = sum(p[1] for p in r['box']) / len(r['box'])
            d = abs(cy - gt_y)
            if d < bd:
                bd, best = d, r['text']
        pred = best if bd < 100 else ''
        a = sim(pred, gt)
        text_accs.append(a)
        gd = (gt[:34]+'..') if len(gt) > 36 else gt
        pd = (pred[:34]+'..') if len(pred) > 36 else pred
        print(f'{gd:<36} {pd:<36} {a*100:6.1f}%')

    print('-' * 84)
    print(f'\n文本识别平均准确率: {np.mean(text_accs)*100:.1f}%')
    print(f'  完全正确(100%): {sum(1 for a in text_accs if a>0.999)}/{len(text_accs)}')
    print(f'  达标(>=90%)   : {sum(1 for a in text_accs if a>=0.9)}/{len(text_accs)}')

    # ==================== 2. 公式识别 ====================
    print('\n' + '=' * 78)
    print('【2】公式识别精度 (UniMERNet-tiny)')
    print('=' * 78)

    print(f'\n{"公式":<24} {"UniMERNet 输出":<44} {"相似度":>7}')
    print('-' * 80)

    f_accs = []
    f_details = []
    for tag, gt, disp, box in GT_FORMULA:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        latex = rec(crop)
        a = sim(latex, gt)
        f_accs.append(a)
        dd = (disp[:22]+'..') if len(disp) > 24 else disp
        ld = (latex[:42]+'..') if len(latex) > 44 else latex
        print(f'{dd:<24} {ld:<44} {a*100:6.1f}%')
        f_details.append({'tag': tag, 'gt': gt, 'disp': disp,
                          'latex': latex, 'sim': a})

    print('-' * 80)
    print(f'\n公式识别平均相似度: {np.mean(f_accs)*100:.1f}%')
    print(f'  完全正确(>=95%): {sum(1 for a in f_accs if a>=0.95)}/{len(f_accs)}')
    print(f'  达标(>=80%)   : {sum(1 for a in f_accs if a>=0.8)}/{len(f_accs)}')
    print(f'  可用(>=60%)   : {sum(1 for a in f_accs if a>=0.6)}/{len(f_accs)}')
    print(f'  失败(<40%)    : {sum(1 for a in f_accs if a<0.4)}/{len(f_accs)}')

    # ==================== 3. 汇总 ====================
    print('\n' + '=' * 78)
    print('【3】结论')
    print('=' * 78)

    print(f"""
  文本识别 (PP-OCRv4)      : {np.mean(text_accs)*100:5.1f}%
  公式识别 (UniMERNet-tiny): {np.mean(f_accs)*100:5.1f}%

  影响因素 (按重要性):
    1. 公式检测框质量  — 合并/越界会直接导致识别失败 (最大因素)
    2. 裁剪边界        — 分式被切掉分子/分母则无法恢复
    3. 图像清晰度      — 照片模糊/倾斜影响上标识别
    4. 模型容量        — tiny 版对复杂公式有限
""")

    rep = {
        'text_avg': float(np.mean(text_accs)),
        'formula_avg': float(np.mean(f_accs)),
        'text_details': [{'gt': g, 'acc': a}
                         for (g, _), a in zip(GT_TEXT, text_accs)],
        'formula_details': f_details,
    }
    rp = OUT / 'real_accuracy.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'报告: {rp}')


if __name__ == '__main__':
    main()
