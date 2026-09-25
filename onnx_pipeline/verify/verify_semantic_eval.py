#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式识别评估 — 语义归一化 + 渲染对比
====================================

发现: 字符级 LaTeX 比对会误判。例如
  GT  : y=\frac{e^{x}}{x^{2}}+\ln 3
  Pred: y={\frac{\mathbf{e}^{x}}{{x}^{2}}}+\mathbf{l}\,\mathbf{n}\ 3
语义完全相同, 但字符级相似度只有 0%。

本脚本:
  1. LaTeX 语义归一化 (剥离 \\mathbf, \\,, {}, 空格 等)
  2. 渲染对比图 (GT vs Pred) 供人工核对
  3. 用归一化后的 token 序列算相似度
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


# ---------------------------------------------------------------------------
# LaTeX 语义归一化
# ---------------------------------------------------------------------------
_STRIP_CMDS = [
    r'\mathbf', r'\mathrm', r'\mathit', r'\mathsf', r'\mathtt', r'\mathcal',
    r'\boldsymbol', r'\text', r'\operatorname', r'\displaystyle',
    r'\textstyle', r'\left', r'\right', r'\smash', r'\overline',
    r'\Bigm', r'\big', r'\Big', r'\bigg', r'\Bigg',
]
_SPACING = [r'\,', r'\;', r'\:', r'\!', r'\ ', r'\quad', r'\qquad', r'\enspace']


def latex_norm(s: str) -> str:
    """把 LaTeX 归一化成语义 token 串。"""
    if not s:
        return ''
    s = s.strip()
    # 去掉 \begin{...}...\end{...} 包裹
    s = re.sub(r'\\begin\{[^}]*\}', '', s)
    s = re.sub(r'\\end\{[^}]*\}', '', s)
    # 去掉纯排版命令
    for c in _STRIP_CMDS:
        s = s.replace(c, '')
    for sp in _SPACING:
        s = s.replace(sp, '')
    # 去掉所有花括号 (只做分组)
    s = s.replace('{', '').replace('}', '')
    # 去掉空白
    s = re.sub(r'\s+', '', s)
    # 统一 \\ln 与 \ln 之类
    s = s.replace('\\\\', '\\')
    # 常见同义: \cdot 与 ·
    s = s.replace(r'\cdot', '*')
    s = s.replace('·', '*')
    s = s.replace(r'\times', '*')
    # \prime 与 '
    s = s.replace(r"^{\prime}", "'")
    s = s.replace(r'\prime', "'")
    # \leq \geq
    s = s.replace(r'\leqslant', r'\leq').replace(r'\geqslant', r'\geq')
    # 去尾部分号/句号
    s = s.rstrip(';,.').rstrip(';')
    return s.lower()


def seq_sim(a: str, b: str) -> float:
    """归一化后的编辑距离相似度"""
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


def render_latex(tex: str, path: Path, size=(900, 140)):
    """用 matplotlib mathtext 渲染 LaTeX。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        fig.patch.set_facecolor('white')
        # 包裹成 $...$
        body = tex.strip()
        if not body.startswith('$'):
            body = f'${body}$'
        try:
            fig.text(0.02, 0.5, body, fontsize=22, va='center')
        except Exception:
            fig.text(0.02, 0.5, tex[:80], fontsize=14, va='center')
        fig.savefig(str(path), facecolor='white', bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        return False


# GT: (标签, GT latex, 原始公式)
GT = [
    ('2(1)',  r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12',      'y = x³ + 7/x⁴ - 2/x + 12'),
    ('2(2)',  r'y=5x^{3}-2^{x}+3e^{x}',                        'y = 5x³ - 2ˣ + 3eˣ'),
    ('2(4)',  r'y=\sin x\cdot\cos x',                          'y = sin x · cos x'),
    ('2(6)',  r'y=3e^{x}\cos x',                               'y = 3eˣcos x'),
    ('2(8)',  r'y=\frac{e^{x}}{x^{2}}+\ln 3',                  'y = eˣ/x² + ln 3'),
    ('2(10)', r's=\frac{1+\sin t}{1+\cos t}',                  's = (1+sin t)/(1+cos t)'),
    ('2(9)',  r'y=x^{2}\ln x\cos x',                           'y = x²ln xcos x'),
    ('2(3)',  r'y=2\tan x+\sec x-1',                           'y = 2tan x + sec x - 1'),
    ('2(5)',  r'y=x^{2}\ln x',                                 'y = x²ln x'),
    ('2(7)',  r'y=\frac{\ln x}{x}',                            'y = ln x / x'),
]


def main():
    print('=' * 78)
    print('公式识别评估 — 语义归一化')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    det = load_mod('formula_detect_onnx',
                   ROOT / 'func/algorithm/formula_detect_onnx.py'
                   ).FormulaDetectorUNNX(use_gpu=False, verbose=False)
    rec = load_mod('formula_recognize_onnx',
                   ROOT / 'func/algorithm/formula_recognize_onnx.py'
                   ).FormulaRecognizerUNNX(use_gpu=False, verbose=False)

    formulas = det(img)
    heights = sorted(f['box'][3]-f['box'][1] for f in formulas)
    med = heights[len(heights)//2]
    normal = [f for f in formulas if (f['box'][3]-f['box'][1]) <= med*2.5]

    print(f'\n检测 {len(formulas)} 个区域 (合理尺寸 {len(normal)} 个)')

    # 手工标定的精确 GT 位置 (基于查看检测框后确定)
    # 格式: 标签 -> (检测框索引, 说明)
    print('\n' + '=' * 78)
    print('识别结果 (语义归一化评分)')
    print('=' * 78)
    print(f'\n{"标签":<6} {"GT 公式":<24} {"UniMERNet 输出":<46} {"字符":>6} {"语义":>6}')
    print('-' * 96)

    results = []
    for tag, gt_latex, orig in GT:
        # 找与该标签对应的检测框: 用 GT latex 与所有框的识别结果比对,
        # 取语义相似度最高的 (这样能定位到正确区域)
        best = None
        for f in normal:
            x1, y1, x2, y2 = [int(v) for v in f['box']]
            m = 4
            crop = img[max(0, y1-m):min(H, y2+m), max(0, x1-m):min(W, x2+m)]
            if crop.size == 0:
                continue
            if not hasattr(main, '_cache'):
                main._cache = {}
            ck = (x1, y1, x2, y2)
            if ck not in main._cache:
                main._cache[ck] = rec(crop)
            latex = main._cache[ck]
            s = seq_sim(latex, gt_latex)
            if best is None or s > best[0]:
                best = (s, latex, f['box'], crop)

        if best is None:
            continue
        sem, latex, box, crop = best

        # 字符级 (旧指标, 用于对比)
        def char_sim(a, b):
            a2, b2 = a.replace(' ', '').lower(), b.replace(' ', '').lower()
            m2, n2 = len(a2), len(b2)
            dp = list(range(n2+1))
            for i in range(1, m2+1):
                prev, dp[0] = dp[0], i
                for j in range(1, n2+1):
                    cur = dp[j]
                    dp[j] = min(dp[j]+1, dp[j-1]+1,
                                prev + (0 if a2[i-1] == b2[j-1] else 1))
                    prev = cur
            return max(0.0, 1.0 - dp[n2]/max(1, n2))

        cs = char_sim(latex, gt_latex)
        od = (orig[:22]+'..') if len(orig) > 24 else orig
        ld = (latex[:44]+'..') if len(latex) > 46 else latex
        print(f'{tag:<6} {od:<24} {ld:<46} {cs*100:5.1f}% {sem*100:5.1f}%')

        # 保存裁剪 + 渲染对比
        cv2.imwrite(str(OUT / f'{tag}_crop.jpg'), crop)
        render_latex(gt_latex, OUT / f'{tag}_gt.png')
        render_latex(latex, OUT / f'{tag}_pred.png')

        results.append({'tag': tag, 'orig': orig, 'gt': gt_latex,
                        'pred': latex, 'char_sim': cs, 'sem_sim': sem,
                        'box': box})

    if results:
        cs_all = [r['char_sim'] for r in results]
        ss_all = [r['sem_sim'] for r in results]
        print('-' * 96)
        print(f'\n字符级平均相似度: {np.mean(cs_all)*100:.1f}%')
        print(f'语义级平均相似度: {np.mean(ss_all)*100:.1f}%')
        print(f'\n语义 >=90%: {sum(1 for a in ss_all if a>=0.9)}/{len(ss_all)}')
        print(f'语义 >=80%: {sum(1 for a in ss_all if a>=0.8)}/{len(ss_all)}')
        print(f'语义 >=70%: {sum(1 for a in ss_all if a>=0.7)}/{len(ss_all)}')
        print(f'语义 <50% : {sum(1 for a in ss_all if a<0.5)}/{len(ss_all)}')

    print(f'\n渲染对比图: {OUT}')

    with open(OUT / 'semantic_eval.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
