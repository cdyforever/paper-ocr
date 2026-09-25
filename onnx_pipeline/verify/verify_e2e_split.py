#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端验证: 检测(含切分修复) + 识别
===================================

不使用手工标定的框, 完全依赖检测器输出, 验证切分修复的实际效果。

重点对比: 修复前会合并成 1 个大框的 4 道题, 现在能否正确识别。
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


# 那 4 道曾被合并的题 (GT)
MERGED_GT = [
    ('2(1)', r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12', 'y = x3 + 7/x4 - 2/x + 12'),
    ('2(3)', r'y=2\tan x+\sec x-1', 'y = 2tan x + sec x - 1'),
    ('2(5)', r'y=x^{2}\ln x', 'y = x2 ln x'),
    ('2(7)', r'y=\frac{\ln x}{x}', 'y = ln x / x'),
]

# 大框的 y 范围 (用于定位切分出的子框)
MERGED_Y_RANGE = (1008, 1770)


def main():
    print('=' * 78)
    print('端到端验证: 检测(含切分修复) + 识别')
    print('=' * 78)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    fd_mod = load_mod('formula_detect_onnx',
                      ROOT / 'func/algorithm/formula_detect_onnx.py')
    rec_mod = load_mod('formula_recognize_onnx',
                       ROOT / 'func/algorithm/formula_recognize_onnx.py')
    rec = rec_mod.FormulaRecognizerONNX(use_gpu=False, verbose=False)

    # ---------- 对比: 开/关切分 ----------
    print('\n' + '=' * 78)
    print('对比实验: split_tall_boxes = False vs True')
    print('=' * 78)

    det_off = fd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False,
                                         split_tall_boxes=False)
    det_on = fd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False,
                                        split_tall_boxes=True)

    res_off = det_off(img)
    res_on, dbg_on = det_on(img, return_debug=True)

    print(f'\n  关闭切分: {len(res_off)} 个框')
    print(f'  开启切分: {len(res_on)} 个框')
    print(f'  切分记录: {len(dbg_on["split_records"])} 个')

    # ---------- 定位被合并的区域 ----------
    print('\n' + '=' * 78)
    print('曾被合并的 4 道题 (y 1008-1770)')
    print('=' * 78)

    y1, y2 = MERGED_Y_RANGE

    def boxes_in_range(results):
        out = []
        for r in results:
            cy = (r['box'][1] + r['box'][3]) / 2
            if y1 <= cy <= y2:
                out.append(r)
        return sorted(out, key=lambda r: r['box'][1])

    off_in = boxes_in_range(res_off)
    on_in = boxes_in_range(res_on)

    print(f'\n  [关闭切分] 该区域 {len(off_in)} 个框:')
    for r in off_in:
        h = r['box'][3] - r['box'][1]
        print(f"    {[int(v) for v in r['box']]}  高={h:.0f}px  {r['type']}")

    print(f'\n  [开启切分] 该区域 {len(on_in)} 个框:')
    for r in on_in:
        h = r['box'][3] - r['box'][1]
        print(f"    {[int(v) for v in r['box']]}  高={h:.0f}px  {r['type']}")

    # ---------- 识别对比 ----------
    print('\n' + '=' * 78)
    print('识别效果对比')
    print('=' * 78)

    def recognize_region(results):
        """识别指定区域内的所有框, 返回 (latex, sim) 列表"""
        out = []
        for r in boxes_in_range(results):
            bx1, by1, bx2, by2 = [int(v) for v in r['box']]
            crop = img[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)]
            if crop.size == 0:
                continue
            latex = rec(crop)
            # 与该区域任一 GT 求最佳相似度
            best = max(sim(latex, gt) for _, gt, _ in MERGED_GT)
            out.append({'box': r['box'], 'latex': latex, 'best_sim': best,
                        'height': by2 - by1})
        return out

    print('\n--- 关闭切分 ---')
    off_rec = recognize_region(res_off)
    for i, r in enumerate(off_rec, 1):
        ld = (r['latex'][:56]+'..') if len(r['latex']) > 58 else r['latex']
        print(f"  [{i}] 高={r['height']:3.0f}px  最佳匹配={r['best_sim']*100:5.1f}%")
        print(f"      {ld}")

    print('\n--- 开启切分 ---')
    on_rec = recognize_region(res_on)
    for i, r in enumerate(on_rec, 1):
        ld = (r['latex'][:56]+'..') if len(r['latex']) > 58 else r['latex']
        print(f"  [{i}] 高={r['height']:3.0f}px  最佳匹配={r['best_sim']*100:5.1f}%")
        print(f"      {ld}")

    # ---------- 汇总 ----------
    print('\n' + '=' * 78)
    print('汇总')
    print('=' * 78)

    off_avg = np.mean([r['best_sim'] for r in off_rec]) if off_rec else 0
    on_avg = np.mean([r['best_sim'] for r in on_rec]) if on_rec else 0
    off_ok = sum(1 for r in off_rec if r['best_sim'] >= 0.8)
    on_ok = sum(1 for r in on_rec if r['best_sim'] >= 0.8)

    print(f'\n  {"指标":<24} {"关闭切分":>12} {"开启切分":>12}')
    print('  ' + '-' * 50)
    print(f'  {"检测框数":<24} {len(off_in):>12} {len(on_in):>12}')
    print(f'  {"平均最佳相似度":<22} {off_avg*100:>11.1f}% {on_avg*100:>11.1f}%')
    print(f'  {"达标数 (>=80%)":<22} {off_ok:>12} {on_ok:>12}')
    print(f'  {"4 道题覆盖":<24} '
          f'{len(off_in):>11}/4 {len(on_in):>11}/4')

    # 逐题核对
    print('\n--- 逐题核对 (开启切分) ---')
    matched = 0
    for tag, gt, disp in MERGED_GT:
        best_s, best_l = 0.0, ''
        for r in on_rec:
            s = sim(r['latex'], gt)
            if s > best_s:
                best_s, best_l = s, r['latex']
        ok = best_s >= 0.8
        matched += ok
        dd = (disp[:22]+'..') if len(disp) > 24 else disp
        ld = (best_l[:40]+'..') if len(best_l) > 42 else best_l
        print(f"  {'OK ' if ok else 'BAD'} [{tag}] {dd:<24} {best_s*100:5.1f}%  {ld}")

    print(f'\n  4 道题正确识别: {matched}/4')

    rep = {
        'split_off': {'n_boxes': len(off_in), 'avg_sim': float(off_avg),
                      'n_ok': off_ok, 'details': off_rec},
        'split_on': {'n_boxes': len(on_in), 'avg_sim': float(on_avg),
                     'n_ok': on_ok, 'details': on_rec},
        'per_question': matched,
    }
    rp = OUT / 'e2e_split_verify.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'\n报告: {rp}')


if __name__ == '__main__':
    main()
