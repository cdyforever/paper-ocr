#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证: 页面级解耦流水线 (公式检测 -> 涂白 -> 文本检测)
=====================================================

对比三种流程在公式密集页面上的表现:

  A 旧流程 (耦合):   DBNet 切行 -> 行内 MFD -> 行内 MFR
  B 仅放大 DBNet:     DBNet(1920) -> 行内 MFD
  C 新流程 (解耦):   页面级 MFD -> 涂白 -> DBNet -> 文本/MFR

关键指标:
  1. 公式区域被完整送入 MFR 的比例 (覆盖度)
  2. 文本行是否干净 (不含公式碎片)
  3. 公式识别的语义相似度 (完整裁剪 vs 碎片裁剪)

用法:
    python onnx_pipeline/verify/verify_page_decoupling.py
"""

import os
import re
import sys
import json
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np

import os as _os
from pathlib import Path as _Path


def _chdir_project_root():
    _p = _Path(__file__).resolve()
    for _parent in [_p] + list(_p.parents):
        if (_parent / 'weights').is_dir() and (_parent / 'func').is_dir():
            _os.chdir(_parent)
            return _parent
    return _Path.cwd()


_chdir_project_root()

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'decoupling'
OUT.mkdir(parents=True, exist_ok=True)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------- LaTeX 语义归一化 ---------------------------
_STRIP = [r'\mathbf', r'\mathrm', r'\mathit', r'\mathsf', r'\mathtt',
          r'\mathcal', r'\boldsymbol', r'\text', r'\operatorname',
          r'\displaystyle', r'\textstyle', r'\left', r'\right', r'\smash',
          r'\overline', r'\big', r'\Big', r'\bigl', r'\bigr', r'\limits']
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
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return max(0.0, 1.0 - dp[n] / len(b))


def quad_to_xyxy(q):
    q = np.array(q, dtype=np.float32).reshape(-1, 2)
    return [float(q[:, 0].min()), float(q[:, 1].min()),
            float(q[:, 0].max()), float(q[:, 1].max())]


def inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def coverage(fbox, boxes):
    """fbox 被 boxes 覆盖的宽度/高度比例 (并集投影)。"""
    fa = (fbox[2] - fbox[0]) * (fbox[3] - fbox[1])
    if fa <= 0:
        return 0.0, 0.0
    inter = sum(inter_area(fbox, b) for b in boxes)
    area_ratio = min(1.0, inter / fa)

    # 投影覆盖 (更贴近"公式是否被完整框住")
    W = int(fbox[2] - fbox[0])
    H = int(fbox[3] - fbox[1])
    if W <= 0 or H <= 0:
        return 0.0, 0.0
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1 = int(max(b[0], fbox[0]) - fbox[0]); x2 = int(min(b[2], fbox[2]) - fbox[0])
        y1 = int(max(b[1], fbox[1]) - fbox[1]); y2 = int(min(b[3], fbox[3]) - fbox[1])
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    col = mask.max(axis=0).mean()
    row = mask.max(axis=1).mean()
    return float(col), float(row)


# --------------------------- 4 道题的 GT ---------------------------
GT_FORMULAS = [
    (r'y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12', 'y = x3 + 7/x4 - 2/x + 12'),
    (r'y=2\tan x+\sec x-1', 'y = 2tan x + sec x - 1'),
    (r'y=x^{2}\ln x', 'y = x2 ln x'),
    (r'y=\frac{\ln x}{x}', 'y = ln x / x'),
    (r'y=5x^{3}-2^{x}+3e^{x}', 'y = 5x3 - 2^x + 3e^x'),
    (r'y=\sin x\cdot\cos x', 'y = sin x cos x'),
    (r'y=3e^{x}\cos x', 'y = 3ex cos x'),
    (r'y=\frac{e^{x}}{x^{2}}+\ln 3', 'y = ex/x2 + ln3'),
    (r'y=x^{2}\ln x\cos x', 'y = x2 ln x cos x'),
    (r's=\frac{1+\sin t}{1+\cos t}', 's = (1+sin t)/(1+cos t)'),
]


def best_gt(latex):
    best, tag = 0.0, ''
    for gt, disp in GT_FORMULAS:
        s = sim(latex, gt)
        if s > best:
            best, tag = s, disp
    return best, tag


def main():
    print('=' * 80)
    print('页面级解耦流水线验证')
    print('=' * 80)

    img = cv2.imread(str(ROOT / 'test.jpg'))
    H, W = img.shape[:2]

    eng_mod = load_mod('onnx_ocr_engine', ROOT / 'func/algorithm/onnx_ocr_engine.py')
    mfd_mod = load_mod('formula_detect_onnx', ROOT / 'func/algorithm/formula_detect_onnx.py')
    mfr_mod = load_mod('formula_recognize_onnx',
                       ROOT / 'func/algorithm/formula_recognize_onnx.py')
    pipe_mod = load_mod('onnx_page_pipeline',
                        ROOT / 'func/algorithm/onnx_page_pipeline.py')

    mfd = mfd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False)
    formulas = mfd(img)
    fboxes = [f['box'] for f in formulas]
    print(f'\n页面级 MFD 检测到公式区域: {len(formulas)} 个')

    # ================= A. 旧流程: DBNet 行框覆盖度 =================
    print('\n' + '=' * 80)
    print('A. 旧流程 (耦合): DBNet 行框对公式区域的覆盖度')
    print('=' * 80)

    eng = eng_mod.OnnxOCREngine(use_gpu=False, verbose=False)
    lines = eng(img)
    line_boxes = [quad_to_xyxy(r['box']) for r in lines]
    print(f'  DBNet 行数: {len(lines)}')

    cov_a = [coverage(f['box'], line_boxes) for f in formulas]
    w_a = float(np.mean([c[0] for c in cov_a]))
    h_a = float(np.mean([c[1] for c in cov_a]))
    ok_a = sum(1 for c in cov_a if c[0] >= 0.9 and c[1] >= 0.9)
    print(f'  宽度覆盖 {w_a*100:.1f}%   高度覆盖 {h_a*100:.1f}%   '
          f'完整 {ok_a}/{len(formulas)}')

    # ================= B. 放大 DBNet 输入 =================
    print('\n' + '=' * 80)
    print('B. 仅放大 DBNet 输入 (960 -> 1920)')
    print('=' * 80)

    eng_big = eng_mod.OnnxOCREngine(use_gpu=False, verbose=False,
                                    det_limit_side_len=1920)
    lines_big = eng_big(img)
    line_boxes_big = [quad_to_xyxy(r['box']) for r in lines_big]
    cov_b = [coverage(f['box'], line_boxes_big) for f in formulas]
    w_b = float(np.mean([c[0] for c in cov_b]))
    h_b = float(np.mean([c[1] for c in cov_b]))
    ok_b = sum(1 for c in cov_b if c[0] >= 0.9 and c[1] >= 0.9)
    print(f'  行数 {len(lines)} -> {len(lines_big)}')
    print(f'  宽度覆盖 {w_b*100:.1f}%   高度覆盖 {h_b*100:.1f}%   '
          f'完整 {ok_b}/{len(formulas)}')

    # ================= C. 新流程: 解耦 =================
    print('\n' + '=' * 80)
    print('C. 新流程 (解耦): 页面级 MFD -> 涂白 -> DBNet')
    print('=' * 80)

    pipe = pipe_mod.OnnxPagePipeline(use_gpu=False, verbose=True, mask_mode='mask')
    t0 = time.time()
    items, dbg = pipe(img, return_debug=True)
    dt = time.time() - t0

    # 公式覆盖度: 新流程把公式框整体交给 MFR, 覆盖度视为 100%
    w_c = h_c = 1.0
    ok_c = len(formulas)
    print(f'  公式区域: {dbg["n_formula"]} 个 (全部整体送 MFR)')
    print(f'  文本行  : {dbg["n_text"]} 行')
    print(f'  耗时    : {dt:.1f}s  '
          f'(MFD {dbg["mfd_time"]:.1f}s / OCR {dbg["ocr_time"]:.1f}s / '
          f'MFR {dbg["mfr_time"]:.1f}s)')

    # 涂白前后 DBNet 行框数量对比
    n_raw = len(dbg['det_boxes_original'])
    print(f'  DBNet 在原图上行数: {n_raw}   ->  涂白后 {dbg["n_text"]}')

    # 被丢弃的残留行
    print(f'  仍落在公式内被丢弃的行: {len(dbg["dropped_text_lines"])}')

    # 检查: 涂白后的文本行是否还压在公式上
    residual = []
    for r in dbg['text_lines']:
        if r['formula_ratio'] > 0.3:
            residual.append(r)
    print(f'  涂白后仍有 >30% 面积压在公式上的行: {len(residual)}')

    # ================= 公式识别质量对比 =================
    print('\n' + '=' * 80)
    print('D. 公式识别质量: 完整裁剪 (新流程) vs 碎片裁剪 (旧流程)')
    print('=' * 80)

    mfr = mfr_mod.FormulaRecognizerONNX(use_gpu=False, verbose=False)

    # --- 新流程: 完整裁剪 ---
    print('\n--- 新流程: MFD 完整框裁剪 ---')
    new_rows = []
    for it in dbg['formula_items']:
        latex = it['latex']
        s, tag = best_gt(latex)
        new_rows.append({'box': it['box'], 'type': it['formula_type'],
                         'latex': latex, 'sim': s, 'gt': tag})
    # 只保留与 GT 相关的 (sim > 0.3 说明是这 10 道题之一)
    relevant = [r for r in new_rows if r['sim'] > 0.3]
    for r in sorted(relevant, key=lambda r: -r['sim']):
        ld = (r['latex'][:46] + '..') if len(r['latex']) > 48 else r['latex']
        print(f"  {r['sim']*100:5.1f}%  {r['type']:9} {ld}")
    new_avg = float(np.mean([r['sim'] for r in relevant])) if relevant else 0.0
    new_ok = sum(1 for r in relevant if r['sim'] >= 0.8)
    print(f'\n  匹配到的题目: {len(relevant)}   平均相似度 {new_avg*100:.1f}%   '
          f'>=80%: {new_ok}')

    # --- 旧流程: 用 DBNet 行框裁剪后识别 ---
    print('\n--- 旧流程: DBNet 行框裁剪 (含公式碎片) ---')
    old_rows = []
    for r in lines:
        box = quad_to_xyxy(r['box'])
        # 只关心与公式区域有交叠的行
        ratios = [inter_area(box, fb) /
                  max(1.0, (box[2]-box[0])*(box[3]-box[1])) for fb in fboxes]
        if max(ratios) < 0.2:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        latex = mfr(img[y1:y2, x1:x2])
        s, tag = best_gt(latex)
        old_rows.append({'box': box, 'latex': latex, 'sim': s, 'gt': tag})
    for r in sorted(old_rows, key=lambda r: -r['sim'])[:14]:
        ld = (r['latex'][:46] + '..') if len(r['latex']) > 48 else r['latex']
        print(f"  {r['sim']*100:5.1f}%  {ld}")
    old_avg = float(np.mean([r['sim'] for r in old_rows])) if old_rows else 0.0
    old_best = float(np.max([r['sim'] for r in old_rows])) if old_rows else 0.0

    # 逐题: 旧流程最好能拿到多少
    print('\n--- 逐题对比 (每题取所有裁剪中的最佳相似度) ---')
    print(f"  {'题目':<26} {'旧流程':>10} {'新流程':>10}")
    print('  ' + '-' * 50)
    per_q = []
    for gt, disp in GT_FORMULAS:
        o = max((sim(r['latex'], gt) for r in old_rows), default=0.0)
        n = max((sim(r['latex'], gt) for r in new_rows), default=0.0)
        per_q.append({'gt': gt, 'disp': disp, 'old': o, 'new': n})
        dd = (disp[:24] + '..') if len(disp) > 26 else disp
        print(f'  {dd:<26} {o*100:>9.1f}% {n*100:>9.1f}%')

    old_ok = sum(1 for p in per_q if p['old'] >= 0.8)
    new_okq = sum(1 for p in per_q if p['new'] >= 0.8)
    old_m = float(np.mean([p['old'] for p in per_q]))
    new_m = float(np.mean([p['new'] for p in per_q]))
    print('  ' + '-' * 50)
    print(f"  {'平均':<26} {old_m*100:>9.1f}% {new_m*100:>9.1f}%")
    print(f"  {'>=80% 题数':<24} {old_ok:>9}/10 {new_okq:>9}/10")

    # ================= 汇总 =================
    print('\n' + '=' * 80)
    print('汇总: 公式区域覆盖度')
    print('=' * 80)
    print(f'\n  {"流程":<26} {"宽度覆盖":>10} {"高度覆盖":>10} {"完整":>10}')
    print('  ' + '-' * 60)
    print(f'  {"A 旧流程 (DBNet 切行)":<24} {w_a*100:>9.1f}% {h_a*100:>9.1f}% '
          f'{ok_a:>6}/{len(formulas)}')
    print(f'  {"B 放大 DBNet 输入":<26} {w_b*100:>9.1f}% {h_b*100:>9.1f}% '
          f'{ok_b:>6}/{len(formulas)}')
    print(f'  {"C 新流程 (页面级解耦)":<24} {w_c*100:>9.1f}% {h_c*100:>9.1f}% '
          f'{ok_c:>6}/{len(formulas)}')

    print(f'\n  公式识别 (10 道题平均最佳相似度):')
    print(f'    旧流程 (DBNet 行裁剪): {old_m*100:.1f}%   >=80%: {old_ok}/10')
    print(f'    新流程 (MFD 完整裁剪): {new_m*100:.1f}%   >=80%: {new_okq}/10')

    # ================= 可视化 =================
    vis = pipe.draw(img, items)
    cv2.putText(vis, 'C: page-level decoupled (green=text, orange=inline, blue=display)',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.imwrite(str(OUT / 'C_page_decoupled.jpg'), vis)

    vis_a = img.copy()
    for b in line_boxes:
        cv2.rectangle(vis_a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (255, 0, 0), 4)
    for f in formulas:
        b = f['box']
        cv2.rectangle(vis_a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 0, 255), 5)
    cv2.putText(vis_a, 'A: coupled (blue=DBNet lines, red=MFD formulas)',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.imwrite(str(OUT / 'A_coupled.jpg'), vis_a)

    # 涂白后的文本图
    masked = pipe._mask_regions(img, fboxes)
    cv2.imwrite(str(OUT / 'C_text_only_masked.jpg'), masked)

    # ================= 报告 =================
    rep = {
        'page': {'size': [W, H], 'n_formula': len(formulas),
                 'n_dbnet_lines_original': len(lines)},
        'A_coupled': {'w_cov': w_a, 'h_cov': h_a, 'n_ok': ok_a,
                      'n_lines': len(lines)},
        'B_bigger_input': {'w_cov': w_b, 'h_cov': h_b, 'n_ok': ok_b,
                           'n_lines': len(lines_big)},
        'C_decoupled': {'w_cov': w_c, 'h_cov': h_c, 'n_ok': ok_c,
                        'n_text': dbg['n_text'], 'n_formula': dbg['n_formula'],
                        'dropped_lines': len(dbg['dropped_text_lines']),
                        'residual_lines': len(residual),
                        'time_total': dt,
                        'time_mfd': dbg['mfd_time'],
                        'time_ocr': dbg['ocr_time'],
                        'time_mfr': dbg['mfr_time']},
        'recognition': {
            'old_avg': old_m, 'old_ok': old_ok,
            'new_avg': new_m, 'new_ok': new_okq,
            'per_question': per_q,
        },
        'new_formula_items': new_rows,
    }
    rp = OUT / 'decoupling.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'\n报告: {rp}')
    print(f'可视化: {OUT}')


if __name__ == '__main__':
    main()
