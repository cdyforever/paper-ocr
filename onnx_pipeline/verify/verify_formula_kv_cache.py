#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KV cache 快速路径 vs 无 cache 旧路径 — 端到端对分 + 测速
========================================================

用 30 张公式 crop (dump_intermediates.py 产物) 对比:
  - 逐 token 一致性 (KV 路径必须与旧路径完全一致)
  - 两种解码的总耗时

前置: 先运行 dump_intermediates.py 生成
      weights/onnx_ocr/intermediates/20_formula_crops/*.jpg

用法: python onnx_pipeline/verify/verify_formula_kv_cache.py
"""

import time
import importlib.util
from pathlib import Path

import numpy as np

# --- 自动定位项目根, 使 Path.cwd() 指向项目根 ---
ROOT = Path(__file__).resolve().parents[2]
import os as _os
_os.chdir(ROOT)

_spec = importlib.util.spec_from_file_location(
    'formrec', ROOT / 'func/algorithm/formula_recognize_onnx.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FormulaRecognizerONNX = _mod.FormulaRecognizerONNX

CROPS = sorted(Path('weights/onnx_ocr/intermediates/20_formula_crops').glob('*.jpg'))
assert CROPS, '缺少公式 crop, 请先运行 dump_intermediates.py'

slow = FormulaRecognizerONNX(use_kv=False, verbose=False)
fast = FormulaRecognizerONNX(use_kv=True, verbose=False)
print(f'crop 数: {len(CROPS)}   旧路径 kv_mode={slow.kv_mode}   '
      f'KV 路径 kv_mode={fast.kv_mode}')

t_s = t_f = 0.0
n_same = 0
diff_examples = []
for p in CROPS:
    img = str(p)
    t0 = time.time(); latex_s, ids_s = slow(img, return_ids=True); t_s += time.time() - t0
    t0 = time.time(); latex_f, ids_f = fast(img, return_ids=True); t_f += time.time() - t0
    same = ids_s == ids_f
    n_same += same
    if not same:
        diff_examples.append((p.name, ids_s[:12], ids_f[:12],
                              len(ids_s), len(ids_f)))
    print(f'{p.name}: {"OK" if same else "DIFF"}  slow={len(ids_s)}tok  fast={len(ids_f)}tok')

print()
print(f'token-equal: {n_same}/{len(CROPS)}   (要求 100%)')
print(f'slow (no-cache): {t_s:.2f}s   fast (KV): {t_f:.2f}s   speedup x{t_s / max(t_f, 1e-9):.2f}')
print('注: CPU 上解码耗时由权重内存带宽主导 (~150MB/token 全读), '
      'KV 消除历史重算但省不掉权重搬运, 故 CPU 提速有限、GPU 上收益显著。')
for ex in diff_examples:
    print('DIFF', ex)
assert n_same == len(CROPS), 'KV 路径与旧路径 token 不一致!'
