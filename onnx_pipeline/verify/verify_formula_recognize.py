#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证纯 ONNX 公式识别器
======================

1. 与 PyTorch (官方代码) 结果对比
2. 验证无 torch/transformers 依赖
3. 完整公式识别测试
"""

import os
import sys
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os
from pathlib import Path as _Path

# --- 自动定位项目根, 使 Path.cwd() 指向项目根 (shim) ---
def _chdir_project_root():
    _p = _Path(__file__).resolve()
    for _parent in [_p] + list(_p.parents):
        if (_parent / 'weights').is_dir() and (_parent / 'func').is_dir():
            _os.chdir(_parent)
            return _parent
    return _Path.cwd()

_chdir_project_root()
# --- shim end ---

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr'
OUT.mkdir(parents=True, exist_ok=True)

FORMULAS = [
    ('f(x) = x² + 2x - 3', 'quadratic'),
    ('(a+b)/2 ≥ √(ab)', 'inequality'),
    ('x = (-b ± √(b² - 4ac)) / 2a', 'quad_formula'),
    ('∫₀¹ x² dx = 1/3', 'integral'),
    ('y = (a+b)/(c-d)', 'fraction'),
    ('E = mc²', 'einstein'),
    ('a² + b² = c²', 'pythagoras'),
]


def make_crops():
    font_path = r'C:\Windows\Fonts\msyh.ttc'
    if not os.path.exists(font_path):
        font_path = r'C:\Windows\Fonts\simhei.ttf'
    crops = []
    for text, name in FORMULAS:
        font = ImageFont.truetype(font_path, 56)
        img = Image.new('RGB', (860, 140), 'white')
        d = ImageDraw.Draw(img)
        d.text((25, 30), text, fill='black', font=font)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        p = OUT / f'formula_crop_{name}.jpg'
        cv2.imwrite(str(p), bgr)
        crops.append((name, text, bgr, p))
    return crops


class Blocker:
    """
    拦截 torch / transformers，证明纯 ONNX 路径无深度学习框架依赖。

    注意: 不拦截 tokenizers —— 它是独立的 Rust 实现，仅用于 BPE 解码，
    不引入 torch。
    """
    BLOCKED = ('torch', 'transformers', 'paddle', 'paddleocr')

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.BLOCKED:
            raise ImportError(f'[BLOCKED] {fullname}')
        return None


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    print('=' * 74)
    print('纯 ONNX 公式识别器验证 (UniMERNet-tiny)')
    print('=' * 74)

    print('\n【0】生成公式测试图')
    print('-' * 74)
    crops = make_crops()
    for name, text, _, p in crops:
        print(f'  {name:16} {text!r}')

    # ---------- 1. PyTorch 参考 ----------
    print('\n【1】PyTorch 参考结果 (官方代码)')
    print('-' * 74)
    ref = {}
    try:
        LEGACY = ROOT / 'venv_legacy_tf'
        CODE_ROOT = ROOT / 'weights' / 'formula_recognize' / 'unimernet_code'
        MODEL_DIR = ROOT / 'weights' / 'formula_recognize' / 'unimernet_tiny'
        sys.path.insert(0, str(CODE_ROOT))
        sys.path.insert(0, str(LEGACY))

        import torch
        from unimernet.encoder_decoder import DonutEncoderDecoder, DonutTokenizer

        tokenizer = DonutTokenizer(str(MODEL_DIR))
        model = DonutEncoderDecoder(
            str(MODEL_DIR), num_tokens=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        sd = torch.load(str(MODEL_DIR / 'unimernet_tiny_pdfkit.pth'),
                        map_location='cpu', weights_only=False)
        if 'model' in sd:
            sd = sd['model']
        sd = {k[len('model.'):] if k.startswith('model.') else k: v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        print('  [OK] PyTorch 模型就绪')

        for name, text, bgr, _ in crops:
            arr = np.array(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert('L'))
            arr = cv2.resize(arr, (672, 192)).astype(np.float32) / 255.0
            arr = (arr - 0.7931) / 0.1738
            pv = torch.tensor(np.stack([arr]*3)[None])
            with torch.no_grad():
                out = model.generate(
                    pixel_values=pv, temperature=0.0, max_new_tokens=512,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    do_sample=False, top_p=0.95)
            s = tokenizer.token2str(out)
            ref[name] = s[0] if isinstance(s, list) else s
            print(f'  [{name}] {ref[name]!r}')

    except Exception as e:
        import traceback
        print(f'  [WARN] PyTorch 参考失败: {type(e).__name__}: {e}')
        traceback.print_exc(limit=3)

    # ---------- 2. 纯 ONNX (拦截 torch) ----------
    print('\n【2】纯 ONNX 结果 (拦截 torch/transformers)')
    print('-' * 74)

    for n in list(sys.modules):
        if n.split('.')[0] in Blocker.BLOCKED:
            del sys.modules[n]
    import gc
    gc.collect()
    sys.meta_path.insert(0, Blocker())

    for m in ('torch', 'transformers'):
        try:
            __import__(m)
            print(f'  [FAIL] {m} 未被拦截')
            return 1
        except ImportError:
            print(f'  [OK] {m} 已拦截')

    mod = load_mod('formula_recognize_onnx',
                   ROOT / 'func/algorithm/formula_recognize_onnx.py')
    rec = mod.FormulaRecognizerONNX(use_gpu=False, verbose=True)

    print()
    results = {}
    for name, text, bgr, _ in crops:
        t0 = time.time()
        latex = rec(bgr)
        dt = time.time() - t0
        results[name] = latex
        print(f'  [{name}] {latex!r}  ({dt:.2f}s)')

    # ---------- 3. 对比 ----------
    print('\n【3】ONNX vs PyTorch 对比')
    print('-' * 74)
    if ref:
        match = 0
        for name in results:
            a = results[name].replace(' ', '')
            b = ref.get(name, '').replace(' ', '')
            ok = a == b
            match += ok
            print(f'  {name:16} {"一致" if ok else "不一致"}')
            if not ok:
                print(f'      ONNX: {results[name]!r}')
                print(f'      Torch: {ref.get(name)!r}')
        print(f'\n  一致率: {match}/{len(results)} ({match/len(results)*100:.0f}%)')

    # ---------- 4. 可视化 ----------
    print('\n【4】可视化')
    print('-' * 74)
    tiles = []
    for name, text, bgr, _ in crops:
        tile = rec.draw(bgr, results[name])
        tiles.append(tile)
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles]
        grid = np.vstack(tiles)
        vis_path = OUT / 'formula_recognize_onnx_result.jpg'
        cv2.imwrite(str(vis_path), grid)
        print(f'  {vis_path}')

    loaded = [m for m in sys.modules if m.split('.')[0] in
              ('torch', 'transformers', 'paddle', 'paddleocr')]
    print(f'\n已加载的 torch/transformers 模块: {loaded if loaded else "无"}')
    print('\n' + '=' * 74)
    if not loaded:
        print('[SUCCESS] 纯 ONNX 公式识别器运行成功，无 torch/transformers 依赖')
    return 0


if __name__ == '__main__':
    sys.exit(main())
