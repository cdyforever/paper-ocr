#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniMERNet-tiny 导出为 ONNX
==========================

策略: 分离导出
  1. encoder.onnx  : pixel_values -> encoder_hidden_states
  2. decoder.onnx  : (input_ids, encoder_hidden_states) -> logits
                     (单步, 无 KV cache, 由 Python 侧做自回归循环)

这样可以在纯 ONNX Runtime 下完成自回归解码，不需要 transformers。
"""

import os
import sys
import time
import json
from pathlib import Path

ROOT = Path.cwd()
LEGACY = ROOT / 'venv_legacy_tf'
CODE_ROOT = ROOT / 'weights' / 'formula_recognize' / 'unimernet_code'
MODEL_DIR = ROOT / 'weights' / 'formula_recognize' / 'unimernet_tiny'
OUT_DIR = ROOT / 'weights' / 'formula_recognize' / 'onnx'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(LEGACY))

import torch
import torch.nn as nn
import numpy as np
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


# ---------------------------------------------------------------------------
class EncoderWrapper(nn.Module):
    """pixel_values -> last_hidden_state"""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        return out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state


class DecoderWrapper(nn.Module):
    """
    单步解码器: (input_ids, encoder_hidden_states) -> logits

    input_ids: [batch, seq_len]  (完整历史，无 cache)
    encoder_hidden_states: [batch, enc_len, hidden]
    logits: [batch, seq_len, vocab]
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, input_ids, encoder_hidden_states):
        out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def main():
    print('=' * 74)
    print('UniMERNet-tiny -> ONNX')
    print('=' * 74)

    import transformers
    print(f'\ntransformers: {transformers.__version__}')

    from unimernet.encoder_decoder import DonutEncoderDecoder, DonutTokenizer

    # ---------- 加载 ----------
    print('\n【1】加载 PyTorch 模型')
    print('-' * 74)
    tokenizer = DonutTokenizer(str(MODEL_DIR))
    model = DonutEncoderDecoder(
        str(MODEL_DIR),
        num_tokens=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sd = torch.load(str(MODEL_DIR / 'unimernet_tiny_pdfkit.pth'),
                    map_location='cpu', weights_only=False)
    if 'model' in sd:
        sd = sd['model']
    new_sd = {k[len('model.'):] if k.startswith('model.') else k: v
              for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f'  missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()

    vocab_size = model.model.decoder.config.vocab_size
    print(f'  vocab_size={vocab_size}')

    # ---------- 导出 encoder ----------
    print('\n【2】导出 encoder.onnx')
    print('-' * 74)
    enc = EncoderWrapper(model.model).eval()
    dummy_pv = torch.randn(1, 3, 192, 672)

    enc_path = OUT_DIR / 'unimernet_encoder.onnx'
    with torch.no_grad():
        enc_out = enc(dummy_pv)
    print(f'  encoder 输出 shape: {tuple(enc_out.shape)}')

    torch.onnx.export(
        enc, dummy_pv, str(enc_path),
        input_names=['pixel_values'],
        output_names=['encoder_hidden_states'],
        dynamic_axes={
            'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
            'encoder_hidden_states': {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f'  [OK] {enc_path.name}  ({enc_path.stat().st_size/1e6:.1f} MB)')

    # ---------- 导出 decoder ----------
    print('\n【3】导出 decoder.onnx')
    print('-' * 74)
    dec = DecoderWrapper(model.model).eval()

    dummy_ids = torch.tensor([[0, 100, 200]], dtype=torch.long)
    dummy_enc = enc_out.detach()

    dec_path = OUT_DIR / 'unimernet_decoder.onnx'
    with torch.no_grad():
        logits = dec(dummy_ids, dummy_enc)
    print(f'  decoder 输出 shape: {tuple(logits.shape)}')

    torch.onnx.export(
        dec, (dummy_ids, dummy_enc), str(dec_path),
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'encoder_hidden_states': {0: 'batch'},
            'logits': {0: 'batch', 1: 'seq_len'},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f'  [OK] {dec_path.name}  ({dec_path.stat().st_size/1e6:.1f} MB)')

    # ---------- 保存配置 ----------
    print('\n【4】保存推理配置')
    print('-' * 74)
    cfg = {
        'vocab_size': vocab_size,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'max_seq_len': 1536,
        'image_size': [192, 672],
        'image_mean': 0.7931,
        'image_std': 0.1738,
    }
    cfg_path = OUT_DIR / 'unimernet_config.json'
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'  [OK] {cfg_path.name}')
    print(json.dumps(cfg, indent=2))

    # ---------- 验证 ----------
    print('\n【5】验证 ONNX')
    print('-' * 74)
    import onnxruntime as ort
    for name in ['unimernet_encoder.onnx', 'unimernet_decoder.onnx']:
        p = OUT_DIR / name
        sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
        print(f'\n  {name}:')
        for i in sess.get_inputs():
            print(f'    IN  {i.name:26} {i.shape} {i.type}')
        for o in sess.get_outputs():
            print(f'    OUT {o.name:26} {o.shape} {o.type}')

    print(f'\n输出目录: {OUT_DIR}')
    for p in sorted(OUT_DIR.iterdir()):
        print(f'  {p.name:32} {p.stat().st_size/1e6:8.2f} MB')


if __name__ == '__main__':
    main()
