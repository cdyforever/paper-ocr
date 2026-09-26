#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniMERNet-tiny 解码器 -> ONNX (KV Cache 版)
===========================================

背景: 原 decoder.onnx 每步重跑整个历史序列 (O(n^2)), 30 个公式 MFR 耗时 96.4s。
本脚本把解码器拆成两个带 KV Cache 的图:

  unimernet_decoder_prefill.onnx
    输入:  input_ids [1, seq], encoder_hidden_states [1, enc_len, 512], position_ids [1, seq]
    输出:  logits [1, seq, vocab] + 8 层 x (self_k, self_v, cross_k, cross_v)
    (seq=1 即可完成预填充: 只喂 [bos])

  unimernet_decoder_step.onnx
    输入:  input_ids [1, 1], encoder_hidden_states, position_ids [1, 1],
           8 层 x (past_self_k [1,16,past,16], past_self_v [1,16,past,32],
                   past_cross_k [1,16,enc_len,16], past_cross_v [1,16,enc_len,32])
    输出:  logits [1, 1, vocab] + 8 层 x (present_self_k, present_self_v)
    cross K/V 只依赖 encoder 输出 -> 预填充算一次, 每步原样传回 (图中走复用分支)

实现要点 (与 CustomMBartDecoder.forward 逐行对齐):
  - embed = embed_tokens(ids) * embed_scale   (MBartScaledWordEmbedding 内已乘一次, 再乘一次)
  - pos   = embedding(position_ids + offset, embed_positions.weight)   (offset=2)
  - hidden = layernorm_embedding(embed + pos)  -> 8 x layer -> layer_norm -> lm_head
  - attention_mask 全程 None (与现有无 cache 版一致, 单步解码天然满足因果性)
  - position_ids 显式作为图输入, 避免 arange(past_len) 被 trace 成常量

导出环境: 主环境 torch + venv_legacy_tf(transformers 4.36) 路径 shim。
运行时 (识别器) 仍然只用 onnxruntime, 不引入任何框架依赖。
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path.cwd()
LEGACY = ROOT / 'venv_legacy_tf'
CODE_ROOT = ROOT / 'weights' / 'formula_recognize' / 'unimernet_code'
MODEL_DIR = ROOT / 'weights' / 'formula_recognize' / 'unimernet_tiny'
OUT_DIR = ROOT / 'weights' / 'formula_recognize' / 'onnx_kv'
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


def flatten(lst):
    out = []
    for t in lst:
        out.extend(t)
    return out


# ---------------------------------------------------------------------------
def _named_axes(name, axes):
    """ONNX 导出的 (名, {轴:符号}) 对。"""
    return name, axes


def prefill_io(n_layers, enc_sym='enc_len'):
    """prefill 图的输入/输出命名与动态轴。"""
    ins, outs = [], []
    ins.append(_named_axes('input_ids', {1: 'seq_len'}))
    ins.append(_named_axes('encoder_hidden_states', {1: enc_sym}))
    ins.append(_named_axes('position_ids', {1: 'seq_len'}))
    outs.append(_named_axes('logits', {1: 'seq_len'}))
    for i in range(n_layers):
        outs.append(_named_axes(f'present_self_key_{i}', {2: 'seq_len'}))
        outs.append(_named_axes(f'present_self_value_{i}', {2: 'seq_len'}))
        outs.append(_named_axes(f'present_cross_key_{i}', {2: enc_sym}))
        outs.append(_named_axes(f'present_cross_value_{i}', {2: enc_sym}))
    return ins, outs


def step_io(n_layers, enc_sym='enc_len'):
    """step 图的输入/输出命名与动态轴。"""
    ins, outs = [], []
    ins.append(_named_axes('input_ids', {}))
    ins.append(_named_axes('encoder_hidden_states', {1: enc_sym}))
    ins.append(_named_axes('position_ids', {}))
    for i in range(n_layers):
        ins.append(_named_axes(f'past_self_key_{i}', {2: 'past_len'}))
        ins.append(_named_axes(f'past_self_value_{i}', {2: 'past_len'}))
        ins.append(_named_axes(f'past_cross_key_{i}', {2: enc_sym}))
        ins.append(_named_axes(f'past_cross_value_{i}', {2: enc_sym}))
    outs.append(_named_axes('logits', {}))
    for i in range(n_layers):
        outs.append(_named_axes(f'present_self_key_{i}', {2: 'present_len'}))
        outs.append(_named_axes(f'present_self_value_{i}', {2: 'present_len'}))
    return ins, outs


def export_kwargs(io_pairs, opset=14):
    """把 (名, {轴:符号}) 转成 torch.onnx.export 的扁平 dynamic_axes 参数。"""
    ins, outs = io_pairs
    names = [n for n, _ in ins + outs]
    axes = {n: a for n, a in ins + outs if a}
    return dict(input_names=[n for n, _ in ins],
                output_names=[n for n, _ in outs],
                dynamic_axes=axes,
                opset_version=opset,
                do_constant_folding=True)


# ---------------------------------------------------------------------------
class DecoderKVBase(nn.Module):
    """手工复刻 CustomMBartDecoder 的前向, 使其能被拆成 prefill / step 两图。"""

    def __init__(self, causal_lm):
        super().__init__()
        self.dec = causal_lm.model.decoder          # CustomMBartDecoder
        self.lm_head = causal_lm.lm_head
        self.n_layers = len(self.dec.layers)

    def _embed(self, input_ids, position_ids):
        dec = self.dec
        embeds = dec.embed_tokens(input_ids) * dec.embed_scale          # 双重缩放, 与源码一致
        pos = nn.functional.embedding(
            position_ids + dec.embed_positions.offset,
            dec.embed_positions.weight)
        hidden = dec.layernorm_embedding(embeds + pos)
        return hidden

    def _run_layers(self, hidden, enc, past_by_layer):
        presents = []
        for i, layer in enumerate(self.dec.layers):
            pkv = past_by_layer[i] if past_by_layer is not None else None
            out = layer(
                hidden,
                attention_mask=None,
                encoder_hidden_states=enc,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=pkv,
                output_attentions=False,
                use_cache=True,
            )
            hidden = out[0]
            presents.append(out[1])                 # (self_k, self_v, cross_k, cross_v)
        return hidden, presents


class DecoderPrefill(DecoderKVBase):
    """无 cache 前向 (通常 seq=1 喂 [bos]): 输出全部 8x4 cache, cross K/V 在此计算。"""

    def forward(self, input_ids, encoder_hidden_states, position_ids):
        hidden, presents = self._run_layers(
            self._embed(input_ids, position_ids), encoder_hidden_states, None)
        logits = self.lm_head(self.dec.layer_norm(hidden))
        return (logits,) + tuple(flatten(presents))


class DecoderStep(DecoderKVBase):
    """单步 + cache 前向: 输入 8x4 cache, 只更新/输出 self cache (cross 原样复用)。"""

    def forward(self, input_ids, encoder_hidden_states, position_ids, *past):
        assert len(past) == 4 * self.n_layers
        past_by_layer = [tuple(past[i * 4:(i + 1) * 4]) for i in range(self.n_layers)]
        hidden, presents = self._run_layers(
            self._embed(input_ids, position_ids), encoder_hidden_states, past_by_layer)
        logits = self.lm_head(self.dec.layer_norm(hidden))
        # presents[i] = (self_k, self_v, cross_k, cross_v); cross 未变, 不再输出
        self_kv = tuple(t for p in presents for t in p[:2])
        return (logits,) + self_kv


# ---------------------------------------------------------------------------
def load_model():
    from unimernet.encoder_decoder import DonutEncoderDecoder, DonutTokenizer

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
    return model, tokenizer


def eager_reference(causal_lm, ids, enc):
    """生产路径(无 cache, 全序列) logits —— 与 Python 识别器循环完全一致。"""
    with torch.no_grad():
        out = causal_lm(input_ids=torch.tensor([ids], dtype=torch.long),
                        encoder_hidden_states=enc, use_cache=False, return_dict=True)
    return out.logits[0]                             # [T, vocab]


def eager_kv_chain(prefill, step, ids, enc):
    """prefill+step 链 (PyTorch eager) 的逐步 logits, 用于交叉校验。"""
    bos, rest = ids[0], ids[1:]
    with torch.no_grad():
        out = prefill(torch.tensor([[bos]], dtype=torch.long), enc,
                      torch.tensor([[0]], dtype=torch.long))
        logits = [out[0][0, 0]]
        presents = [tuple(out[1 + i * 4: 5 + i * 4]) for i in range(8)]
        for t, tok in enumerate(rest):
            past = tuple(flatten(presents))
            out = step(torch.tensor([[tok]], dtype=torch.long), enc,
                       torch.tensor([[t + 1]], dtype=torch.long), *past)
            logits.append(out[0][0, 0])
            new_self = out[1:]
            presents = [(new_self[i * 2], new_self[i * 2 + 1],
                         presents[i][2], presents[i][3]) for i in range(8)]
    return torch.stack(logits)                        # [T, vocab]


def ort_kv_chain(ops, oprefill, opstep, ids, enc_np):
    """ONNX Runtime 上的 prefill+step 链。"""
    bos, rest = ids[0], ids[1:]
    out = oprefill.run(None, {
        'input_ids': np.array([[bos]], np.int64),
        'encoder_hidden_states': enc_np,
        'position_ids': np.array([[0]], np.int64)})
    logits = [out[0][0, 0]]
    presents = [tuple(out[1 + i * 4: 5 + i * 4]) for i in range(8)]
    step_in = {i.name for i in opstep.get_inputs()}
    past_names = []
    for i in range(8):
        past_names += [f'past_self_key.{i}', f'past_self_value.{i}',
                       f'past_cross_key.{i}', f'past_cross_value.{i}']
    for t, tok in enumerate(rest):
        feeds = {
            'input_ids': np.array([[tok]], np.int64),
            'position_ids': np.array([[t + 1]], np.int64)}
        if 'encoder_hidden_states' in step_in:
            feeds['encoder_hidden_states'] = enc_np
        feeds.update({n: v for n, v in zip(past_names, flatten(presents))
                      if n in step_in})
        out = opstep.run(None, feeds)
        logits.append(out[0][0, 0])
        new_self = out[1:]
        presents = [(new_self[i * 2], new_self[i * 2 + 1],
                     presents[i][2], presents[i][3]) for i in range(8)]
    return np.stack(logits)                           # [T, vocab]


# ---------------------------------------------------------------------------
def main():
    print('=' * 74)
    print('UniMERNet-tiny 解码器 KV Cache ONNX 导出')
    print('=' * 74)

    import transformers
    print(f'\ntransformers: {transformers.__version__}  torch: {torch.__version__}')

    print('\n【1】加载 PyTorch 模型')
    print('-' * 74)
    model, tokenizer = load_model()
    causal_lm = model.model.decoder
    vocab_size = causal_lm.config.vocab_size
    prefill = DecoderPrefill(causal_lm).eval()
    step = DecoderStep(causal_lm).eval()

    n_heads = causal_lm.config.decoder_attention_heads
    d_model = causal_lm.config.d_model
    qk_dim = prefill.dec.layers[0].self_attn.q_proj.out_features
    v_dim = prefill.dec.layers[0].self_attn.v_proj.out_features
    print(f'  vocab={vocab_size} d_model={d_model} heads={n_heads} '
          f'qk_proj={qk_dim} v_proj={v_dim}')

    # ---------- 数值: eager 链 vs 生产无 cache 路径 ----------
    print('\n【2】PyTorch 侧校验: prefill+step 链 == 全序列无 cache')
    print('-' * 74)
    torch.manual_seed(0)
    enc = model.model.encoder(pixel_values=torch.randn(1, 3, 192, 672))[0].detach()
    enc_len = enc.shape[1]
    print(f'  encoder_hidden_states: {tuple(enc.shape)}')

    ids = [0, 100, 200, 42, 7, 512, 3333, 1, 1, 40000, 655]
    ref = eager_reference(causal_lm, ids, enc)
    kv = eager_kv_chain(prefill, step, ids, enc)
    diff = (ref[:len(ids)] - kv).abs().max().item()
    print(f'  max |logits| diff = {diff:.3e}   argmax equal: '
          f'{torch.allclose(ref[:len(ids)].argmax(-1), kv.argmax(-1))}')
    assert diff < 1e-3, 'KV 链与生产路径不一致'

    # ---------- 导出 ----------
    print('\n【3】导出 prefill / step ONNX (opset 14)')
    print('-' * 74)

    cache_shape_k = (1, n_heads, enc_len, qk_dim // n_heads)
    cache_shape_v = (1, n_heads, enc_len, v_dim // n_heads)

    pre_names = ['input_ids', 'encoder_hidden_states', 'position_ids']
    pre_outs = ['logits']
    pre_axes = {
        'input_ids': {1: 'seq'}, 'position_ids': {1: 'seq'},
        'encoder_hidden_states': {1: 'enc_len'}, 'logits': {1: 'seq'},
    }
    step_names = ['input_ids', 'encoder_hidden_states', 'position_ids']
    step_outs = ['logits']
    step_axes = {}   # 注意: cross cache 直接复用后, step 图中 encoder_hidden_states 未被使用,
                     # 导出器会将其从图输入中剔除 (按位置命名, 其余输入名不受影响)

    for i in range(8):
        for kind in ('self_key', 'self_value', 'cross_key', 'cross_value'):
            pre_outs.append(f'present_{kind}.{i}')
        pre_axes[f'present_self_key.{i}'] = {2: 'seq'}
        pre_axes[f'present_self_value.{i}'] = {2: 'seq'}
        pre_axes[f'present_cross_key.{i}'] = {2: 'enc_len'}
        pre_axes[f'present_cross_value.{i}'] = {2: 'enc_len'}

        past_k = f'past_self_key.{i}'
        past_v = f'past_self_value.{i}'
        step_names += [past_k, past_v, f'past_cross_key.{i}', f'past_cross_value.{i}']
        step_axes[past_k] = {2: 'past'}
        step_axes[past_v] = {2: 'past'}
        step_axes[f'past_cross_key.{i}'] = {2: 'enc_len'}
        step_axes[f'past_cross_value.{i}'] = {2: 'enc_len'}
        step_outs += [f'present_self_key.{i}', f'present_self_value.{i}']
        step_axes[f'present_self_key.{i}'] = {2: 'past1'}
        step_axes[f'present_self_value.{i}'] = {2: 'past1'}

    with torch.no_grad():
        # prefill: seq=1 (喂 [bos]); enc_len 由真实 enc 决定
        dp_id = torch.tensor([[0]], dtype=torch.long)
        dp_pos = torch.tensor([[0]], dtype=torch.long)
        torch.onnx.export(
            prefill, (dp_id, enc, dp_pos),
            str(OUT_DIR / 'unimernet_decoder_prefill.onnx'),
            input_names=pre_names, output_names=pre_outs,
            dynamic_axes=pre_axes, opset_version=14, do_constant_folding=True,
            dynamo=False)
        sz = (OUT_DIR / 'unimernet_decoder_prefill.onnx').stat().st_size / 1e6
        print(f'  [OK] unimernet_decoder_prefill.onnx ({sz:.1f} MB)')

        # step: trace 时 past=3 (past>1 且 enc_len 不匹配 seq, 避免分支误折叠)
        dummy_past = []
        for i in range(8):
            dummy_past += [
                torch.randn(1, n_heads, 3, qk_dim // n_heads),
                torch.randn(1, n_heads, 3, v_dim // n_heads),
                torch.randn(*cache_shape_k),
                torch.randn(*cache_shape_v)]
        ds_id = torch.tensor([[42]], dtype=torch.long)
        ds_pos = torch.tensor([[3]], dtype=torch.long)
        torch.onnx.export(
            step, (ds_id, enc, ds_pos, *dummy_past),
            str(OUT_DIR / 'unimernet_decoder_step.onnx'),
            input_names=step_names, output_names=step_outs,
            dynamic_axes=step_axes, opset_version=14, do_constant_folding=True,
            dynamo=False)
        sz = (OUT_DIR / 'unimernet_decoder_step.onnx').stat().st_size / 1e6
        print(f'  [OK] unimernet_decoder_step.onnx ({sz:.1f} MB)')

    # ---------- ONNX Runtime 侧校验 ----------
    print('\n【4】ONNX Runtime 校验 (不同 past 长度逐步对比)')
    print('-' * 74)
    import onnxruntime as ort
    prov = ['CPUExecutionProvider']
    oprefill = ort.InferenceSession(str(OUT_DIR / 'unimernet_decoder_prefill.onnx'), providers=prov)
    opstep = ort.InferenceSession(str(OUT_DIR / 'unimernet_decoder_step.onnx'), providers=prov)

    enc_np = enc.numpy()
    ids_long = [0] + [int(x) % vocab_size for x in range(101, 101 + len(ids) - 1)]
    ort_logits = ort_kv_chain(None, oprefill, opstep, ids_long, enc_np)
    ref_np = eager_reference(causal_lm, ids_long, enc)[:len(ids_long)].numpy()
    d = np.abs(ref_np - ort_logits).max()
    tok_eq = (ref_np.argmax(-1) == ort_logits.argmax(-1)).all()
    print(f'  len(ids)={len(ids_long)}  max |diff| = {d:.3e}  argmax all equal: {tok_eq}')
    assert d < 5e-3 and tok_eq

    # greedy 抽样对比: 与生产无 cache 解码器逐 token 一致性在 verify 脚本中做
    print('\n【5】IO 概览')
    print('-' * 74)
    for s, tag in [(oprefill, 'prefill'), (opstep, 'step')]:
        ins, outs = s.get_inputs(), s.get_outputs()
        print(f'  {tag}: {len(ins)} in / {len(outs)} out')
        print(f'    IN  {ins[0].name} {ins[0].shape} | {ins[1].name} {ins[1].shape}')
        print(f'    OUT {outs[0].name} {outs[0].shape} | {outs[1].name} {outs[1].shape}')

    print(f'\n输出目录: {OUT_DIR}')
    for p in sorted(OUT_DIR.iterdir()):
        print(f'  {p.name:36} {p.stat().st_size / 1e6:8.2f} MB')
    print('\n下一步: python onnx_pipeline/export/3_formula_recognize_quantize_fp16.py')


if __name__ == '__main__':
    main()
