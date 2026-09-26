#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniMERNet ONNX FP16 量化
========================

将 encoder / decoder(无cache) / decoder_prefill / decoder_step 转为 FP16。
权重转 FP16；keep_io_types=True 保持图输入输出 FP32 (含 KV cache 张量)，
LayerNorm/softmax 等敏感算子由转换器保持 FP32。

用法:
    python onnx_pipeline/export/3_formula_recognize_quantize_fp16.py
    python ... --only kv     # 仅转换 KV 两图并清理其 FP32 源 (日常增量的快路径)
"""

import sys
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort

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
FP32_DIR = {
    'unimernet_encoder.onnx': ROOT / 'weights' / 'formula_recognize' / 'onnx',
    'unimernet_decoder.onnx': ROOT / 'weights' / 'formula_recognize' / 'onnx',
    'unimernet_decoder_prefill.onnx': ROOT / 'weights' / 'formula_recognize' / 'onnx_kv',
    'unimernet_decoder_step.onnx': ROOT / 'weights' / 'formula_recognize' / 'onnx_kv',
}
DST = ROOT / 'weights' / 'formula_recognize' / 'onnx_fp16'
DST.mkdir(parents=True, exist_ok=True)

ALL_FILES = list(FP32_DIR.keys())


def fmt(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'


def convert(name):
    src = FP32_DIR[name] / name
    dst = DST / name
    if not src.exists():
        print(f'  [SKIP] {name} (FP32 源不存在: {src})')
        return False
    src_size = src.stat().st_size

    # 清理上次残留的外部数据文件
    for old in dst.parent.glob(dst.name + '.data'):
        old.unlink()

    from onnxconverter_common import float16
    model = onnx.load(str(src))
    # TorchScript 导出会写 value_info；FP16 转换后这些声明变陈旧,
    # ORT 加载会报 type mismatch -> 转换前清空, 让 ORT 重新推导
    del model.graph.value_info[:]
    if name.startswith('unimernet_decoder_'):
        # KV 图: 中间张量与图输出同名 (cache 直传), keep_io_types=True 会产生
        # fp32/fp16 类型冲突 -> 全图 FP16 (logits/cache 均为 fp16, argmax 无碍)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    else:
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    del model_fp16.graph.value_info[:]
    onnx.save(model_fp16, str(dst))

    # 校验以 ORT 加载为准 (infer_shapes 对 fp16 图不稳, 略过)
    ext = list(dst.parent.glob(dst.name + '.data'))
    print(f'  [OK] {name}: {fmt(src_size)} -> {fmt(dst.stat().st_size)}'
          + ('  [WARN 外部数据]' if ext else ''))
    return True


def main():
    only_kv = '--only' in sys.argv and 'kv' in sys.argv
    files = ([n for n in ALL_FILES if 'prefill' in n or 'step' in n]
             if only_kv else ALL_FILES)

    print('=' * 74)
    print('UniMERNet ONNX FP16 量化' + ('  (仅 KV 图)' if only_kv else ''))
    print('=' * 74)

    print('\n【1】转换模型')
    print('-' * 74)
    done = [n for n in files if convert(n)]

    # 复制配置 (与 encoder 同级)
    cfg_src = FP32_DIR['unimernet_encoder.onnx'] / 'unimernet_config.json'
    if cfg_src.exists():
        shutil.copy2(cfg_src, DST / 'unimernet_config.json')

    print('\n【2】验证 FP16 模型可加载')
    print('-' * 74)
    ok = True
    for name in files:
        p = DST / name
        if not p.exists():
            continue
        try:
            so = ort.SessionOptions()
            if 'prefill' in name or 'step' in name:
                # 规避 ORT SimplifiedLayerNormFusion 对 InsertedPrecisionFreeCast
                # 的初始化 bug (1.24 复现); EXTENDED 级别不跑该图变换。
                # 运行时 (FormulaRecognizerONNX) 同样用 EXTENDED 加载 KV 图。
                so.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
            sess = ort.InferenceSession(str(p), sess_options=so,
                                        providers=['CPUExecutionProvider'])
            ins, outs = sess.get_inputs(), sess.get_outputs()
            print(f'  {name}: {len(ins)} in / {len(outs)} out  [OK]')
            for i in ins[:3]:
                print(f'    IN  {i.name:26} {i.shape} {i.type}')
            for o in outs[:2]:
                print(f'    OUT {o.name:26} {o.shape} {o.type}')
        except Exception as e:
            ok = False
            print(f'  {name}: [ERROR] {type(e).__name__}: {e}')

    # KV 的 FP32 中间产物 (~620MB) 仅在验证全部通过后删除
    if ok and done and only_kv:
        kv_dir = FP32_DIR[done[0]]
        for p in kv_dir.glob('unimernet_decoder_*'):
            p.unlink()
        print(f'  [CLEAN] 已删除 {kv_dir} 下的 KV FP32 中间文件')

    print('\n【3】产物清单')
    print('-' * 74)
    for p in sorted(DST.iterdir()):
        print(f'  {p.name:40} {fmt(p.stat().st_size)}')


if __name__ == '__main__':
    main()
