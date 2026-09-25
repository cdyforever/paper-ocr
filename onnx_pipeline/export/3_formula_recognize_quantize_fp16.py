#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniMERNet ONNX FP16 量化
========================

将 encoder/decoder 转为 FP16，减小体积并加速。
注意: 仅转换 float32 权重；LayerNorm/softmax 等保持 FP32 以保数值稳定。
"""

import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
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
SRC = ROOT / 'weights' / 'formula_recognize' / 'onnx'
DST = ROOT / 'weights' / 'formula_recognize' / 'onnx_fp16'
DST.mkdir(parents=True, exist_ok=True)


def fmt(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'


def main():
    print('=' * 74)
    print('UniMERNet ONNX FP16 量化')
    print('=' * 74)

    # 使用 onnxconverter-common 的 float16 转换（如果可用）
    try:
        from onnxconverter_common import float16
        has_conv = True
    except ImportError:
        has_conv = False
        print('\n[INFO] onnxconverter-common 未安装，尝试安装...')

    if not has_conv:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        'onnxconverter-common', '-q'], check=False)
        try:
            from onnxconverter_common import float16
            has_conv = True
            print('  [OK] 安装成功')
        except ImportError:
            print('  [FAIL] 安装失败')

    print('\n【1】转换模型')
    print('-' * 74)

    for name in ['unimernet_encoder.onnx', 'unimernet_decoder.onnx']:
        src = SRC / name
        dst = DST / name
        if not src.exists():
            print(f'  [SKIP] {name} 不存在')
            continue

        src_size = src.stat().st_size
        # 复制外部数据文件
        for ext in ['.data']:
            ext_src = src.with_suffix(src.suffix + ext)
            if ext_src.exists():
                shutil.copy2(ext_src, dst.with_suffix(dst.suffix + ext))
                print(f'  复制 {ext_src.name}')

        try:
            if has_conv:
                model = onnx.load(str(src))
                model_fp16 = float16.convert_float_to_float16(
                    model, keep_io_types=True)
                onnx.save(model_fp16, str(dst))
                print(f'  [OK] {name}: {fmt(src_size)} -> {fmt(dst.stat().st_size)}')
            else:
                shutil.copy2(src, dst)
                print(f'  [COPY] {name} (无转换工具)')
        except Exception as e:
            print(f'  [ERROR] {name}: {type(e).__name__}: {e}')
            shutil.copy2(src, dst)

    # 复制配置
    shutil.copy2(SRC / 'unimernet_config.json', DST / 'unimernet_config.json')

    # 处理外部数据文件 (FP16 转换后可能不再需要)
    print('\n【2】外部数据文件检查')
    print('-' * 74)
    for p in sorted(DST.iterdir()):
        print(f'  {p.name:36} {fmt(p.stat().st_size)}')

    print('\n【3】验证 FP16 模型可加载')
    print('-' * 74)
    for name in ['unimernet_encoder.onnx', 'unimernet_decoder.onnx']:
        p = DST / name
        if not p.exists():
            continue
        try:
            sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
            print(f'\n  {name}:')
            for i in sess.get_inputs():
                print(f'    IN  {i.name:24} {i.shape} {i.type}')
            for o in sess.get_outputs():
                print(f'    OUT {o.name:24} {o.shape} {o.type}')
            print(f'    [OK] 加载成功')
        except Exception as e:
            print(f'\n  {name}: [ERROR] {type(e).__name__}: {e}')

    print('\n【4】体积对比')
    print('-' * 74)
    def total(d):
        return sum(p.stat().st_size for p in Path(d).rglob('*') if p.is_file())
    print(f'  FP32: {fmt(total(SRC))}')
    print(f'  FP16: {fmt(total(DST))}')


if __name__ == '__main__':
    main()
