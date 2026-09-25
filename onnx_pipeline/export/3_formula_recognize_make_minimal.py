#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建 UniMERNet 模型代码的最小隔离副本
=====================================

只复制推理所需的模型文件，重写 __init__.py 以避免拉入
训练依赖 (iopath / datasets / tasks 等)。
"""

import os
import shutil
from pathlib import Path
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
# 官方源码 (已随导出链路保留在 weights 下)
SRC = ROOT / 'weights' / 'formula_recognize' / 'unimernet_src' / 'unimernet'
DST = ROOT / 'weights' / 'formula_recognize' / 'unimernet_code'


def main():
    print('=' * 74)
    print('创建 UniMERNet 最小代码副本')
    print('=' * 74)

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # 只需要的文件
    files = [
        ('models/unimernet/configuration_unimernet_encoder.py',
         'unimernet/configuration_unimernet_encoder.py'),
        ('models/unimernet/configuration_unimernet_decoder.py',
         'unimernet/configuration_unimernet_decoder.py'),
        ('models/unimernet/modeling_unimernet_encoder.py',
         'unimernet/modeling_unimernet_encoder.py'),
        ('models/unimernet/modeling_unimernet_decoder.py',
         'unimernet/modeling_unimernet_decoder.py'),
        ('models/unimernet/encoder_decoder.py',
         'unimernet/encoder_decoder.py'),
        ('models/unimernet/processor.py',
         'unimernet/processor.py'),
        ('models/unimernet/utils.py',
         'unimernet/utils.py'),
    ]

    print('\n复制文件:')
    for src_rel, dst_rel in files:
        src = SRC / src_rel
        dst = DST / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  [OK] {dst_rel}  ({src.stat().st_size:,} bytes)')
        else:
            print(f'  [MISSING] {src_rel}')

    # 重写 __init__.py (空)
    (DST / '__init__.py').write_text('', encoding='utf-8')
    (DST / 'unimernet' / '__init__.py').write_text('', encoding='utf-8')
    print('\n  [OK] 写入空的 __init__.py')

    # 检查 encoder_decoder.py 的导入，看是否需要修正
    ed = DST / 'unimernet' / 'encoder_decoder.py'
    text = ed.read_text(encoding='utf-8')
    print('\n--- encoder_decoder.py 的 unimernet 内部导入 ---')
    for i, ln in enumerate(text.split('\n'), 1):
        s = ln.strip()
        if s.startswith(('from unimernet', 'import unimernet',
                         'from .', 'from ..')):
            print(f'  L{i}: {s}')

    print(f'\n输出目录: {DST}')


if __name__ == '__main__':
    main()
