#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载并分析 UniMERNet-tiny
=========================

先看清架构再决定导出策略。
"""

import os
import json
import requests
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
OUT = Path.cwd() / 'weights' / 'formula_recognize' / 'unimernet_tiny'
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def get(url, **kw):
    return requests.get(url, timeout=60, headers=HEADERS, **kw)


def download_small(repo, path):
    """下载小文件 (配置等)"""
    url = f'https://huggingface.co/{repo}/resolve/main/{path}'
    try:
        r = get(url)
        if r.status_code != 200:
            print(f'  [FAIL] {path}: HTTP {r.status_code}')
            return None
        dest = OUT / Path(path).name
        dest.write_bytes(r.content)
        print(f'  [OK] {dest.name}  ({len(r.content):,} bytes)')
        return r.content
    except Exception as e:
        print(f'  [ERROR] {path}: {e}')
        return None


def main():
    print('=' * 74)
    print('UniMERNet-tiny 下载与架构分析')
    print('=' * 74)

    # 候选仓库
    repos = [
        'wanderkid/unimernet_tiny',
        'opendatalab/PDF-Extract-Kit-1.0',
    ]

    # ---------- 1. 下载配置 ----------
    print('\n【1】下载配置文件')
    print('-' * 74)
    cfg_files = ['config.json', 'preprocessor_config.json',
                 'tokenizer_config.json', 'generation_config.json',
                 'unimernet_tiny.yaml', 'configuration.json']
    for f in cfg_files:
        download_small('wanderkid/unimernet_tiny', f)

    # PDF-Extract-Kit 的版本
    print('\n  --- PDF-Extract-Kit 版本 ---')
    for f in ['models/MFR/unimernet_tiny/config.json',
              'models/MFR/unimernet_tiny/preprocessor_config.json',
              'models/MFR/unimernet_tiny/unimernet_tiny.yaml',
              'models/MFR/unimernet_tiny/tokenizer_config.json']:
        download_small('opendatalab/PDF-Extract-Kit-1.0', f)

    # ---------- 2. 打印配置内容 ----------
    print('\n【2】配置内容')
    print('-' * 74)
    for p in sorted(OUT.glob('*.json')) + sorted(OUT.glob('*.yaml')):
        print(f'\n  === {p.name} ===')
        try:
            txt = p.read_text(encoding='utf-8')
            if p.suffix == '.json':
                d = json.loads(txt)
                print(json.dumps(d, indent=2, ensure_ascii=False)[:2500])
            else:
                print(txt[:2500])
        except Exception as e:
            print(f'  (读取失败: {e})')

    # ---------- 3. tokenizer ----------
    print('\n【3】下载 tokenizer')
    print('-' * 74)
    for f in ['tokenizer.json', 'vocab.json', 'merges.txt',
              'special_tokens_map.json', 'added_tokens.json']:
        download_small('wanderkid/unimernet_tiny', f)

    # ---------- 4. 架构分析 ----------
    print('\n【4】架构分析')
    print('-' * 74)
    cfg = OUT / 'config.json'
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding='utf-8'))
        print('  config.json 顶层键:')
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'    {k}: <dict, {len(v)} keys>')
                for kk, vv in list(v.items())[:12]:
                    print(f'        {kk} = {vv}')
            else:
                print(f'    {k} = {v}')


if __name__ == '__main__':
    main()
