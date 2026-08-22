"""
    Preparing data for developing algorithm of text-structured
    Author: Chen Yu
    Date  : 7/12/2021
"""

import os
import json
import shutil
import requests
from tqdm import tqdm
from urllib import parse


SOURCE_DIR = 'tmp'
RESULT_DIR = 'json'

OCR_URL = 'http://localhost:9911/photo_ocr'


def prepare_data(source_dir=SOURCE_DIR,
                 result_dir=RESULT_DIR
                 ):
    os.makedirs(result_dir, exist_ok=True)

    src_paths = []
    for root, subdir, srcs in os.walk(source_dir):
        for src in srcs:
            if src.endswith('.jpg') or src.endswith('.png'):
                src_paths.append(os.path.join(root, src))

    for src_path in tqdm(src_paths):
        mark = os.path.basename(src_path)[:-4]
        files = {'image': open(src_path, 'rb')}
        result = requests.post(OCR_URL, files=files).json()['result']

        with open(os.path.join(result_dir, mark + '.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    prepare_data()












