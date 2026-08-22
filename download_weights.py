"""
    Download models weight automatically

    Author: Chen Yu
    Date  : 7/29/2021
"""

import os
import math
from tqdm import tqdm
import requests
from zipfile import ZipFile
import tempfile
import shutil


def _download(file_url, save_dir):
    file_name = os.path.basename(file_url)
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)

    response = requests.get(file_url, stream=True)
    file_size = math.ceil(int(response.headers["Content-Length"]) / 1024 + 0.5)
    with open(save_path, 'wb') as f:
        print('Downloading %s Size %dkB' % (file_name, file_size))
        for chunk in tqdm(iterable=response.iter_content(1024), total=file_size, unit='kB', desc=None):
            f.write(chunk)


def _extract(file_path, target_dir):
    extractor = ZipFile(file_path)
    extractor.extractall(target_dir)
    extractor.close()


def check_and_download_weights(file_url, target_dir):
    if not os.path.exists(os.path.join(target_dir, 'weights')):
        zip_file_name = os.path.basename(file_url)
        zip_file_dir = tempfile.mkdtemp()
        _download(file_url, zip_file_dir)
        zip_file_path = os.path.join(zip_file_dir, zip_file_name)
        _extract(zip_file_path, target_dir)
        os.remove(zip_file_path)