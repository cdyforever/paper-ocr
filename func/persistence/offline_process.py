"""
    OffLine process for receiving a batch of OCR tasks
    will be replaced by Redis in future

    Author: Chen Yu
    Date  : 7/29/2021
"""

from config import PORT

import requests
from multiprocessing import Process


class OffLineOCRProcess(Process):

    def __init__(self, task_queue, result_dict):
        super(OffLineOCRProcess, self).__init__()
        self.task_queue = task_queue
        self.result_dict = result_dict

    def run(self):
        while True:
            if self.task_queue.qsize() > 0:
                task_id, img_url = self.task_queue.get()
                img_bytes = requests.get(img_url).content
                data = {
                    'image': img_bytes
                }
                url = 'http://localhost:' + str(PORT) + '/photo_ocr'
                response_json = requests.post(url, files=data).json()
                result = response_json['result']
                cost_time = response_json['cost_time']
                is_succeed = True if response_json['code'] == 1 else False
                self.result_dict[str(task_id)] = {
                    'result': result,
                    'cost_time': cost_time,
                    'is_succeed': is_succeed
                }



