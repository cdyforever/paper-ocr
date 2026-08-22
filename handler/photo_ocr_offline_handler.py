from .ocr_handler import OCRHandler

import json


class PhotoOCROffLineHandler(OCRHandler):
    task_id = 0

    def initialize(self, task_queue):
        self.task_queue = task_queue

    async def post(self):
        img_url = self.get_body_argument('img_url')
        task_id = self.get_task_id()
        self.task_queue.put((task_id, img_url))
        response = {
            "task_id": task_id,
            "code": 0
        }
        self.write(json.dumps(response))

    @staticmethod
    def get_task_id():
        PhotoOCROffLineHandler.task_id += 1
        return PhotoOCROffLineHandler.task_id
