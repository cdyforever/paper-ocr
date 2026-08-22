from .ocr_handler import OCRHandler

from tornado import ioloop

import cv2
import json
import numpy as np


class PhotoOCRTestHandler(OCRHandler):

    def initialize(self, paper_ocr, data_collector):
        self.paper_ocr = paper_ocr
        self.data_collector = data_collector

    async def post(self):
        img_http = self.request.files['image'][0]
        img_bytes = img_http.get('body')
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        try:
            combined_msg, page_result, cost_time = await ioloop.IOLoop.current().run_in_executor(
                super().executor, self.paper_ocr.predict, img, 'photo')
            img_save_name = self.data_collector.collect_ocr_result(img, combined_msg, page_result)
            response = {
                'code': 1,
                'img_name': img_http.get('filename'),
                'result': combined_msg,
                'save_name': img_save_name,
                'cost_time': cost_time
            }
        except:
            response = {
                'code': -1,
                'img_name': img_http.get('filename'),
                'result': '',
                'save_name': '',
                'cost_time': 0
            }
        self.write(json.dumps(response))
