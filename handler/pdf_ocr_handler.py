from .ocr_handler import OCRHandler

from tornado import ioloop

import cv2
import json
import numpy as np


class PDFOCRHandler(OCRHandler):

    def initialize(self, paper_ocr):
        self.paper_ocr = paper_ocr

    async def post(self):
        img_http = self.request.files['image'][0]
        img_bytes = img_http.get('body')
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if super().check_authority():
            try:
                combined_msg, _, cost_time = await ioloop.IOLoop.current().run_in_executor(
                    super().executor, self.paper_ocr.predict, img, 'pdf')
                response = {
                    'code': 1,
                    'result': combined_msg,
                    'img_name': img_http.get('filename'),
                    'cost_time': cost_time
                }
            except:
                response = {
                    'code': -1,
                    'result': '',
                    'img_name': img_http.get('filename'),
                    'cost_time': 0
                }
            self.write(json.dumps(response))
        else:
            response = {
                'code': -2,
                'result': '',
                'img_name': img_http.get('filename'),
                'cost_time': 0
            }
            self.write(json.dumps(response))
