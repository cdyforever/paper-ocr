from .ocr_handler import OCRHandler

import json
import numpy as np
from urllib import parse


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class QueryOffLineHandler(OCRHandler):

    def initialize(self, result_dict):
        self.result_dict = result_dict

    async def post(self):
        task_id_str = self.get_body_argument("task_id", "-1")
        task_id = str(task_id_str)
        if task_id in self.result_dict.keys():
            res = self.result_dict.pop(task_id)
            if res['is_succeed']:
                response = {"code": 1, "result": parse.quote(json.dumps(res['result'], cls=MyEncoder)), "cast_time": res['cost_time']}
            else:
                response = {"code": -1, "result": "", "cast_time": 0}
        else:
            response = {"code": 0, "result": "", "cast_time": 0}
        self.write(json.dumps(response))
