from config import MATHPIX_AUTH

import cv2
import requests
import json
import base64

# TODO: global config
default_headers = {
    'Authorization': MATHPIX_AUTH,
    'User-Agent': "Mathpix Snip Windows App v02.05.0009",
    'Content-type': 'application/json',
    'Content-Length': '94926',
    'Connection': 'Keep-Alive',
    'HOST': "api.mathpix.com",
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN, en, *'
}

service = 'https://api.mathpix.com/v1/snips'


class MathPixAPI:

    def __init__(self):
        return

    def _to_base64(self, im):
        imenc = cv2.imencode('.jpg', im)[1]
        return "data:picture/jpg;base64," + base64.b64encode(imenc).decode()

    def _from_mathpix(self, args, headers=None, timeout=1000):
        if headers is None:
            headers = default_headers
        try:
            r = requests.post(service, data=json.dumps(args), headers=headers, timeout=timeout)
            res = json.loads(r.text)
            if 'text' not in res.keys():
                return {'text': None}
            else:
                return res
        except:
            return {'text': None}

    def get_result(self, im):
        b64str = self._to_base64(im)
        req_dict = {
            "config": {
                "math_display_delimiters": [
                    "\n$$\n",
                    "\n$$\n"
                ],
                "math_inline_delimiters": [
                    "$",
                    "$"
                ],
                "ocr_version": 2,
                "rm_fonts": False
            },
            "metadata": {
                "count": 15,
                "input_type": "crop",
                "platform": "windows 10",
                "skip_recrop": True,
                "user_id": "34b45f95106241b382bbdd3bdbdc6f6c",
                "version": "snip.windows@02.07.0002"
            }
        }
        req_dict['src'] = b64str
        res = self._from_mathpix(req_dict)
        return res['text']


# if __name__ == "__main__":
#     from TextRecognition.ench.paddle_rec import TextRecognizer
#     api = MathPixAPI()
#     image_data = cv2.imread("../DIR_Textline/35_20.jpg")
#     #from TextLineDivision.TextLineSplitProcess import
#
#     paddle_rec = TextRecognizer(model_dir="../data/model/paddle")
#     res = paddle_rec.predic(image_data)
#
#     print(res)


    #print(api.get_result(image_data))
