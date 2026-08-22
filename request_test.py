import requests
import os
import json


def test_photo_ocr():
    url = 'http://localhost:9911/photo_subject'
    files = {'image': open('tmp/1294.jpg', 'rb')}
    response = requests.post(url, files=files)
    return response


IMAGE_DIR = "http://123.60.217.149:9911/image/"
IMAGE = "test.png"
image_url = os.path.join(IMAGE_DIR, IMAGE)


def test_rec():
    url = 'http://192.168.1.190:8080/rec'
    data = {
        'img_url': image_url,
        # 'coordinates': None,
        # 'mode': 2,
        # 'img_type': None
    }
    response = requests.post(url, data)
    return response


def test_query_recognition_result(task_id):
    url = 'http://192.168.1.190:8080/recognition_result_query'
    data = {
        'task_id': task_id,
    }
    response = requests.post(url, data)
    return response


if __name__ == '__main__':

    res_http = test_photo_ocr()
    res = res_http.json()
    print(res['result'])

    # res_http = test_rec()
    # res = res_http.json()
    # print(res['task_id'])
    #
    # res_http = test_query_recognition_result(2)
    # res = res_http.json()
    # print(res)
