from tornado.web import RequestHandler

from concurrent.futures.thread import ThreadPoolExecutor
import time

from config import CALL_LIMIT


# TODO: whether the image file is satisfied with conditions
def check_image_data(img):
    pass


class OCRHandler(RequestHandler):
    executor = ThreadPoolExecutor(max_workers=2)
    n_limit = CALL_LIMIT
    n_cur = 0
    start_time = time.time()

    def check_authority(self):
        self.n_limit = CALL_LIMIT + int((time.time() - self.start_time) / 3600)
        if self.n_cur <= self.n_limit:
            self.n_cur += 1
            return True
        else:
            return False

    def data_received(self, chunk):
        pass
