"""
    Canpoint OCR Server

    Author: Chen Yu
    Date  : 7/29/2021
"""

from func.algorithm import PaperOCR
from func.persistence import DataCollector
from func.persistence.offline_process import OffLineOCRProcess
from handler import PhotoOCRHandler, \
    PDFOCRHandler, \
    PhotoOCRTestHandler, \
    PhotoOCROffLineHandler, \
    QueryOffLineHandler, \
    PhotoSubjectHandler, \
    PDFSubjectHandler
from config import PORT

from tornado import web
from tornado import ioloop
import tornado.websocket
import tornado.httpserver
from tornado.options import define, options


import os

from multiprocessing import Queue, Manager

define("port", default=PORT, help="run on the given port", type=int)


class Application(web.Application):
    def __init__(self):
        handlers = [
            # formal interface for paper OCR
            (r'/photo_ocr', PhotoOCRHandler, dict(paper_ocr=paper_ocr)),
            (r'/pdf_ocr', PDFOCRHandler, dict(paper_ocr=paper_ocr)),
            # interface for extracting subjects
            (r'/photo_subject', PhotoSubjectHandler, dict(paper_ocr=paper_ocr)),
            (r'/pdf_subject', PDFSubjectHandler, dict(paper_ocr=paper_ocr)),
            # test interface for collecting and analysing data
            (r'/photo_ocr_test', PhotoOCRTestHandler, dict(paper_ocr=paper_ocr, data_collector=data_collector)),
            # old interface for canpoint platform
            (r'/rec', PhotoOCROffLineHandler, dict(task_queue=offline_task_queue)),
            (r'/recognition_result_query', QueryOffLineHandler, dict(result_dict=offline_result_dict)),
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates")
        )
        tornado.web.Application.__init__(self, handlers, **settings)


if __name__ == '__main__':

    paper_ocr = PaperOCR()
    data_collector = DataCollector()

    offline_task_queue = Queue()
    offline_result_dict = Manager().dict()
    op = OffLineOCRProcess(offline_task_queue, offline_result_dict)
    op.start()

    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(options.port)
    ioloop.IOLoop.instance().start()
