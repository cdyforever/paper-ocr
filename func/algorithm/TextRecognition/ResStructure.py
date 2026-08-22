"""
    A formal structure for transmitting recognition results of text segments

    Author: Chen Yu
    Date  : 7/29/2021
"""

from collections.abc import Sequence


class TextLineRecognizeResult(Sequence):

    def __init__(self, text_line_pos, is_completed_by_mathpix=False):
        self.is_by_mathpix_completed = is_completed_by_mathpix
        self.texts_str = []     # result str
        self.texts_type = []    # ench or formula
        self.texts_label = []   # mathpix or native
        self.texts_pos = []     # [x1,y1,x2,y2] in image
        self.text_line_pos = text_line_pos
        super().__init__()

    def push_back(self, text_str, text_type, text_label, text_pos):
        self.texts_str.append(text_str)
        self.texts_type.append(text_type)
        self.texts_label.append(text_label)
        self.texts_pos.append(text_pos)

    def __getitem__(self, idx):
        return self.texts_str[idx], \
               self.texts_type[idx], \
               self.texts_label[idx], \
               self.texts_pos[idx]

    def __len__(self):
        return len(self.texts_str)
