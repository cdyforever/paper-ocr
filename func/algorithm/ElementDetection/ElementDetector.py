from .picture.PictureDetector import PictureDetector
from .table.TableDetector import TableDetector


class PictureElementDetector(object):

    def __init__(self, model_path, device='cpu'):
        self.picture_detector = PictureDetector(model_path, device)

    def predict(self, img):
        pic_imgs, pic_boxes, img = self._pic_detect(img)
        return pic_imgs, pic_boxes, img

    def _pic_detect(self, img):
        pic_boxes = self.picture_detector.detect(img)
        pic_imgs = []
        for pic_box in pic_boxes:
            x1, y1, x2, y2 = pic_box
            pic_imgs.append(img[y1:y2, x1:x2].copy())
            img[y1:y2, x1:x2] = (255, 255, 255)
        return pic_imgs, pic_boxes, img


class TableElementDetector(object):

    # def __init__(self, model_dir, device='cpu'):
    #     self.table_detector = TableDetector(model_dir, device)
    def __init__(self, model_dir):
        self.table_detector = TableDetector(model_dir)

    def predict(self, img):
        table_imgs, table_lines, img = self._table_detect(img)
        return table_imgs, table_lines, img

    def _table_detect(self, img):
        table_lines = self.table_detector.detect(img, None)
        table_imgs = []
        for table_line in table_lines:
            b = table_line[2]
            for i in range(4):
                b[i] = int(b[i])
            table_imgs.append(img[b[1]:b[3], b[0]:b[2]].copy())
            img[b[1]:b[3], b[0]:b[2]] = (255, 255, 255)
        return table_imgs, table_lines, img











    # @staticmethod
    # def verticalToHorizontal(img):
    #     ih, iw = img.shape[:2]
    #     gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 13)
    #     y, x = np.where(binary_img < 10)
    #     if len(x) < 10:
    #         return img
    #     x1 = np.min(x)
    #     x2 = np.max(x)
    #     y1 = np.min(y)
    #     y2 = np.max(y)
    #     x1 = max(x1 - 1, 0)
    #     x2 = min(x2 + 1, iw)
    #     y1 = max(y1 - 1, 0)
    #     y2 = min(y2 + 1, ih)
    #     h = y2 - y1
    #     w = x2 - x1
    #     sub_img = img[y1:y2, x1:x2]
    #     if h > w:
    #         mat1 = binary_img[y1:y2, x1:x2]
    #         mat1 = (255 - mat1) / 255
    #         col_sum = np.sum(mat1, axis=1)
    #         l = len(col_sum)
    #         i = 0
    #         segment_list = []
    #         max_h = -1
    #         while i < l:
    #             if col_sum[i] > 1:
    #                 s0 = i
    #                 while i < l and col_sum[i] > 1:
    #                     i += 1
    #                 segment_list.append((s0, i))
    #                 max_h = max(max_h, i - s0)
    #             i += 1
    #         # 合并上下结构的字 （待）
    #         # 旋转括号等小型字符 （待）
    #         p_lr = int(0.1 * w)
    #         joint_img = None
    #         for e in segment_list:
    #             img_t = sub_img[e[0]:e[1], :]
    #             h_t = e[1] - e[0]
    #             p_h = max_h - h_t
    #             p_top = int(p_h / 2)
    #             p_down = p_h - p_top
    #             img_t = cv.copyMakeBorder(img_t, p_top, p_down, p_lr, p_lr, cv.BORDER_CONSTANT, value=(255, 255, 255))
    #             if joint_img is None:
    #                 joint_img = img_t
    #             else:
    #                 joint_img = np.hstack((joint_img, img_t))
    #         return joint_img
    #     else:
    #         return sub_img
    #
    # @staticmethod
    # def cell_dectection(row_lines, col_lines, padding=5):
    #     min_x = min(np.min(row_lines[:, [0, 2]]), np.min(col_lines[:, [0, 2]]))
    #     min_y = min(np.min(row_lines[:, [1, 3]]), np.min(col_lines[:, [1, 3]]))
    #     row_lines[:, [0, 2]] -= min_x
    #     row_lines[:, [1, 3]] -= min_y
    #     col_lines[:, [0, 2]] -= min_x
    #     col_lines[:, [1, 3]] -= min_y
    #     row_lines = np.int32(row_lines)
    #     col_lines = np.int32(col_lines)
    #     max_x = max(np.max(row_lines[:, [0, 2]]), np.max(col_lines[:, [0, 2]]))
    #     max_y = max(np.max(row_lines[:, [1, 3]]), np.max(col_lines[:, [1, 3]]))
    #     white_img = np.ones([max_y + 2 * padding, max_x + 2 * padding], dtype=np.uint8) * 255
    #     for e in row_lines:
    #         cv.line(white_img, (e[0] + padding, e[1] + padding), (e[2] + padding, e[3] + padding), (0, 0, 0),
    #                 thickness=3)
    #     for e in col_lines:
    #         cv.line(white_img, (e[0] + padding, e[1] + padding), (e[2] + padding, e[3] + padding), (0, 0, 0),
    #                 thickness=3)
    #     n, labels = cv.connectedComponents(white_img)
    #     area = (max_y + 2 * padding) * (max_x + 2 * padding)
    #     box_list = []
    #     for i in range(n):
    #         if i == 0:
    #             continue
    #         y, x = np.where(labels == i)
    #         point = np.asarray([y, x])
    #         rect = cv.minAreaRect(point.T)
    #         box = cv.boxPoints(rect)
    #         x1 = np.min(box[:, 1])
    #         y1 = np.min(box[:, 0])
    #         x2 = np.max(box[:, 1])
    #         y2 = np.max(box[:, 0])
    #         if rect[1][0] * rect[1][1] > 0.95 * area:
    #             continue
    #         else:
    #             # box_list.append([x1 - padding + min_x, y1 - padding + min_y, x2 - padding + min_x, y2 - padding + min_y])
    #             # 避免与单元格边界重合
    #             box_list.append([x1 + padding, y1 + padding, x2 - padding, y2 - padding])
    #     box_info = []
    #     for e in box_list:
    #         tx1 = 0
    #         min_tx1 = 1e6
    #         tx2 = 0
    #         min_tx2 = 1e6
    #         ty1 = 0
    #         min_ty1 = 1e6
    #         ty2 = 0
    #         min_ty2 = 1e6
    #         for ind, row in enumerate(row_lines):
    #             t1 = abs((row[1] + row[3]) / 2 - e[1])
    #             t2 = abs((row[1] + row[3]) / 2 - e[3])
    #             if t1 < min_tx1:
    #                 tx1 = ind
    #                 min_tx1 = t1
    #             if t2 < min_tx2:
    #                 tx2 = ind
    #                 min_tx2 = t2
    #         for ind, col in enumerate(col_lines):
    #             t1 = abs((col[0] + col[2]) / 2 - e[0])
    #             t2 = abs((col[0] + col[2]) / 2 - e[2])
    #             if t1 < min_ty1:
    #                 ty1 = ind
    #                 min_ty1 = t1
    #             if t2 < min_ty2:
    #                 ty2 = ind
    #                 min_ty2 = t2
    #         e[0] += min_x
    #         e[1] += min_y
    #         e[2] += min_x
    #         e[3] += min_y
    #         box_info.append({"cols": list(range(ty1, ty2)), "rows": list(range(tx1, tx2)), "box": e})
    #     return box_info
    #
    # @staticmethod
    # def save_exception_img(img, path="./except/img"):
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
    #     name = time_str + str(np.random.randint(0, 1e6)).rjust(6, "0") + ".png"
    #     save_path = os.path.join(path, name)
    #     try:
    #         cv.imwrite(save_path, img)
    #     except:
    #         traceback.print_exc()
