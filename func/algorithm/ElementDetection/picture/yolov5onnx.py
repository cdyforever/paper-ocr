import os
import logging
from typing import List
import onnxruntime as rt
import numpy as np
import cv2
import torch
from .postprocessing import postprocessing


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class YOLOv5ONNX:
    _anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]
    def __init__(
            self,
            model_path: str = None,
            nms: float = 0.45,
            conf: float = 0.45
    ):
        path_to_model = model_path
        try:
            self._session = rt.InferenceSession(path_to_model)
            logger.info("Photo Detect ONNX session successfully started")
        except Exception as e:
            logger.info(f"Failed to init Photo Detect ONNX session. Error: {e}")
            raise e
        self._NMS = nms
        self._CONF = conf
        self._input_name = self._session.get_inputs()[0].name
        self._batch_size = self._session.get_inputs()[0].shape[0]
        self._img_height = self._session.get_inputs()[0].shape[2]
        self._img_width = self._session.get_inputs()[0].shape[3]
        self._classes = ["photo"]


    def infer(self, image: np.ndarray) -> List[list]:
        original_image = image.copy()
        preprocessed_img = self._preprocess_image(image)
        outputs = self._session.run(None, {self._input_name: preprocessed_img})
        detections = self._apply_nms_conf_thresholds(outputs)[0]  # batch 1
        if detections is not None:
            scaled_detections = self._postprocess_detections(
                detections, original_image
            )
            return postprocessing(scaled_detections, original_image)
        else:
            return []

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image, *_ = YOLOv5ONNX.letterbox(
            image, (self._img_height, self._img_width)
        )
        image = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        image = np.expand_dims(image, 0)
        image /= 255.0
        return image

    def _apply_nms_conf_thresholds(self, outputs: list) -> List[torch.Tensor]:
        boxs = []
        a = torch.tensor(YOLOv5ONNX._anchors).float().view(3, -1, 2)
        anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
        for index, out in enumerate(outputs):
            out = torch.from_numpy(out)
            feature_w = out.shape[2]
            feature_h = out.shape[3]
            # Feature map corresponds to the original image zoom factor
            stride_w = int(self._img_width / feature_w)
            stride_h = int(self._img_height / feature_h)
            grid_x, grid_y = np.meshgrid(np.arange(feature_w),
                                         np.arange(feature_h))
            # cx, cy, w, h
            pred_boxes = torch.FloatTensor(out[..., :4].shape)
            pred_boxes[..., 0] = (torch.sigmoid(
                out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
            pred_boxes[..., 1] = (torch.sigmoid(
                out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
            pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * \
                                   anchor_grid[index]  # wh
            conf = torch.sigmoid(out[..., 4])
            pred_cls = torch.sigmoid(out[..., 5:])
            output = torch.cat(
                (pred_boxes.view(self._batch_size, -1, 4),
                conf.view(self._batch_size, -1, 1),
                pred_cls.view(self._batch_size, -1, len(self._classes))), -1
            )
            boxs.append(output)
        outputx = torch.cat(boxs, 1)
        batch_detections = YOLOv5ONNX.w_non_max_suppression(
            outputx, len(self._classes), self._CONF, self._NMS
        )
        return batch_detections

    def _postprocess_detections(
            self,
            detections: torch.Tensor,
            original_image: np.ndarray
    ) -> List[list]:
        labels = [self._classes[int(i)] for i in detections[..., -1]]
        boxes = YOLOv5ONNX.scale_coords(
            img1_shape=(self._img_height, self._img_width),
            coords=detections[..., :4],
            img0_shape=original_image.shape[:2]
        ).round()
        confs = detections[..., 4].tolist()
        out = []
        for label, box, conf in zip(labels, boxes, confs):
            # out.append(
            #    [label, *[int(e) for e in box.tolist()], round(conf, 4)]
            # )
            out.append([*[int(e) for e in box.tolist()]])
        return out

    @staticmethod
    def letterbox(
            img: np.ndarray,
            new_shape: tuple = (640, 640),
            color: tuple = (114, 114, 114),
            auto: bool = False,
            scale_fill: bool = False,
            scaleup: bool = True
    ) -> tuple:
        # Resize image to a 32-pixel-multiple rectangle
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # wh padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            # width, height ratios
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return img, ratio, (dw, dh)

    @staticmethod
    def w_non_max_suppression(
            prediction, num_classes, conf_thres=0.5, nms_thres=0.4
    ) -> List[torch.Tensor]:
        # CREDITS - https://github.com/ultralytics/yolov5/issues/343
        # box_corner = prediction.new(prediction.shape)
        box_corner = torch.FloatTensor(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            if not image_pred.size(0):
                continue

            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + num_classes], 1, keepdim=True
            )
            # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
            )
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                _, conf_sort_index = torch.sort(
                    detections_class[:, 4], descending=True
                )
                detections_class = detections_class[conf_sort_index]
                max_detections = []
                while detections_class.size(0):
                    max_detections.append(detections_class[0].unsqueeze(0))
                    if len(detections_class) == 1:
                        break
                    ious = YOLOv5ONNX.w_bbox_iou(
                        max_detections[-1], detections_class[1:]
                    )
                    detections_class = detections_class[1:][ious < nms_thres]
                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None \
                    else torch.cat(
                    (output[image_i], max_detections)
                )
        return output

    @staticmethod
    def w_bbox_iou(box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = (box1[:, 0] - box1[:, 2] / 2,
                            box1[:, 0] + box1[:, 2] / 2)
            b1_y1, b1_y2 = (box1[:, 1] - box1[:, 3] / 2,
                            box1[:, 1] + box1[:, 3] / 2)
            b2_x1, b2_x2 = (box2[:, 0] - box2[:, 2] / 2,
                            box2[:, 0] + box2[:, 2] / 2)
            b2_y1, b2_y2 = (box2[:, 1] - box2[:, 3] / 2,
                            box2[:, 1] + box2[:, 3] / 2)
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = (box1[:, 0], box1[:, 1],
                                          box1[:, 2], box1[:, 3])
            b2_x1, b2_y1, b2_x2, b2_y2 = (box2[:, 0], box2[:, 1],
                                          box2[:, 2], box2[:, 3])
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (
                torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) *
                torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        )
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    @staticmethod
    def scale_coords(img1_shape, coords, img0_shape):
        def clip_coords(boxes, img_shape):
            # Clip bounding xyxy bounding boxes to image shape (height, width)
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2

        # Rescale coords (xyxy) from img1_shape to img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, \
                  (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        clip_coords(coords, img0_shape)
        return coords