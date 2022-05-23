# functions from: https://github.com/BlueMirrors/cvu/blob/master/cvu/postprocess/nms/yolov5.py

import time
from typing import Callable, List, Tuple

import numpy as np

def nms_np(detections: np.ndarray, scores: np.ndarray, max_det: int,
           thresh: float) -> List[np.ndarray]:
    """Standard Non-Max Supression Algorithm for filter out detections.
    Args:
        detections (np.ndarray): bounding-boxes of shape num_detections,4
        scores (np.ndarray): confidence scores of each bounding box
        max_det (int): Maximum number of detections to keep.
        thresh (float): IOU threshold for NMS
    Returns:
        List[np.ndarray]: Filtered boxes.
    """
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # get boxes with more ious first
    order = scores.argsort()[::-1]

    # final output boxes
    keep = []

    while order.size > 0 and len(keep) < max_det:
        # pick maxmum iou box
        i = order[0]
        keep.append(i)

        # get iou
        ovr = get_iou((x1, y1, x2, y2), order, areas, idx=i)

        # drop overlaping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def get_iou(xyxy: Tuple[np.ndarray], order: np.ndarray, areas: np.ndarray, idx: int) -> float:
    """Helper function for nms_np to calculate IoU.
    Args:
        xyxy (Tuple[np.ndarray]): tuple of x1, y1, x2, y2 coordinates.
        order (np.ndarray): boxs' indexes sorted according to there
        confidence scores
        areas (np.ndarray): area of each box
        idx (int): base box to calculate iou for
    Returns:
        float: [description]
    """
    x1, y1, x2, y2 = xyxy
    xx1 = np.maximum(x1[idx], x1[order[1:]])
    yy1 = np.maximum(y1[idx], y1[order[1:]])
    xx2 = np.minimum(x2[idx], x2[order[1:]])
    yy2 = np.minimum(y2[idx], y2[order[1:]])

    max_width = np.maximum(0.0, xx2 - xx1 + 1)
    max_height = np.maximum(0.0, yy2 - yy1 + 1)
    inter = max_width * max_height

    return inter / (areas[idx] + areas[order[1:]] - inter)

def non_max_suppression_np(predictions: np.ndarray,
                           conf_thres: float = 0.25,
                           iou_thres: float = 0.45,
                           agnostic: bool = False,
                           multi_label: bool = False,
                           nms: Callable = nms_np) -> List[np.ndarray]:
    """Runs Non-Maximum Suppression (NMS used in Yolov5) on inference results.
    Args:
        predictions (np.ndarray): predictions from yolov inference
        conf_thres (float, optional): confidence threshold in range 0-1.
        Defaults to 0.25.
        iou_thres (float, optional): IoU threshold in range 0-1 for NMS filtering.
        Defaults to 0.45.
        agnostic (bool, optional): agnostic to width-height. Defaults to False.
        multi_label (bool, optional): apply Multi-Label NMS. Defaults to False.
        nms (Callable[[np.ndarray, np.ndarray, int, float], List[np.ndarray]]): Base NMS
        function to be applied. Defaults to nms_np.
    Returns:
        List[np.ndarray]: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Settings
    maximum_detections = 300
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after

    # number of classes > 1 (multiple labels per box (adds 0.5ms/img))
    multi_label &= (predictions.shape[2] - 5) > 1

    start_time = time.time()
    output = [np.zeros((0, 6))] * predictions.shape[0]
    confidences = predictions[..., 4] > conf_thres
    # print(confidences)

    # image index, image inference
    for batch_index, prediction in enumerate(predictions):

        # confidence
        prediction = prediction[confidences[batch_index]]
        # print(prediction)

        # If none remain process next image
        if not prediction.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        prediction = detection_matrix(prediction, multi_label, conf_thres)

        # Check shape; # number of boxes
        if not prediction.shape[0]:  # no boxes
            continue

        # excess boxes
        if prediction.shape[0] > max_nms:
            prediction = prediction[np.argpartition(-prediction[:, 4],
                                                    max_nms)[:max_nms]]

        # Batched NMS
        classes = prediction[:, 5:6] * (0 if agnostic else max_wh)
        indexes = nms(prediction[:, :4] + classes, prediction[:, 4],
                      maximum_detections, iou_thres)

        # pick relevant boxes
        output[batch_index] = prediction[indexes, :]

        # check if time limit exceeded
        if (time.time() - start_time) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def detection_matrix(predictions: np.ndarray, multi_label: bool,
                     conf_thres: float) -> np.ndarray:
    """Prepare Detection Matrix for Yolov5 NMS
    Args:
        predictions (np.ndarray): one batch of predictions from yolov inference.
        multi_label (bool): apply Multi-Label NMS.
        conf_thres (float): confidence threshold in range 0-1.
    Returns:
        np.ndarray: detections matrix nx6 (xyxy, conf, cls).
    """

    # Compute conf = obj_conf * cls_conf
    predictions[:, 5:] *= predictions[:, 4:5]

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(predictions[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (predictions[:, 5:] > conf_thres).nonzero().T
        predictions = np.concatenate(
            (box[i], predictions[i, j + 5, None], j[:, None].astype('float')),
            1)

    # best class only
    else:
        j = np.expand_dims(predictions[:, 5:].argmax(axis=1), axis=1)
        conf = np.take_along_axis(predictions[:, 5:], j, axis=1)

        # print(box)
        # print(conf)

        predictions = np.concatenate((box, conf, j.astype('float')), 1)[conf.reshape(-1) > conf_thres]

    return predictions

def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    Args:
        xywh (np.ndarray): array of 4 float [center_x, center_y, width, height]
    Returns:
        np.ndarray: array of 4 float [x1, y1, x2, y2] where (x1,y1)==top-left
        and (x2,y2)==bottom-right.
    """
    xyxy = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return xyxy