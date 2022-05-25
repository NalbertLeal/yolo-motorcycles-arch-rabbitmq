from typing import Tuple

import numpy as np

from .base_encoder import int_to_bytes, str_to_bytes_with_size
from .base_decoder import bytes_to_int, bytes_to_str_with_size
from .opencv_frame import encode as opencv_frame_encode, decode as opencv_frame_decode

def encode(name: str, bbox: np.ndarray, frame: np.ndarray) -> bytes:
    name_bytes = str_to_bytes_with_size(name, 30)

    (number_box, _) = bbox.shape
    number_box_bytes = int_to_bytes(number_box, bytes_size=2)
    bbox_bytes_number = bbox.dtype.itemsize * number_box
    bbox_bytes_number_bytes = int_to_bytes(bbox_bytes_number, bytes_size=4)
    bboxes_bytes = bbox.tobytes()

    frame_bytes = opencv_frame_encode('', frame, name_size=0)

    return name_bytes + number_box_bytes + bbox_bytes_number_bytes + bboxes_bytes + frame_bytes

def decode(bboxes_frame_bytes: bytes, bboxes_type=np.float64, frame_type=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    name = bytes_to_str_with_size(bboxes_frame_bytes[:30])

    number_box = bytes_to_int(bboxes_frame_bytes[30:32])
    bbox_bytes_number = bytes_to_int(bboxes_frame_bytes[32:36]) * 6
    bboxes_bytes = bboxes_frame_bytes[36:36+bbox_bytes_number]
    bboxes = None
    if len(bboxes_bytes) > 0:
        bboxes = np.frombuffer(bboxes_bytes, dtype=bboxes_type).reshape((number_box, 6))

    frame_bytes = bboxes_frame_bytes[36+bbox_bytes_number:]
    _, frame = opencv_frame_decode(frame_bytes, array_type=frame_type, name_size=0)

    return name, bboxes, frame