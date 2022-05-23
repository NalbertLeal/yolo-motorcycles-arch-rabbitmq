import struct

import numpy as np

def serialize(name: str, bboxes: np.ndarray, frame: np.ndarray) -> bytes:
    (height, width, channels) = frame.shape
    (bboxes_number, _) = bboxes.shape

    bboxes_bytes = bboxes.tobytes()
    frame_bytes = frame.tobytes()
    return struct.pack(f'30s H H H H {len(frame_bytes)}s', name, height, width, channels, bboxes_number, bboxes_bytes, frame_bytes)

def deserialize(bboxes_package: bytes) -> np.ndarray:
    name = struct.unpack('30s', bboxes_package[:30])
    height = struct.unpack('H', bboxes_package[30:32])
    width = struct.unpack('H', bboxes_package[32:34])
    channels = struct.unpack('H', bboxes_package[34:36])
    bboxes_number = struct.unpack('H', bboxes_package[36:38])

    float64_size = 8
    bbox_size = float64_size * 6
    raw_bboxes_size = (bboxes_number * bbox_size) + 38
    raw_bboxes = struct.unpack(f'{raw_bboxes_size}s', bboxes_package[8:raw_bboxes_size])

    raw_frame_size = len(bboxes_package) - raw_bboxes_size
    raw_frame = struct.unpack(f'{raw_frame_size}s', bboxes_package[raw_bboxes_size:])
    
    
    bboxes = np.frombuffer(raw_bboxes, dtype=np.float64).reshape((bboxes_number, 6))
    frame = np.frombuffer(raw_frame, dtype=np.float32).reshape((height, width, channels))
    return name.rstrip('\x00'), bboxes, frame