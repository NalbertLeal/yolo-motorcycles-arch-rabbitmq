import struct

import numpy as np

def serialize(name: str, frame: np.ndarray) -> bytes:
    (height, width, channels) = frame.shape
    frame_bytes = frame.tobytes()
    return struct.pack(f'30s H H H {len(frame_bytes)}s', name, height, width, channels, frame_bytes)

def deserialize(name: str, frame_package: bytes) -> np.ndarray:
    frame_bytes_size = len(frame_package) - 36
    (height, width, channels, raw_frame) = struct.unpack(f'30s H H H {frame_bytes_size}s', frame_package)
    return name.rstrip('\x00'), np.frombuffer(raw_frame, dtype=np.int8).reshape((height, width, channels))