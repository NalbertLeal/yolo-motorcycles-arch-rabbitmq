import numpy as np

from .base_encoder import int_to_bytes, str_to_bytes_with_size 
from .base_decoder import bytes_to_int, bytes_to_str_with_size

def encode(name: str, frame: np.ndarray, name_size=30) -> bytes:
    name_bytes = str_to_bytes_with_size(name, name_size)

    (height, width, channels) = frame.shape
    height_bytes = int_to_bytes(height, bytes_size=2)
    width_bytes = int_to_bytes(width, bytes_size=2)
    channels_bytes = int_to_bytes(channels, bytes_size=2)
    frame_bytes = frame.tobytes()
    return name_bytes + height_bytes + width_bytes + channels_bytes + frame_bytes

def decode(frame_bytes, array_type=np.uint8, name_size=30):
    name = bytes_to_str_with_size(frame_bytes[:name_size])
    height = bytes_to_int(frame_bytes[name_size:name_size+2])
    width = bytes_to_int(frame_bytes[name_size+2:name_size+4])
    channels = bytes_to_int(frame_bytes[name_size+4:name_size+6])
    return name, np.frombuffer(frame_bytes[name_size+6:], dtype=array_type).reshape((height, width, channels))