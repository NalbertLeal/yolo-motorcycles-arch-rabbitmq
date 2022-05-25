import cv2
import numpy as np

YOLOv5_IMAGE_HEIGHT = 640
YOLOv5_IMAGE_WIDTH = 640

def preprocess_image_to_onnx(frame: np.ndarray) -> np.ndarray:
    '''
        Prepare image to an onnx model process.
    '''
    img = resize_image(frame, YOLOv5_IMAGE_HEIGHT, YOLOv5_IMAGE_WIDTH)
    img = to_onnx_format(img, YOLOv5_IMAGE_HEIGHT, YOLOv5_IMAGE_WIDTH)
    img = normalize(img)
    # img = add_batch_dimension(img)
    return img

def resize_image(img: np.ndarray, height: int=640, width: int=640) -> np.ndarray:
	return cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_LINEAR)

def normalize(img: np.ndarray) -> np.ndarray:
	img /= 255
	return img

def to_onnx_format(img: np.ndarray, height: int=640, width: int=640) -> np.ndarray:
	return np.ascontiguousarray(img).astype(np.float32).transpose(2, 0, 1)

def add_batch_dimension(img: np.ndarray) -> np.ndarray:
	return np.expand_dims(img, axis=0)