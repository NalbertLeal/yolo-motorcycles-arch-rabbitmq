import os

import numpy as np
import onnxruntime as onnx
import pydantic

from yolo_motorcycles.model import onnx_posprocess

SUPPORTED_DEVICES = {
	'cpu': {
		'provider': 'CPUExecutionProvider',
	},
	'gpu': {
		'provider': 'CUDAExecutionProvider'
	},
	'tpu': {
		'provider': 'TensorrtExecutionProvider',
	}
}

class YOLOv5Onnx(pydantic.BaseModel):
	device: str
	session: onnx.InferenceSession

	class Config:
		arbitrary_types_allowed = True

def new_YOLOv5Onnx(weights: str, device: str) -> YOLOv5Onnx:
	if not _is_acceptable_device(device):
		raise BaseException('Device not acceptable')
	if not _does_weights_file_exists(weights):
		raise BaseException('Weights file not found')
	session = _create_session(weights, device)
	return YOLOv5Onnx(device=device, session=session)

def _is_acceptable_device(device: str) -> bool:
	return device in SUPPORTED_DEVICES.keys()

def _does_weights_file_exists(weights: str) -> bool:
	return os.path.exists(weights)

def _create_session(weights: str, device: str) -> onnx.InferenceSession:
	try:
		provider = SUPPORTED_DEVICES[device]['provider']
		model = onnx.InferenceSession(weights, providers=[provider])
		return model
	except:
		providers = onnx.get_available_providers()
		raise BaseException('Problem while creating the model.\nThe available providers are ' + providers)

def run_model(model: YOLOv5Onnx, img: np.ndarray):
    '''
        Run an image on an onnx model
    '''
    output_name = model.session.get_outputs()[0].name
    input_name = model.session.get_inputs()[0].name
    output = model.session.run([output_name], {input_name: img})[0]
    return onnx_posprocess.non_max_suppression_np(output)[0]