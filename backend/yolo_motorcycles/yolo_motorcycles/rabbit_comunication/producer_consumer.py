import ujson
from typing import List

# import cv2
import numpy as np
import pika

from yolo_motorcycles.encode_decode import opencv_frame, yolo_result
from yolo_motorcycles.model import model

class RabbitMQProducerConsumer():
    def __init__(self, host: str, consumer_routing_keys: List[str], producer_routing_keys: List[str]):
        self.host = host
        self.consumer_routing_keys = consumer_routing_keys
        self.producer_routing_keys = producer_routing_keys
        self.model = model.new_YOLOv5Onnx('onnx_models/yolov5s.onnx', 'gpu')

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='motorcycles', exchange_type='topic')

        # setup consumer
        result = self.channel.queue_declare('', exclusive=True)
        self.queue_name = result.method.queue

        for routing_key in self.consumer_routing_keys:
            self.channel.queue_bind(exchange='motorcycles', routing_key=routing_key, queue=self.queue_name)

        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self._callback, auto_ack=True)
        self.channel.start_consuming()
    
    def __del__(self):
        self.connection.close()

    def _deserialize(self, frame_bytes):
        # json_package = ujson.loads(package)
        # return json_package['name'], np.array(json_package['frame'], dtype=np.float32)
        return opencv_frame.decode(frame_bytes, array_type=np.float32)
    
    def _serialize(sellf, name, bboxes, frame):
        # (_, c, h, w) = frame.shape
        # package = {
        #     'name': name,
        #     'height': h,
        #     'width': w,
        #     'channels': c,
        #     'bboxes': bboxes.tolist(),
        #     'frame': frame.tolist(),
        # }
        # return ujson.dumps(package)
        frame = np.squeeze(frame)
        return yolo_result.encode(name, bboxes, frame)

    def _callback(self, ch, method, properties, body):
        name, frame = self._deserialize(body)
        frame = np.expand_dims(frame, axis=0)
        bboxes = model.run_model(self.model, frame)
        self._send(name, bboxes, frame)

    def _send(self, name, bboxes, frame):
        for routing_key in self.producer_routing_keys:
            serialized = self._serialize(name, bboxes, frame)
            self.channel.basic_publish(exchange='motorcycles', routing_key=routing_key+name, body=serialized)