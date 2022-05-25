import ujson
from typing import List

# import cv2
import numpy as np
import pika

from yolo_motorcycles.encode_decode import opencv_frame, yolo_result
from yolo_motorcycles.model import model
import yolo_motorcycles.proto.motorcycle_pb2 as mpb

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

    # def _deserialize(self, frame_bytes):
        # return opencv_frame.decode(frame_bytes, array_type=np.float32)
    def _deserialize(self, body):
        yolo_pg = mpb.YOLOv5Package()
        yolo_pg.ParseFromString(body)
        return yolo_pg
    
    # def _serialize(sellf, name, bboxes, frame):
    #     frame = np.squeeze(frame)
    #     return yolo_result.encode(name, bboxes, frame)
    def _serialize(sellf, name, bboxes, frame):
        yolo_pg = mpb.YOLOv5Package()
        yolo_pg.name = name
        yolo_pg.frame.shape.extend(frame.shape)
        yolo_pg.frame.frame = frame.tobytes()
        yolo_pg.bboxes.shape.extend(bboxes.shape)
        yolo_pg.bboxes.bboxes = bboxes.tobytes()
        return yolo_pg.SerializeToString()

    def _callback(self, ch, method, properties, body):
        yolo_pg = self._deserialize(body)
        frame = np.frombuffer(yolo_pg.frame.frame, dtype=np.float32).reshape(yolo_pg.frame.shape)
        bboxes = model.run_model(self.model, frame)
        self._send(yolo_pg.name, bboxes, frame)

    def _send(self, name, bboxes, frame):
        for routing_key in self.producer_routing_keys:
            serialized = self._serialize(name, bboxes, frame)
            self.channel.basic_publish(exchange='motorcycles', routing_key=routing_key+name, body=serialized)