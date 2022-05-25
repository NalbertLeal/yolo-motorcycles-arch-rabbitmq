import ujson
from typing import List

import cv2
import numpy as np
import pika

from preprocess.encode_decode import opencv_frame
from preprocess.onnx_preprocessing import image
import preprocess.proto.motorcycle_pb2 as mpb

class RabbitMQProducerConsumer():
    def __init__(self, host: str, consumer_routing_keys: List[str], producer_routing_key: List[str], number_yolos: int):
        self.host = host
        self.consumer_routing_keys = consumer_routing_keys
        self.producer_routing_key = producer_routing_key
        self.number_yolos = number_yolos
        self.media_to_yolos = {}
        self.next_yolo = 0

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
    #     return opencv_frame.decode(frame_bytes)
    def _deserialize(self, body):
        yolo_pg = mpb.YOLOv5Package()
        yolo_pg.ParseFromString(body)
        return yolo_pg
    
    # def _serialize(sellf, name, frame):
    #     return opencv_frame.encode(name, frame)
    def _serialize(sellf, name, onnx_image):
        yolo_pg = mpb.YOLOv5Package()
        yolo_pg.name = name
        yolo_pg.frame.shape.extend(onnx_image.shape)
        yolo_pg.frame.frame = onnx_image.tobytes()
        return yolo_pg.SerializeToString()

    def _callback(self, ch, method, properties, body):
        yolo_pg = self._deserialize(body)
        frame = np.frombuffer(yolo_pg.frame.frame, dtype=np.uint8).reshape(yolo_pg.frame.shape)
        onnx_image = image.preprocess_image_to_onnx(frame)
        self._send(yolo_pg.name, onnx_image)

    def _send(self, name, onnx_image):
        if name not in self.media_to_yolos.keys():
            self.media_to_yolos[name] = self.next_yolo
            if self.next_yolo+1 == self.number_yolos:
                self.next_yolo = 0
            else:
                self.next_yolo += 1
        # for routing_key in self.producer_routing_keys:
        serialized = self._serialize(name, onnx_image)
        self.channel.basic_publish(exchange='motorcycles', routing_key=self.producer_routing_key+str(self.media_to_yolos[name]), body=serialized)