import json
from typing import List

import cv2
import numpy as np
import pika

from preprocess.onnx_preprocessing import image

class RabbitMQProducerConsumer():
    def __init__(self, host: str, consumer_routing_keys: List[str], producer_routing_keys: List[str]):
        self.host = host
        self.consumer_routing_keys = consumer_routing_keys
        self.producer_routing_keys = producer_routing_keys

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

    def _deserialize(self, package):
        json_package = json.loads(package)
        # (height, width, channels) = response['height'], response['width'], response['channels']
        
        # arr = np.array(response['frame'])
        # return np.frombuffer(arr, dtype=np.uint64).reshape((height, width, channels)).astype(np.uint8)
        json_package['frame'] = np.array(json_package['frame'], dtype=np.uint8)
        return json_package
    
    def _serialize(sellf, json_package, frame):
        (_, c, h, w) = frame.shape
        package = {
            'name': json_package['name'],
            'height': h,
            'width': w,
            'channels': c,
            'frame': frame.tolist(),
        }
        return json.dumps(package)

    def _callback(self, ch, method, properties, body):
        json_package = self._deserialize(body)
        onnx_image = image.preprocess_image_to_onnx(json_package['frame'])
        self._send(json_package, onnx_image)

    def _send(self, json_package, frame):
        for routing_key in self.producer_routing_keys:
            serialized = self._serialize(json_package, frame)
            self.channel.basic_publish(exchange='motorcycles', routing_key=routing_key, body=serialized)