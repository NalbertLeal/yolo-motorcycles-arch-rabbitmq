import json
from threading import Thread
from typing import List

import cv2
import numpy as np
import pika

class RabbitMQConsumer(Thread):
    def __init__(self, host: str, routing_keys: List[str]):
        Thread.__init__(self)

        self.host = host

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='motorcycles', exchange_type='topic')
        result = self.channel.queue_declare('', exclusive=True)
        self.queue_name = result.method.queue

        for routing_key in routing_keys:
            self.channel.queue_bind(exchange='motorcycles', routing_key=routing_key, queue=self.queue_name)

        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self._callback, auto_ack=True)
    
    def __del__(self):
        self.connection.close()

    def _deserialize(self, package):
        response = json.loads(package)
        (height, width, channels) = response['height'], response['width'], response['channels']
        
        arr = np.array(response['frame'])
        return np.frombuffer(arr, dtype=np.int64).reshape((height, width, channels)).astype(np.uint8)

    def _callback(self, ch, method, properties, body):
        response_msg = self._deserialize(body)
        cv2.imshow('video result', response_msg)

        cv2.waitKey(1)

    def run(self):
        self.channel.start_consuming()