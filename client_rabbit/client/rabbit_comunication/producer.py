import json
from threading import Thread
from typing import List

# import numpy as np
import pika

class RabbitMQProducer(Thread):
    def __init__(self, host: str, routing_keys: List[str], video_reader: object, file_name: str):
        Thread.__init__(self)

        self.host = host
        self.routing_keys = routing_keys
        self.video_reader = video_reader
        self.file_name = file_name

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='motorcycles', exchange_type='topic')
    
    def __del__(self):
        self.connection.close()
    
    def _serialize(self, frame):
        (h, w, c) = frame.shape
        package = {
            'name': self.file_name,
            'height': h,
            'width': w,
            'channels': c,
            'frame': frame.tolist(), 
        }
        return json.dumps(package)

    def send_frame(self, frame):
        for routing_key in self.routing_keys:
            serialized = self._serialize(frame)
            self.channel.basic_publish(exchange='motorcycles', routing_key=routing_key, body=serialized)

    def run(self):
        while True:
            has_frame, frame = self.video_reader.next_frame()
            if not has_frame:
                break
            self.send_frame(frame)