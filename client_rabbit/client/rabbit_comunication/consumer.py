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
        json_package = json.loads(package)
        (height, width, channels) = json_package['height'], json_package['width'], json_package['channels']

        bboxes = np.array(json_package['bboxes'], dtype=np.float64)

        frame = np.array(json_package['frame'], dtype=np.float32)
        frame *= 255
        return bboxes, frame.reshape((height, width, channels)).astype(np.uint8)

    def _callback(self, ch, method, properties, body):
        bboxes, frame = self._deserialize(body)
        print(bboxes.shape)
        if bboxes.shape[0] > 0:
            # bboxes = np.frombuffer(data, dtype=np.float64).reshape((len(data)//48, 6))

            for box in bboxes:
                [minx, miny, maxx, maxy, confidence, _] = box
                cv2.rectangle(frame,(int(minx), int(miny)), (int(maxx), int(maxy)), (0,255,0), 10)
                cv2.putText(frame, 'Motorcycle '+str(confidence)[2:4]+'%', (int(minx), int(miny) - 12), 0, 0.005 * (int(maxy) - int(miny)), (0,255,0), 10//3)
        cv2.imshow('video result', frame)

        cv2.waitKey(1)

    def run(self):
        self.channel.start_consuming()