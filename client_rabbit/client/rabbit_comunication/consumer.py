import ujson
from threading import Thread
from typing import List

import cv2
import numpy as np
import pika

from client.encode_decode import opencv_frame, yolo_result

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

    def _deserialize(self, bboxes_frame):
        return yolo_result.decode(bboxes_frame)

    def convert(self, img, target_type_min, target_type_max, target_type):
        imin = np.min(img)
        imax = np.max(img)

        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

    def _callback(self, ch, method, properties, body):
        _, bboxes, frame = self._deserialize(body)
        frame = frame.transpose(1, 2, 0)
        
        frame = self.convert(frame, 0, 255, np.uint8)

        if bboxes is not None and bboxes.shape[0] > 0:
            # bboxes = np.frombuffer(data, dtype=np.float64).reshape((len(data)//48, 6))

            for box in bboxes:
                [minx, miny, maxx, maxy, confidence, _] = box
                minx = int(max(minx, 0))
                miny = int(max(miny, 0))
                maxx = int(min(maxx, 639))
                maxy = int(min(maxy, 639))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0, 254 ,0), thickness=10)
                cv2.putText(frame, 'Motorcycle '+str(confidence)[2:4]+'%', (int(minx), int(miny) - 12), 0, 0.005 * (int(maxy) - int(miny)), (0,255,0), 10//3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('video result', frame)

        cv2.waitKey(1)
        cv2.imshow('video result', frame)

        cv2.waitKey(1)

    def run(self):
        self.channel.start_consuming()