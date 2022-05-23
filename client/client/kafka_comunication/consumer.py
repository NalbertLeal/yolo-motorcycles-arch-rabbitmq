import struct

import numpy as np
import kafka

class KafkaConsumer():
    __slots__ = ('host', 'port', 'video_partition', 'consumer')

    def __init__(self, host: str, port: int, video_partition: int):
        self.host = host
        self.port = port
        self.video_partition = video_partition
        self.consumer = kafka.KafkaConsumer(
            # 'FINAL_RESULTS',
            bootstrap_servers=[f'{self.host}:{self.port}'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='client',
        )
        self.consumer.assign([kafka.TopicPartition('FINAL_RESULTS', self.video_partition)])

    def _header_from_response(self, kafka_response):
        heigth = struct.unpack('I', kafka_response[:4])
        width = struct.unpack('I', kafka_response[4:8])
        channels = struct.unpack('I', kafka_response[8:12])
        bboxes_float64_size = struct.unpack('I', kafka_response[12:16])
        frame_float32_size = len(kafka_response) - 16 - bboxes_float64_size
        return (heigth, width, channels, bboxes_float64_size, frame_float32_size)

    def _deserialize_bboxes(self, raw_bboxes):
        if len(raw_bboxes) == 0:
            return np.array([])
        float64_size = 8
        bboxes_number = float64_size * 6
        return np.array(raw_bboxes, dtype=np.float64).reshape((bboxes_number, 6))

    def _deserialize_frame(self, heigth, width, channels, raw_frame):
        if len(raw_frame) == 0:
            raise BaseException('Empty frame from kafka')
        return np.array(raw_frame, dtype=np.float32).reshape((heigth, width, channels))

    def deserialize_kafka_response(self, kafka_response: bytes):
        (heigth, width, channels, bboxes_float64_size, frame_float32_size) = self._header_from_response(kafka_response)
        (_, _, _, raw_bboxes, raw_frame) = struct.unpack(f'I I I I {bboxes_float64_size}s {frame_float32_size}s')
        bboxes = self._deserialize_bboxes(raw_bboxes)
        frame = self._deserialize_frame(heigth, width, channels, raw_frame)
        return (bboxes, frame)