import struct

import numpy as np
import kafka

class kafkaProducer():
    __slots__ = ('host', 'port', 'video_partition', 'producer')

    def __init__(self, host: str, port: int, video_partition: str):
        self.host = host
        self.port = port
        self.video_partition = str.encode(video_partition)
        # self.video_partition = video_partition
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=[f'{self.host}:{self.port}'],
            max_request_size=10000000,
            # partitioner=lambda p: print(p)
        )
        print('Partition = ', self.producer._partition('NON_COMPUTED_FRAME', None, self.video_partition, None, self.video_partition, None))

    def _serialize_frame(self, frame: np.ndarray):
        (height, width, channels) = frame.shape
        frame_bytes = frame.tobytes()

        bytes_size = height * width * channels
        return struct.pack(f'I I I {bytes_size}s', height, width, channels, frame_bytes)

    def send_frame(self, frame: np.ndarray):
        serialized = self._serialize_frame(frame)
        matedata = self.producer.send(topic='NON_COMPUTED_FRAME', value=serialized, key= self.video_partition)
        matedata.get(timeout=1)
        # self.producer.flush()