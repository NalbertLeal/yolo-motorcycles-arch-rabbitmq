import cv2
import numpy as np

class Reader():
    def __init__(self, video_file_path: str):
        self.video_file_path = video_file_path
        self.video = None
        self.height = 0
        self.width = 0

        self._init_video()

    def _init_video(self):
        self.video = cv2.VideoCapture(self.video_file_path)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)

    def next_frame(self) -> np.ndarray:
        return self.video.read()