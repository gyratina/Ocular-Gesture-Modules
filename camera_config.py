import logging

import cv2 as cv

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class CameraConfig:
    def __init__(
        self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30
    ) -> None:
        self.camera_index: int = camera_index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps

    def set_camera(self) -> cv.VideoCapture:
        video: cv.VideoCapture = cv.VideoCapture(self.camera_index)
        if not video.isOpened():
            log.error("Impossibile aprire la telecamera.\n")
            raise RuntimeError("Impossibile aprire la telecamera.")

        video.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=self.width)
        video.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=self.height)
        video.set(propId=cv.CAP_PROP_FPS, value=self.fps)

        return video
