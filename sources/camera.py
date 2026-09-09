from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import cv2


class CameraSource(ABC):
    """Minimal camera interface used by the runtime.

    Future implementations (OAK-D, RealSense, robot streams) should expose the
    same read/close contract. Extra capabilities such as depth stay optional.
    """

    @abstractmethod
    def read(self) -> tuple[bool, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def metadata(self) -> dict:
        return {"source_type": self.__class__.__name__}


class OpenCVCameraSource(CameraSource):
    """Current webcam adapter backed by cv2.VideoCapture."""

    def __init__(self, device: int | str = 0):
        self.device = device
        self._capture = cv2.VideoCapture(device)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                f"Cannot open camera {device!r}. Change the configured camera device."
            )

    def read(self) -> tuple[bool, Any]:
        return self._capture.read()

    def close(self) -> None:
        self._capture.release()

    def metadata(self) -> dict:
        return {
            "source_type": "opencv_camera",
            "device": self.device,
            "capabilities": ["rgb"],
        }
