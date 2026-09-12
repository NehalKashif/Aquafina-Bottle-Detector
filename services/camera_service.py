from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import cv2

from .detector import AquafinaDetector


class CameraService:
    """Owns exactly one OpenCV VideoCapture instance for the application."""

    def __init__(self, detector: AquafinaDetector, camera_index: int, captures_dir: Path,
                 cv2_module: Any = cv2) -> None:
        self.detector = detector
        self.camera_index = camera_index
        self.captures_dir = Path(captures_dir)
        self.cv2 = cv2_module
        self.capture: Any | None = None
        self.last_frame: Any | None = None
        self.last_detections: list[dict[str, float | str]] = []
        self.error: str | None = None
        self._lock = RLock()

    @property
    def active(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def start(self) -> tuple[bool, str | None]:
        with self._lock:
            if not self.detector.ready:
                return False, self.detector.error or "Detection model is not ready."
            if self.active:
                return True, None
            self.stop()
            candidate = self.cv2.VideoCapture(self.camera_index)
            if not candidate.isOpened():
                candidate.release()
                self.error = "Could not open webcam. Check that it is connected and not in use."
                return False, self.error
            self.capture = candidate
            self.error = None
            return True, None

    def stop(self) -> None:
        with self._lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            self.last_frame = None
            self.last_detections = []

    def annotated_frame(self) -> Any | None:
        with self._lock:
            if not self.active:
                return None
            success, frame = self.capture.read()
            if not success or frame is None:
                self.error = "Unable to read a frame from the webcam."
                return None
            annotated, detections = self.detector.detect(frame)
            self.last_frame = annotated.copy()
            self.last_detections = detections
            self.error = None
            return annotated

    def status(self) -> dict[str, Any]:
        with self._lock:
            highest = max((float(item["confidence"]) for item in self.last_detections), default=0.0)
            return {
                "model_ready": self.detector.ready,
                "camera_active": self.active,
                "detection_active": self.active,
                "aquafina_detected": bool(self.last_detections),
                "confidence": highest,
                "detection_count": len(self.last_detections),
                "model_classes": self.detector.class_names,
                "error": self.error or self.detector.error,
            }

    def capture_image(self) -> tuple[bool, str | None, str | None]:
        with self._lock:
            if self.last_frame is None:
                return False, None, "No camera frame is available to capture."
            self.captures_dir.mkdir(parents=True, exist_ok=True)
            filename = f"aquavision_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            path = self.captures_dir / filename
            if not self.cv2.imwrite(str(path), self.last_frame):
                return False, None, "Could not save the captured frame."
            return True, filename, None
