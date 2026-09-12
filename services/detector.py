from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


class AquafinaDetector:
    """Loads the YOLO model once and produces a clean annotated frame."""

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model: YOLO | None = None
        self.error: str | None = None
        self.class_names: dict[int, str] = {}
        self._load_model()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        if not self.model_path.is_file():
            self.error = f"Model file not found: {self.model_path.name}"
            return
        try:
            self.model = YOLO(str(self.model_path))
            self.class_names = dict(self.model.names)
        except Exception as exc:  # surfaced through the status API
            self.error = f"Model load failed: {exc}"

    def detect(self, frame: Any) -> tuple[Any, list[dict[str, float | str]]]:
        """Return a blue annotated frame and the detections in it."""
        if self.model is None:
            return frame, []

        result = self.model.predict(
            source=frame, conf=self.confidence_threshold, verbose=False
        )[0]
        detections: list[dict[str, float | str]] = []
        boxes = result.boxes
        if boxes is None:
            return frame, detections

        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.class_names.get(class_id, str(class_id))
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append({"label": label, "confidence": confidence})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (206, 114, 0), 2)
            caption = f"{label} {confidence:.0%}"
            (width, height), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_top = max(y1, height + baseline + 8)
            cv2.rectangle(
                frame, (x1, label_top - height - baseline - 8),
                (x1 + width + 12, label_top), (206, 114, 0), -1
            )
            cv2.putText(frame, caption, (x1 + 6, label_top - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return frame, detections
