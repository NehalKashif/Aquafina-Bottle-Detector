from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


class AquafinaDetector:
    """Loads the ONNX model once and produces annotated frames."""

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.session: Any | None = None
        self.input_name: str | None = None
        self.input_width = 640
        self.input_height = 640
        self.error: str | None = None
        self.class_names = {0: "Aquafina"}
        self._load_model()

    @property
    def ready(self) -> bool:
        return self.session is not None and self.input_name is not None

    def _load_model(self) -> None:
        if not self.model_path.is_file():
            self.error = f"Model file not found: {self.model_path.name}"
            return
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            shape = model_input.shape
            if len(shape) != 4 or shape[1] != 3:
                raise ValueError(f"Unexpected model input shape: {shape}")
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_height = shape[2]
                self.input_width = shape[3]
        except Exception as exc:  # surfaced through the status API
            self.error = f"Model load failed: {exc}"
            self.session = None
            self.input_name = None

    def _prepare_image(self, frame: Any) -> tuple[np.ndarray, float, int, int]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels.")
        frame_height, frame_width = frame.shape[:2]
        if frame_height == 0 or frame_width == 0:
            raise ValueError("Received an empty camera frame.")
        scale = min(self.input_width / frame_width, self.input_height / frame_height)
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(frame_height * scale))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None, ...], scale, pad_x, pad_y

    def _parse_detections(
        self, output: np.ndarray, frame_shape: tuple[int, ...], scale: float,
        pad_x: int, pad_y: int,
    ) -> list[tuple[list[int], float]]:
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 5:
            raise ValueError(f"Unexpected model output shape: {output.shape}")
        predictions = output[0].T
        frame_height, frame_width = frame_shape[:2]
        boxes: list[list[int]] = []
        scores: list[float] = []
        for center_x, center_y, width, height, confidence in predictions:
            confidence = float(confidence)
            if confidence < self.confidence_threshold:
                continue
            x1 = (float(center_x) - float(width) / 2 - pad_x) / scale
            y1 = (float(center_y) - float(height) / 2 - pad_y) / scale
            x2 = (float(center_x) + float(width) / 2 - pad_x) / scale
            y2 = (float(center_y) + float(height) / 2 - pad_y) / scale
            x1 = max(0, min(frame_width - 1, round(x1)))
            y1 = max(0, min(frame_height - 1, round(y1)))
            x2 = max(0, min(frame_width - 1, round(x2)))
            y2 = max(0, min(frame_height - 1, round(y2)))
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append([x1, y1, box_width, box_height])
            scores.append(confidence)
        if not boxes:
            return []
        kept = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, 0.45)
        indices = np.asarray(kept).reshape(-1) if len(kept) else []
        return [(boxes[int(index)], scores[int(index)]) for index in indices]

    def detect(self, frame: Any) -> tuple[Any, list[dict[str, float | str]]]:
        """Return a blue annotated frame and the detections in it."""
        if not self.ready:
            return frame, []

        try:
            tensor, scale, pad_x, pad_y = self._prepare_image(frame)
            output = self.session.run(None, {self.input_name: tensor})[0]
            parsed = self._parse_detections(output, frame.shape, scale, pad_x, pad_y)
        except Exception as exc:
            self.error = f"Inference failed: {exc}"
            return frame, []
        self.error = None
        detections: list[dict[str, float | str]] = []
        for box, confidence in parsed:
            x1, y1, box_width, box_height = box
            x2, y2 = x1 + box_width, y1 + box_height
            label = self.class_names[0]
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
