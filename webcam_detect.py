"""Run the ONNX Aquafina detector against a local webcam."""
import cv2

import config
from services.detector import AquafinaDetector

detector = AquafinaDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD)
if not detector.ready:
    raise SystemExit(detector.error or "The detection model is not ready.")

camera = cv2.VideoCapture(config.CAMERA_INDEX)
if not camera.isOpened():
    raise SystemExit("Could not open webcam. Check that it is connected and not in use.")

try:
    print("Webcam started. Press 'q' to quit.")
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read a frame from the webcam.")
            break
        annotated_frame, _ = detector.detect(frame)
        cv2.imshow("Aquafina Detector - Press 'q' to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
