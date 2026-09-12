"""
Real-time Aquafina bottle detection using your trained YOLOv8 model and a webcam.

Run this LOCALLY (not in Colab) — it needs direct access to your machine's webcam.

Usage:
    python webcam_detect.py
"""

from ultralytics import YOLO
import cv2

# Path to your downloaded trained model
MODEL_PATH = "best.pt"  # update this if you place it elsewhere

# Load the trained model
model = YOLO(MODEL_PATH)

# Open the default webcam (0 = first connected camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check that it's connected and not in use by another app.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame from webcam.")
        break

    # Run detection on the current frame
    results = model.predict(source=frame, conf=0.2, verbose=False)

    # results[0].plot() draws boxes/labels directly onto the frame for us
    annotated_frame = results[0].plot()

    cv2.imshow("Aquafina Detector - Press 'q' to quit", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
