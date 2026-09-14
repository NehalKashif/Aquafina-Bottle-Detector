from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "best.onnx"
CAPTURES_DIR = PROJECT_ROOT / "captures"
CAMERA_INDEX = 0
# Kept consistent with the original webcam_detect.py.
CONFIDENCE_THRESHOLD = 0.60
HOST = "127.0.0.1"
PORT = 5000
