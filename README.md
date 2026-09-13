# AquaVision

A local Flask web application for real-time Aquafina bottle detection using the included ONNX model. The model is loaded once when the backend starts; it is not retrained or modified.

## Run on Windows PowerShell

```powershell
cd C:\Users\Mueed Ahmed\Desktop\aquavision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in Chrome. If PowerShell prevents activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that terminal, then activate the environment again.

## Logo

Put the official logo at `static/assets/aquafina_logo.png`. Until it is supplied, the header automatically uses a styled **AquaVision** text logo.

## Controls

Start Camera opens the configured webcam (default index `0`), Stop Camera releases it, and Capture saves the latest annotated frame to `captures/aquavision_YYYYMMDD_HHMMSS.jpg`.

## Test

```powershell
python -m pytest -q
```

A local Flask application that reads webcam frames, runs Aquafina bottle detection with ONNX Runtime, draws the detections, and exposes the live annotated stream in the existing web interface.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- A webcam for local live detection
- `best.onnx` in the project root

The application uses Flask, ONNX Runtime, OpenCV, NumPy, and Gunicorn. It does not use Ultralytics, PyTorch, or the `.pt` model for inference.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

The model must be named `best.onnx` and located beside `app.py`:

```text
D:\Aquafina Bottle Detector\best.onnx
```

## Start The Application

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in a browser. The production-style server command is:

```powershell
gunicorn --bind 0.0.0.0:5000 app:app
```

Gunicorn is primarily useful on Linux-based deployments. On Windows, use `python app.py` for local development.

## Test The Webcam Detector

Use the web interface and select **Start Camera**, or run the standalone local webcam window:

```powershell
python webcam_detect.py
```

Press `q` in the standalone webcam window to stop it. The web interface also provides Stop Camera and Capture controls.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web interface |
| `GET` | `/api/status` | Model, camera, and detection status |
| `POST` | `/api/camera/start` | Open the configured webcam |
| `POST` | `/api/camera/stop` | Release the webcam |
| `GET` | `/video_feed` | MJPEG annotated camera stream |
| `POST` | `/api/capture` | Save the latest annotated frame |

For example:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/status
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/camera/start
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/camera/stop
```

## Configuration

Edit `config.py` to change:

- `MODEL_PATH`: the project-relative ONNX model path
- `CAMERA_INDEX`: OpenCV camera index, normally `0`
- `CONFIDENCE_THRESHOLD`: detection threshold, currently `0.75`
- `HOST` and `PORT`: the local Flask server binding

The ONNX detector loads one `InferenceSession` with `CPUExecutionProvider` when the application starts. It letterboxes frames to the model input size, converts BGR to RGB, normalizes and batches the tensor, parses the `[1, 5, 8400]` output, removes padding, clips boxes, and applies NMS.

## Project Structure

```text
.
├── app.py
├── best.onnx
├── config.py
├── requirements.txt
├── webcam_detect.py
├── services/
│   ├── camera_service.py
│   └── detector.py
├── templates/
│   └── index.html
├── static/
│   ├── assets/
│   ├── css/
│   └── js/
├── captures/
└── tests/
	└── test_app.py
```

`captures/` contains runtime-generated annotated images and is not intended for source control. The dataset directory and generated ONNX test output are also ignored.

## Troubleshooting

- **Model not ready:** confirm that `best.onnx` exists beside `app.py`, then inspect `/api/status` for the load error.
- **Camera cannot open:** close other applications using the webcam and try another `CAMERA_INDEX`.
- **No detections:** improve lighting and framing, and confirm the bottle is visible. The threshold is `0.75`.
- **Inference error:** check the terminal output and `/api/status`; the detector reports model, frame, and output-shape errors there.
- **Missing package:** activate `.venv` and run `python -m pip install -r requirements.txt` again.

## Cloud Limitation

`cv2.VideoCapture(0)` accesses the webcam attached to the machine running Flask. It works locally, but a cloud server cannot access a visitor's laptop webcam. Browser camera capture or browser-side ONNX Runtime inference is required for a public cloud webcam application.

## Tests

```powershell
python -m pytest -q
```
