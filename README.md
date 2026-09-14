# AquaVision

AquaVision is a browser-based Aquafina bottle detector built with Flask, Jinja2, HTML, vanilla JavaScript, CSS, and ONNX Runtime Web.

The primary application runs camera capture and inference in the visitor's browser:

```text
Browser webcam
	-> HTML video
	-> Canvas letterbox preprocessing
	-> ONNX Runtime Web
	-> best.onnx
	-> JavaScript post-processing and NMS
	-> Canvas bounding-box overlay
	-> AquaVision detection UI
```

This architecture works on local development and cloud deployments such as Vercel because the Flask server does not need access to the visitor's webcam.

## Features

- Browser webcam access through `navigator.mediaDevices.getUserMedia()`
- Client-side ONNX inference with WebGPU when available and WASM fallback
- Aquafina bounding boxes, confidence percentages, and detection count
- Existing AquaVision status panel and controls
- Browser-side capture downloads with detections drawn into a PNG
- Retained Flask/OpenCV routes for backward compatibility
- Standalone Python webcam detector for local testing

## Model

The repository includes two model files:

| File | Purpose |
| --- | --- |
| `best.onnx` | ONNX model used by the Python backend and browser frontend |
| `best.pt` | Original PyTorch model artifact; not required for inference |

The browser-served copy is `static/models/best.onnx`.

Verified ONNX model information:

- Input name: `images`
- Input shape: `[1, 3, 640, 640]`
- Input type: `float32`
- Output name: `output0`
- Output shape: `[1, 5, 8400]`
- Output type: `float32`
- Classes: one class, `Aquafina`
- NMS: not included in the model
- Opset: 18
- Export metadata: Ultralytics `8.4.142`

The browser reproduces the Python detector's model behavior:

1. Resize the camera frame while preserving aspect ratio.
2. Letterbox to `640x640` with padding value `114`.
3. Read browser Canvas pixels as RGB.
4. Normalize values by `255`.
5. Convert HWC pixels to CHW tensor layout.
6. Run the model with input name `images`.
7. Interpret each output prediction as `[center_x, center_y, width, height, confidence]`.
8. Filter at confidence `0.75`.
9. Undo letterboxing and clip boxes to the camera frame.
10. Apply class-agnostic NMS with IoU threshold `0.45`.

## Project Structure

```text
.
├── app.py                         Flask application and HTTP routes
├── config.py                      Application paths and settings
├── best.onnx                      ONNX model artifact
├── best.pt                        Original PyTorch model artifact
├── requirements.txt               Python dependencies
├── package.json                   Frontend dependency metadata
├── package-lock.json              Locked npm dependency versions
├── webcam_detect.py               Standalone local Python webcam detector
├── aquafina/                      Dataset files and YOLO annotations
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── services/
│   ├── camera_service.py          Legacy OpenCV camera lifecycle service
│   └── detector.py                Legacy Python ONNX detector
├── static/
│   ├── assets/                    Logos and visual assets
│   ├── css/style.css              Existing AquaVision styling
│   ├── js/app.js                  Browser camera and inference logic
│   └── models/best.onnx           Browser-accessible model copy
├── templates/index.html            Jinja2 application page
├── captures/                      Runtime-generated legacy captures
└── tests/test_app.py               Flask and service tests
```

## Requirements

### Python

- Python 3.10 or newer
- Flask
- NumPy
- OpenCV
- ONNX Runtime
- Gunicorn for Linux production serving

Install the Python dependencies from `requirements.txt`.

### Browser

- A modern browser with WebRTC support
- Camera permission
- HTTPS in deployment, or `localhost` during local development
- JavaScript enabled

The frontend uses the `onnxruntime-web` package. The current browser script also loads the browser runtime from jsDelivr when it is not already present on the page.

## Installation

From the project directory in PowerShell:

```powershell
cd "D:\Aquafina Bottle Detector"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
npm install
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The frontend package check can be run with:

```powershell
npm run check
```

## Run AquaVision Locally

Start the Flask server:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Then:

1. Click **Start Camera**.
2. Allow camera access when the browser asks.
3. Wait for the model status to become **Ready**.
4. Point the camera at an Aquafina bottle.
5. Use **Capture** to download the current video frame and overlay as a PNG.
6. Click **Stop Camera** to stop the media tracks and inference loop.

The application does not request camera permission when the page loads. Permission is requested only after **Start Camera** is clicked.

## Browser Inference Details

The client implementation is in `static/js/app.js`.

Important constants:

```text
MODEL_URL = /static/models/best.onnx
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.75
NMS_IOU_THRESHOLD = 0.45
CLASS_NAME = Aquafina
```

The browser tries the WebGPU execution provider first when `navigator.gpu` is available. If WebGPU session creation fails, it creates a WASM session instead. Inference is throttled to approximately 13 frames per second to avoid unnecessary CPU usage.

The live video is displayed in an HTML `<video>` element. Detection rectangles and labels are drawn on a Canvas positioned above it. The Canvas is resized when video metadata loads and when the browser window changes size.

## Capture Behavior

The browser Capture button does not call the server's `/api/capture` endpoint. It creates a temporary Canvas, draws the current video frame and detection overlay, and downloads a file named like:

```text
aquafina-capture-2026-09-14T12-30-00-000Z.png
```

The legacy server capture endpoint remains available for the Python/OpenCV mode.

## Flask Routes

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the AquaVision interface |
| `GET` | `/api/status` | Reports legacy backend model and camera status |
| `POST` | `/api/camera/start` | Starts the legacy server-side OpenCV camera |
| `POST` | `/api/camera/stop` | Stops the legacy server-side OpenCV camera |
| `GET` | `/video_feed` | Serves the legacy MJPEG stream |
| `POST` | `/api/capture` | Saves the latest legacy server-side annotated frame |

The browser webcam workflow does not call `/api/camera/start`, `/api/camera/stop`, `/video_feed`, or `/api/capture`.

Example status request:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/status
```

## Legacy Python Webcam Mode

The old server-side detector is retained for local compatibility and comparison. It uses OpenCV to access the webcam on the computer running Python:

```powershell
python webcam_detect.py
```

Press `q` in the OpenCV window to stop it.

This mode must not be used as the cloud webcam implementation. A deployed Flask server cannot access a visitor's local webcam through `cv2.VideoCapture(0)`.

## Testing

Run the Python tests:

```powershell
python -m pytest -q
```

Run the JavaScript syntax check:

```powershell
npm run check
```

Verify that Flask serves the browser model:

```powershell
Invoke-WebRequest http://127.0.0.1:5000/static/models/best.onnx -Method Head
```

The expected response is HTTP `200`. The model is approximately 12.3 MB.

## Deployment

The deployed application must serve these resources:

```text
/
/static/js/app.js
/static/css/style.css
/static/models/best.onnx
```

After deployment, verify the model URL directly:

```text
https://YOUR-DOMAIN.example/static/models/best.onnx
```

Then open the HTTPS application URL, click **Start Camera**, grant permission, and verify model loading and detection. Camera access usually fails on plain HTTP except for `localhost`, because browsers require a secure context for `getUserMedia()`.

The server-side OpenCV routes are retained for backward compatibility, but they are not suitable for accessing a user's camera on Vercel or another cloud host.

## Troubleshooting

### Camera permission denied

Allow camera access in the browser's site permissions and click **Start Camera** again. The application reports permission errors through its existing notification UI.

### No camera found

Confirm that the device has a webcam and that another application is not using it.

### Camera already in use

Close video-conferencing or camera applications, then restart the browser camera.

### Model loading failed

Check that this URL returns HTTP `200`:

```text
/static/models/best.onnx
```

Also confirm that the static model file is present and that the browser console does not report a failed ONNX Runtime Web or WASM resource load.

### WebGPU is unavailable

This is supported. The browser falls back to the WASM execution provider. Performance may be lower, especially on older devices.

### No detection appears

Use good lighting, keep the bottle visible and steady, and make sure it is large enough in the camera frame. The current confidence threshold is `0.75`.

### Legacy backend model is not ready

Confirm that `best.onnx` exists beside `app.py`. The backend uses the path configured by `MODEL_PATH` in `config.py`.

## Configuration

The main Python settings are in `config.py`:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | Project root `/best.onnx` | Legacy Python model path |
| `CAPTURES_DIR` | `captures/` | Legacy server capture directory |
| `CAMERA_INDEX` | `0` | Legacy OpenCV camera index |
| `CONFIDENCE_THRESHOLD` | `0.75` | Legacy Python detector threshold |
| `HOST` | `127.0.0.1` | Local Flask bind address |
| `PORT` | `5000` | Local Flask port |

Browser inference constants are defined in `static/js/app.js` so the browser path remains independent of server camera state.

## License and Dataset Information

The dataset metadata in `aquafina/data.yaml` identifies the Roboflow dataset as version 4 under the CC BY 4.0 license. Review the original dataset and model licenses before redistributing the model or deploying the application commercially.
