# AquaVision

A local Flask web application for real-time Aquafina bottle detection using the included YOLO model. The model is loaded once when the backend starts; it is not retrained or modified.

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
