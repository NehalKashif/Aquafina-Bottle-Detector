from __future__ import annotations

import atexit

from flask import Flask, Response, jsonify, render_template

import config
from services.camera_service import CameraService
from services.detector import AquafinaDetector


def create_app(camera_service: CameraService | None = None) -> Flask:
    app = Flask(__name__)
    service = camera_service or CameraService(
        AquafinaDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD),
        config.CAMERA_INDEX,
        config.CAPTURES_DIR,
    )
    app.extensions["camera_service"] = service

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        def generate():
            while service.active:
                frame = service.annotated_frame()
                if frame is None:
                    break
                ok, encoded = service.cv2.imencode(".jpg", frame)
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           encoded.tobytes() + b"\r\n")
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/camera/start")
    def start_camera():
        ok, error = service.start()
        return jsonify({"ok": ok, "error": error, "status": service.status()}), (200 if ok else 503)

    @app.post("/api/camera/stop")
    def stop_camera():
        service.stop()
        return jsonify({"ok": True, "status": service.status()})

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.post("/api/capture")
    def capture():
        ok, filename, error = service.capture_image()
        return jsonify({"ok": ok, "filename": filename, "error": error}), (200 if ok else 409)

    atexit.register(service.stop)
    return app


app = create_app()

if __name__ == "__main__":
    print(f"AquaVision running at http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
