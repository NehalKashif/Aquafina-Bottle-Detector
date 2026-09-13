from pathlib import Path

from app import create_app
from services.camera_service import CameraService


class FakeDetector:
    ready = True
    error = None
    class_names = {0: "Aquafina"}


class FakeCapture:
    def __init__(self, opened=True): self.opened, self.released = opened, False
    def isOpened(self): return self.opened
    def release(self): self.released, self.opened = True, False


class FakeCV2:
    def __init__(self): self.capture = FakeCapture()
    def VideoCapture(self, _): return self.capture
    def imwrite(self, _, __): return True


def test_home_and_status_routes(tmp_path):
    service = CameraService(FakeDetector(), 0, tmp_path, FakeCV2())
    client = create_app(service).test_client()
    assert client.get('/').status_code == 200
    status = client.get('/api/status').get_json()
    assert status['model_ready'] is True
    assert status['camera_active'] is False


def test_camera_start_stop_does_not_require_hardware(tmp_path):
    fake_cv2 = FakeCV2()
    service = CameraService(FakeDetector(), 0, tmp_path, fake_cv2)
    assert service.start()[0] is True
    assert service.active is True
    service.stop()
    assert fake_cv2.capture.released is True
    assert service.active is False


def test_config_paths_are_project_relative():
    import config
    assert config.MODEL_PATH == Path(config.__file__).resolve().parent / 'best.onnx'


def test_capture_creates_capture_directory(tmp_path):
    service = CameraService(FakeDetector(), 0, tmp_path / 'captures', FakeCV2())
    service.last_frame = object()
    saved, filename, error = service.capture_image()
    assert saved is True and filename.startswith('aquavision_') and error is None
    assert service.captures_dir.is_dir()
