"""Runtime regression tests; every app instance and write uses a temporary directory."""
import base64
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ai-camera-runtime-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "app.py"
        shutil.copyfile(Path(__file__).with_name("app.py"), source)
        name = "isolated_camera_runtime"
        spec = importlib.util.spec_from_file_location(name, source)
        self.app = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.app
        self.addCleanup(sys.modules.pop, name, None)
        with patch.dict(os.environ, {"AI_CAMERA_DATA_DIR": str(self.root / "data")}):
            spec.loader.exec_module(self.app)
        self.app.app.testing = True
        self.client = self.app.app.test_client()
        self.headers = self.auth("admin", "admin")
        self.writer = Mock()
        self.writer.isOpened.return_value = True

    @staticmethod
    def auth(user, password):
        value = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        return {"Authorization": "Basic " + value}

    def camera(self, blur=False):
        cfg = self.app.clean_settings({"cameras": [{"id": "fixture", "source": "0",
            "features": {"blur_faces": blur}}]})["cameras"][0]
        with patch.object(self.app.threading.Thread, "start"):
            cam = self.app.Camera(cfg)
        cam.raw = np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        cam.jpeg = cv2.imencode(".jpg", cam.raw)[1].tobytes()
        cam.online = True
        self.app.cameras[cam.id] = cam
        return cam

    def test_unicode_credentials_and_unauthorized_requests(self):
        self.app.settings.update(username="caf\u00e9", password="\u5bc6\u7801")
        response = self.client.get("/api/settings", headers=self.auth("caf\u00e9", "\u5bc6\u7801"))
        self.assertEqual(response.status_code, 200)
        for headers in ({}, self.auth("caf\u00e9", "wrong"), {"Authorization": "Bearer token"}):
            response = self.client.get("/api/settings", headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertIn('charset="UTF-8"', response.headers["WWW-Authenticate"])

    def test_failed_replace_preserves_settings_and_live_configuration(self):
        self.app.save_settings(self.app.settings)
        previous = Path(self.app.SETTINGS_FILE).read_bytes()
        with patch.object(self.app.os, "replace", side_effect=OSError("disk failure")), \
             patch.object(self.app, "sync_cameras") as sync:
            response = self.client.post("/api/settings", headers=self.headers, json={"confidence": .8})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.app.settings["confidence"], .5)
        self.assertEqual(Path(self.app.SETTINGS_FILE).read_bytes(), previous)
        self.assertEqual(list((self.root / "data").glob("*.tmp")), [])
        sync.assert_not_called()

    def test_partial_write_does_not_truncate_previous_settings(self):
        self.app.save_settings(self.app.settings)
        previous = Path(self.app.SETTINGS_FILE).read_bytes()
        def interrupted_dump(data, file, **kwargs):
            file.write('{"incomplete":')
            raise OSError("disk full")
        with patch.object(self.app.json, "dump", side_effect=interrupted_dump):
            with self.assertRaises(OSError):
                self.app.save_settings({"password": "changed"})
        self.assertEqual(Path(self.app.SETTINGS_FILE).read_bytes(), previous)
        self.assertEqual(list((self.root / "data").glob("*.tmp")), [])

    def test_successful_settings_save_uses_configured_data_directory(self):
        self.app.ai = SimpleNamespace(label="test")
        with patch.object(self.app, "sync_cameras") as sync:
            response = self.client.post("/api/settings", headers=self.headers, json={"confidence": .8})
        self.assertEqual(response.status_code, 200)
        sync.assert_called_once()
        self.assertEqual(self.app.load_settings()["confidence"], .8)
        self.assertTrue((self.root / "data" / "settings.json").is_file())
        self.assertFalse((self.root / "settings.json").exists())
        for path in (self.app.STATS_FILE, self.app.INSPECT_DIR, self.app.TRAIN_DIR):
            self.assertEqual(Path(path).parent, self.root / "data")
        self.assertEqual(self.app.training_service.root, self.root / "ai-workspace")

    def test_non_finite_settings_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            response = self.client.post("/api/settings", headers=self.headers, json={"confidence": value})
            self.assertEqual(response.status_code, 400)

    def test_manual_recording_works_when_automatic_recording_is_disabled(self):
        cam = self.camera()
        self.app.settings["record_clips"] = False
        with patch.object(self.app.cv2, "VideoWriter", return_value=self.writer), \
             patch.object(self.app.shutil, "disk_usage", return_value=SimpleNamespace(free=2 << 30)):
            response = self.client.post("/api/record/fixture", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(cam.clip)
        self.assertEqual(self.client.post("/api/record/fixture", headers=self.headers).status_code, 409)
        self.assertEqual(len(self.app.events), 1)

    def test_failed_encoder_returns_error_and_releases_writer(self):
        cam = self.camera()
        self.writer.isOpened.return_value = False
        with patch.object(self.app.cv2, "VideoWriter", return_value=self.writer), \
             patch.object(self.app.shutil, "disk_usage", return_value=SimpleNamespace(free=2 << 30)):
            response = self.client.post("/api/record/fixture", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertIn("encoder", response.json["error"])
        self.assertIsNone(cam.clip)
        self.writer.release.assert_called_once()

    def test_low_disk_recording_returns_error(self):
        cam = self.camera()
        with patch.object(self.app.shutil, "disk_usage", return_value=SimpleNamespace(free=100)), \
             patch.object(self.app.cv2, "VideoWriter") as writer:
            response = self.client.post("/api/record/fixture", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertIn("disk space", response.json["error"])
        writer.assert_not_called()
        self.assertIsNone(cam.clip)

    def test_paused_camera_still_blurs_faces(self):
        cam = self.camera(blur=True)
        cam.paused = True
        cam.frame = cam.raw.copy()
        detector = Mock()
        detector.faces.return_value = [(80, 80, 200, 180)]
        with patch.object(self.app, "ai", detector), \
             patch.object(self.app.time, "sleep", side_effect=lambda _: setattr(cam, "running", False)):
            cam._process()
        detector.faces.assert_called_once()
        detector.objects.assert_not_called()
        output = cv2.imdecode(np.frombuffer(cam.jpeg, np.uint8), cv2.IMREAD_COLOR)
        difference = np.abs(output[90:170, 90:190].astype(float) - cam.raw[90:170, 90:190]).mean()
        self.assertGreater(difference, 30)
        self.assertIsNone(cam.overlay)

    def test_privacy_hides_image_if_face_detector_fails_or_is_missing(self):
        detector = Mock()
        detector.faces.side_effect = RuntimeError("inference failed")
        detector.objects.return_value = []
        for paused in (False, True):
            for engine in (None, detector):
                with self.subTest(paused=paused, missing=engine is None):
                    cam = self.camera(blur=True)
                    cam.paused = paused
                    cam.frame = cam.raw.copy()
                    with patch.object(self.app, "ai", engine), \
                         patch.object(self.app.time, "sleep", side_effect=lambda _: setattr(cam, "running", False)):
                        cam._process()
                    frame = cv2.imdecode(np.frombuffer(cam.jpeg, np.uint8), cv2.IMREAD_COLOR)
                    self.assertFalse(frame[100:200].any())

    def test_privacy_disables_old_smooth_view_overlay(self):
        cam = self.camera(blur=True)
        before = cam.jpeg
        cam.overlay = (cam.raw.copy(), np.zeros(cam.raw.shape[:2], dtype=bool))
        cam._show(np.zeros_like(cam.raw))
        self.assertEqual(cam.jpeg, before)

    def test_recording_shutdown_does_not_append_raw_frame(self):
        cam = self.camera(blur=True)
        cam.running = False
        temporary = Path(self.app.EVENTS_DIR) / "fixture.tmp.mp4"
        temporary.write_bytes(b"fake encoded video")
        event = {"id": "fixture"}
        cam.clip = {"writer": self.writer, "size": (320, 240), "tmp": str(temporary),
                    "base": "fixture", "ev": event, "end": 0}
        cam._process()
        self.writer.write.assert_not_called()
        self.writer.release.assert_called_once()
        self.assertEqual(event["clip"], "fixture.mp4")
        self.assertIsNone(cam.clip)


if __name__ == "__main__":
    unittest.main()
