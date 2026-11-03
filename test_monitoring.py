"""Isolated tests for monitoring policy, retention and configuration recovery."""
import copy
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import monitoring
import test_runtime as runtime


def timestamp(text):
    return datetime.fromisoformat(text).timestamp()


class ScheduleTests(unittest.TestCase):
    def rule(self, **values):
        return {"schedule": monitoring.clean_schedule({"enabled": True, "days": [0],
            "start": "18:00", "end": "08:00", "timezone": "Asia/Singapore", **values})}

    def test_overnight_belongs_to_starting_day(self):
        cfg = self.rule()
        for text, expected in (("2026-11-02T17:59:00+08:00", False),
                               ("2026-11-02T18:00:00+08:00", True),
                               ("2026-11-03T07:59:00+08:00", True),
                               ("2026-11-03T08:00:00+08:00", False),
                               ("2026-11-03T19:00:00+08:00", False)):
            self.assertEqual(monitoring.scheduled(cfg, timestamp(text)), expected, text)

    def test_daytime_and_all_day_boundaries(self):
        cfg = self.rule(start="09:00", end="17:00")
        self.assertTrue(monitoring.scheduled(cfg, timestamp("2026-11-02T09:00:00+08:00")))
        self.assertFalse(monitoring.scheduled(cfg, timestamp("2026-11-02T17:00:00+08:00")))
        self.assertTrue(monitoring.scheduled(self.rule(start="00:00", end="00:00"), timestamp("2026-11-02T23:59:00+08:00")))
        self.assertTrue(monitoring.scheduled(self.rule(enabled=False), timestamp("2026-11-03T12:00:00+08:00")))

    def test_timezone_daylight_saving_repeated_hour(self):
        cfg = self.rule(days=[6], start="01:00", end="02:00", timezone="America/New_York")
        for hour in ("05", "06"):
            self.assertTrue(monitoring.scheduled(cfg, timestamp(f"2026-11-01T{hour}:30:00+00:00")))
        self.assertFalse(monitoring.scheduled(cfg, timestamp("2026-11-01T07:00:00+00:00")))

    def test_invalid_schedules_rejected(self):
        for changes in ({"timezone": "not/a/zone"}, {"days": []}, {"days": [7]},
                        {"days": [True]}, {"start": "25:00"}, {"end": "08:90"}):
            with self.assertRaises(ValueError):
                self.rule(**changes)


class MonitoringTests(unittest.TestCase):
    setUp = runtime.RuntimeTests.setUp
    camera = runtime.RuntimeTests.camera
    auth = staticmethod(runtime.RuntimeTests.auth)

    def event(self, when, suffix="manual"):
        base = f"fixture_{int(when*1000)}_{suffix}"
        event = {"id": base, "time": when, "image": base + ".jpg"}
        (Path(self.app.EVENTS_DIR) / event["image"]).write_bytes(b"fixture")
        self.app.events.append(event)
        return event

    def test_offline_startup_and_recovery_are_not_repeated(self):
        cam = self.camera()
        cam.online = False
        with patch.object(self.app, "health_event") as emit:
            for now in (100, 159, 160, 200):
                self.app.maintenance.check_health(now)
            self.assertEqual(emit.call_count, 1)
            self.assertEqual(emit.call_args.args[1], "camera_offline")
            cam.online = True
            for now in (201, 205, 206, 210):
                cam.last_frame = now
                self.app.maintenance.check_health(now)
            self.assertEqual(emit.call_count, 2)
            self.assertEqual(emit.call_args.args[1], "camera_recovered")

    def test_stalled_frames_and_flapping_recovery(self):
        cam = self.camera()
        cam.last_frame = 100
        with patch.object(self.app, "health_event") as emit:
            self.app.maintenance.check_health(100)
            self.app.maintenance.check_health(160)
            self.assertEqual(emit.call_count, 1)
            cam.last_frame = 161
            self.app.maintenance.check_health(161)
            cam.online = False
            self.app.maintenance.check_health(164)
            cam.online = True
            cam.last_frame = 165
            self.app.maintenance.check_health(165)
            self.app.maintenance.check_health(169)
            self.assertEqual(emit.call_count, 1)
            cam.last_frame = 170
            self.app.maintenance.check_health(170)
            self.assertEqual(emit.call_count, 2)

    def test_disabled_health_alerts_and_removed_cameras(self):
        cam = self.camera()
        self.app.settings["offline_alerts"] = False
        with patch.object(self.app, "health_event") as emit:
            self.app.maintenance.check_health(100)
            self.app.maintenance.check_health(1000)
            emit.assert_not_called()
        self.app.settings["offline_alerts"] = True
        self.app.maintenance.check_health(100)
        self.app.cameras.clear()
        self.app.maintenance.check_health(200)
        self.assertEqual(self.app.maintenance.health, {})

    def test_health_events_ignore_schedule_and_do_not_record_or_sound_alarm(self):
        cam = self.camera()
        cam.cfg["schedule"] = monitoring.clean_schedule({"enabled": True})
        with patch.object(self.app, "scheduled", return_value=False), \
             patch.object(self.app, "notify") as notify, \
             patch.object(cam, "start_clip") as clip, \
             patch.object(self.app.peripherals, "on_alert") as alarm:
            self.app.health_event(cam, "camera_offline", "Camera offline")
        self.assertEqual(len(self.app.events), 1)
        self.assertIsNone(notify.call_args.args[1])
        clip.assert_not_called()
        alarm.assert_not_called()

    def test_outside_schedule_suppresses_automatic_events_but_not_manual(self):
        cam = self.camera()
        self.app.settings["record_clips"] = False
        with patch.object(self.app, "scheduled", return_value=False):
            self.app.add_event(cam, "object", "person", cam.jpeg)
            self.assertEqual(len(self.app.events), 0)
            self.app.add_event(cam, "manual", "manual recording", cam.jpeg, force=True)
        self.assertEqual(len(self.app.events), 1)

    def test_grouped_anomaly_cooldown_ignores_object_id_and_score(self):
        cam = self.camera()
        self.app.settings["record_clips"] = False
        for label in ("egg #1 rejected (score 1.50)", "egg #2 rejected (score 2.50)"):
            self.app.add_event(cam, "anomaly", label, cam.jpeg)
        self.assertEqual(len(self.app.events), 1)
        self.app.settings["group_alerts"] = False
        self.app.add_event(cam, "anomaly", "egg #3 rejected (score 3.00)", cam.jpeg)
        self.assertEqual(len(self.app.events), 2)

    def test_object_confirmation_waits_for_persistent_track(self):
        cam = self.camera()
        detector = Mock()
        detector.objects.return_value = [("person", .9, 80, 40, 160, 180)]
        with patch.object(self.app, "ai", detector):
            for when, expected in ((100, False), (101, False), (103, True), (104, False)):
                with patch.object(self.app.time, "time", return_value=when):
                    pending = cam._analyze(cam.raw.copy(), self.app.settings, cam.cfg)
                self.assertEqual(("object", "person") in pending, expected)

    def test_stale_track_is_not_treated_as_sustained_activity(self):
        tracker = self.app.Tracker()
        with patch.object(self.app.time, "time", return_value=100):
            first = tracker.update([("person", .9, 0, 0, 100, 100)])[0][0]
        with patch.object(self.app.time, "time", return_value=110):
            second, _, new = tracker.update([("person", .9, 0, 0, 100, 100)])[0]
        self.assertTrue(new)
        self.assertNotEqual(first, second)

    def test_cleanup_is_opt_in(self):
        event = self.event(1)
        self.app.maintenance.cleanup(10000000)
        self.assertTrue((Path(self.app.EVENTS_DIR) / event["image"]).exists())

    def test_retention_keeps_active_and_recent_events(self):
        now = 1000000
        expired = self.event(now - 3*86400)
        active = self.event(now - 2*86400)
        recent = self.event(now - 60)
        cam = self.camera()
        cam.clip = {"base": active["id"]}
        self.app.settings.update(storage_cleanup=True, retention_days=1)
        result = self.app.maintenance.cleanup(now)
        self.assertEqual(result["removed"], 1)
        self.assertFalse((Path(self.app.EVENTS_DIR) / expired["image"]).exists())
        self.assertEqual(list(self.app.events), [active, recent])

    def test_event_history_limit_keeps_active_recording(self):
        cam = self.camera()
        first = self.event(1)
        for number in range(2, 501):
            self.app.events.append({"id": str(number), "time": number})
        cam.clip = {"base": first["id"]}
        self.app.add_event(cam, "manual", "new event", cam.jpeg, force=True)
        self.assertEqual(len(self.app.events), 500)
        self.assertIn(first, self.app.events)
        self.assertTrue((Path(self.app.EVENTS_DIR) / first["image"]).exists())

    def test_space_cleanup_removes_oldest_only_until_reserve_is_met(self):
        old, newer = self.event(100), self.event(200)
        oldest = Path(self.app.EVENTS_DIR) / old["image"]
        self.app.settings.update(storage_cleanup=True, retention_days=0)
        with patch.object(monitoring.shutil, "disk_usage", side_effect=lambda _: SimpleNamespace(free=0 if oldest.exists() else 2*1024**3)):
            result = self.app.maintenance.cleanup(1000)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(list(self.app.events), [newer])

    def test_cleanup_refuses_paths_outside_event_directory(self):
        outside = self.root / "keep.jpg"
        outside.write_bytes(b"keep")
        self.app.events.append({"id": "unsafe", "time": 1, "image": "../keep.jpg"})
        self.app.settings.update(storage_cleanup=True, retention_days=1)
        result = self.app.maintenance.cleanup(1000000)
        self.assertTrue(outside.exists())
        self.assertTrue(result["error"])
        self.assertEqual(len(self.app.events), 1)

    def test_orphan_cleanup_leaves_unrelated_and_active_files(self):
        root = Path(self.app.EVENTS_DIR)
        stale = root / "fixture_100_manual_a1b2c3d4.tmp.mp4"
        keep = root / "family.mp4"
        active = root / "fixture_101_manual_a1b2c3d4.tmp.mp4"
        for path in (stale, keep, active):
            path.write_bytes(b"fixture")
            os.utime(path, (1, 1))
        cam = self.camera()
        cam.clip = {"base": "fixture_101_manual_a1b2c3d4"}
        self.app.settings["storage_cleanup"] = True
        self.app.maintenance.cleanup(1000000)
        self.assertFalse(stale.exists())
        self.assertTrue(keep.exists())
        self.assertTrue(active.exists())

    def backup(self):
        cfg = self.app.clean_settings({"cameras": [{"id": "fixture", "source": "0", "enabled": False,
            "zones": [{"name": "Gate", "points": [[0,0],[1,0],[1,1]]}]}]})
        self.app.settings = cfg
        self.app.training_service.state["auto_collect"] = {"fixture": {"enabled": True, "interval": 30, "confidence": .7, "split": "train"}}
        return self.client.get("/api/backup", headers=self.headers).json

    def test_backup_preview_does_not_modify_settings(self):
        backup = self.backup()
        backup["settings"]["confidence"] = .8
        response = self.client.post("/api/backup/preview", headers=self.headers, json={"backup": backup})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["cameras"], 1)
        self.assertEqual(self.app.settings["confidence"], .5)
        self.assertFalse(Path(self.app.SETTINGS_FILE).exists())

    def test_backup_restore_round_trip_and_authentication(self):
        backup = self.backup()
        backup["settings"].update(password="restored-password", confidence=.8)
        response = self.client.post("/api/backup/restore", headers=self.headers, json={"backup": backup, "confirm": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.app.load_settings()["confidence"], .8)
        self.assertEqual(self.app.settings["cameras"][0]["zones"][0]["name"], "Gate")
        self.assertEqual(self.client.get("/api/backup", headers=self.headers).status_code, 401)
        self.assertEqual(self.client.get("/api/backup", headers=self.auth("admin", "restored-password")).status_code, 200)
        state = json.loads(self.app.training_service.state_file.read_text())
        self.assertEqual(state["auto_collect"], backup["auto_collect"])

    def test_restore_rejects_unconfirmed_invalid_or_busy_requests(self):
        backup = self.backup()
        self.assertEqual(self.client.post("/api/backup/restore", headers=self.headers, json={"backup": backup}).status_code, 400)
        invalid = copy.deepcopy(backup)
        invalid["settings"]["cameras"][0]["schedule"]["timezone"] = "invalid"
        self.assertEqual(self.client.post("/api/backup/restore", headers=self.headers, json={"backup": invalid, "confirm": True}).status_code, 400)
        self.app.training_service.state["jobs"]["busy"] = {"status": "running"}
        self.assertEqual(self.client.post("/api/backup/restore", headers=self.headers, json={"backup": backup, "confirm": True}).status_code, 409)
        self.assertFalse(Path(self.app.SETTINGS_FILE).exists())

    def test_restore_write_failure_preserves_both_configurations(self):
        backup = self.backup()
        service = self.app.training_service
        service.save()
        previous = copy.deepcopy(service.state)
        backup["auto_collect"] = {}
        with patch.object(self.app, "save_settings", side_effect=OSError("disk full")), \
             patch.object(self.app, "sync_cameras") as sync:
            response = self.client.post("/api/backup/restore", headers=self.headers, json={"backup": backup, "confirm": True})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(service.state, previous)
        self.assertEqual(json.loads(service.state_file.read_text()), previous)
        sync.assert_not_called()

    def test_invalid_backup_version_and_camera_rules_rejected(self):
        backup = self.backup()
        invalids = []
        version = copy.deepcopy(backup)
        version["version"] = 999
        invalids.append(version)
        duplicate = copy.deepcopy(backup)
        duplicate["settings"]["cameras"].append(copy.deepcopy(duplicate["settings"]["cameras"][0]))
        invalids.append(duplicate)
        rules = copy.deepcopy(backup)
        rules["auto_collect"]["unknown"] = rules["auto_collect"]["fixture"]
        invalids.append(rules)
        for value in invalids:
            response = self.client.post("/api/backup/preview", headers=self.headers, json={"backup": value})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(Path(self.app.SETTINGS_FILE).exists())


if __name__ == "__main__":
    unittest.main()
