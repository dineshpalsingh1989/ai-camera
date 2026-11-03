"""Monitoring policy, camera health transitions and bounded recording retention."""
from datetime import datetime
from pathlib import Path
import re
import shutil
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def clean_schedule(value):
    if not isinstance(value, dict):
        raise ValueError("Camera schedule must be an object")
    zone = str(value.get("timezone", "UTC"))
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Use an IANA time zone, for example Asia/Singapore") from None
    days = value.get("days", list(range(7)))
    if not isinstance(days, list) or not days or any(type(d) is not int or d not in range(7) for d in days):
        raise ValueError("Select at least one schedule day")
    result = {"enabled": bool(value.get("enabled", False)), "timezone": zone, "days": sorted(set(days))}
    for key, default in (("start", "18:00"), ("end", "08:00")):
        text = str(value.get(key, default))
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", text):
            raise ValueError("Schedule times must use HH:MM")
        result[key] = text
    return result


def scheduled(cfg, now=None):
    rule = cfg.get("schedule", {})
    if not rule.get("enabled"):
        return True
    local = datetime.fromtimestamp(time.time() if now is None else now, ZoneInfo(rule["timezone"]))
    minute = local.hour * 60 + local.minute
    start, end = [int(t[:2]) * 60 + int(t[3:]) for t in (rule["start"], rule["end"])]
    day = local.weekday()
    if start == end:
        return day in rule["days"]  # equal times mean the full selected calendar day
    if start < end:
        return day in rule["days"] and start <= minute < end
    return (day in rule["days"] and minute >= start) or ((day - 1) % 7 in rule["days"] and minute < end)


def alert_key(kind, label, grouped):
    if grouped and kind == "anomaly":
        return re.sub(r" \(score [^)]*\)", "", re.sub(r" #\d+", "", label))
    return label


class Maintenance:
    def __init__(self, context):
        self.ctx = context
        self.health = {}
        self.lock = threading.Lock()
        self.storage = {"last_run": None, "removed": 0, "free_gb": None, "error": ""}

    def check_health(self, now=None):
        now = time.time() if now is None else now
        settings = self.ctx["settings"]
        cameras = dict(self.ctx["cameras"])
        self.health = {cid: state for cid, state in self.health.items() if cid in cameras}
        if not settings["offline_alerts"]:
            self.health.clear()
            return
        for cid, cam in cameras.items():
            state = self.health.setdefault(cid, {"camera": cam, "since": now, "reported": False, "recovery": None})
            if state["camera"] is not cam:
                state = self.health[cid] = {"camera": cam, "since": now, "reported": False, "recovery": None}
            fresh = cam.online and cam.last_frame is not None and now - cam.last_frame < 15
            if fresh:
                state["since"] = cam.last_frame
                if state["reported"]:
                    if state["recovery"] is None:
                        state["recovery"] = now
                    if now - state["recovery"] >= 5:
                        self.ctx["health_event"](cam, "camera_recovered", "Camera connection recovered")
                        state.update(reported=False, recovery=None)
            else:
                state["recovery"] = None
                if not state["reported"] and now - state["since"] >= settings["offline_after"]:
                    self.ctx["health_event"](cam, "camera_offline", "Camera disconnected or frames stopped")
                    state["reported"] = True

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        ctx, removed = self.ctx, 0
        settings = ctx["settings"]
        root = Path(ctx["EVENTS_DIR"]).resolve()
        reserve = settings["min_free_gb"] * 1024**3
        with self.lock, ctx["ev_lock"]:
            # Recent events also cover the interval between creating an event and opening its writer.
            protected = {clip["base"] for c in list(ctx["cameras"].values()) if (clip := c.clip)}
            def safe_file(name):
                path = root / name
                if path.is_symlink() or path.resolve().parent != root or path.suffix not in {".jpg", ".mp4"}:
                    raise ValueError("Unsafe event media path")
                return path
            error = ""
            if settings["storage_cleanup"]:
                for event in sorted(list(ctx["events"]), key=lambda e: e["time"]):
                    if event["id"] in protected or now - event["time"] < 120:
                        continue
                    expired = settings["retention_days"] > 0 and now - event["time"] >= settings["retention_days"] * 86400
                    if not expired and shutil.disk_usage(root).free >= reserve:
                        continue
                    try:
                        paths = [safe_file(event[k]) for k in ("image", "clip") if event.get(k)]
                        for path in paths:
                            path.unlink(missing_ok=True)
                        ctx["events"].remove(event)
                        removed += 1
                    except (OSError, ValueError):
                        error = "Some event files could not be removed"
                # Only clean app-named orphan media, never arbitrary files or active temporary clips.
                referenced = {e[k] for e in ctx["events"] for k in ("image", "clip") if e.get(k)}
                for path in root.iterdir():
                    if path.name in referenced or not re.fullmatch(r"\w+_\d+_[a-z_]+(?:_[0-9a-f]{8})?(?:\.tmp)?\.(?:jpg|mp4)", path.name):
                        continue
                    if any(path.name.startswith(base + ".") for base in protected):
                        continue
                    try:
                        path = safe_file(path.name)
                        if now - path.stat().st_mtime > 86400:
                            path.unlink()
                    except (OSError, ValueError):
                        error = "Some orphan files could not be removed"
                if removed:
                    ctx["save_events"]()
            self.storage = {"last_run": now, "removed": removed,
                            "free_gb": round(shutil.disk_usage(root).free / 1024**3, 2), "error": error}
            return dict(self.storage)

    def run(self):
        last_cleanup = 0
        while True:
            try:
                self.check_health()
                if time.time() - last_cleanup >= 60:
                    self.cleanup()
                    last_cleanup = time.time()
            except Exception as exc:
                self.storage["error"] = "Maintenance failed; check the server log"
                print("Monitoring maintenance failed:", exc)
            time.sleep(5)
