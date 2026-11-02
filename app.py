#!/usr/bin/env python3
"""AI Camera: multi-camera CCTV dashboard on the Hailo-8L.

Object detection + tracking, zones (intrusion / loitering), line counters, pose + fall detection,
face detection / privacy blur, heatmap, motion, event snapshots + video clips, Telegram alerts, analytics.
"""
import atexit, glob, hmac, io, json, os, re, shutil, socket, subprocess, sys, threading, time, uuid, zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2
import numpy as np
import requests
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import hardware  # IRIV EdgeAI CM5 peripherals; no-op on any other machine

BASE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE, "settings.json")
STATS_FILE = os.path.join(BASE, "stats.json")
EVENTS_DIR = os.path.join(BASE, "events")
EVENTS_FILE = os.path.join(EVENTS_DIR, "events.json")
INSPECT_DIR = os.path.join(BASE, "inspect")
TRAIN_DIR = os.path.join(BASE, "training")  # photos collected for labelling on Roboflow
os.makedirs(EVENTS_DIR, exist_ok=True)
MODELS = "/usr/local/hailo/resources/models/hailo8l/"
FONT = cv2.FONT_HERSHEY_SIMPLEX

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"]
SKELETON = [(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (12, 14), (13, 15), (14, 16)]
FEATURES = {"smooth_view": True, "objects": True, "custom": False, "pose": False, "faces": False, "blur_faces": False, "heatmap": False, "trails": True}
QUIET = {"motion"}  # low-value events: snapshot only, no clip, no Telegram

DEFAULTS = {
    "cameras": [],  # added from Settings
    "models": {"detect": MODELS + "yolov8s.hef", "pose": MODELS + "yolov8m_pose.hef", "faces": MODELS + "scrfd_10g.hef"},
    "confidence": 0.5,
    "max_fps": 15,
    "alert_classes": ["person", "car", "motorcycle", "dog", "cat"],
    "alert_cooldown": 60,
    "custom_classes": ["egg"],  # objects the 80-class model doesn't know
    "custom_model": "yolov8s-worldv2.pt",  # YOLO-World = no training; or a .pt trained on Roboflow / Ultralytics
    "custom_confidence": 0.25,
    "motion_enabled": False,  # noisy outdoors (trees, shadows); enable per site
    "motion_sensitivity": 2.0,
    "record_clips": True,
    "clip_seconds": 10,
    "telegram_token": "",
    "telegram_chat_id": "",
    "username": "admin",
    "password": "admin",
}
SECRETS = ("password", "telegram_token")
CONN_KEYS = ("type", "brand", "ip", "port", "user", "pass", "stream", "path", "usb", "http", "file", "url",
             "csicam", "csiw", "csih")  # form fields the UI builds the source from


# ---------------------------------------------------------------- settings

def load_settings():
    s = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            s.update(json.load(f))
    for c in s["cameras"]:  # settings from the first version have no AI fields yet
        c["features"] = {**FEATURES, **c.get("features", {})}
        c.setdefault("zones", [])
        c.setdefault("lines", [])
        c.setdefault("group", "")
        c["inspection"] = clean_inspection(c.get("inspection") or {}, COCO + s["custom_classes"])
    return s


def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)
    os.chmod(SETTINGS_FILE, 0o600)  # holds the password and Telegram token


def num(v, lo, hi, kind=float):
    return min(max(kind(v), lo), hi)


def points(v, n_min, n_max):
    pts = [[num(p[0], 0, 1), num(p[1], 0, 1)] for p in v][:n_max]
    if len(pts) < n_min:
        raise ValueError("shape needs more points")
    return pts


def clean_inspection(d, known):
    """Quality inspection (anomaly detection) settings for one camera."""
    window = num(d.get("window", 4), 1, 30, int)
    return {"enabled": bool(d.get("enabled", False)),
            "roi": points(d.get("roi") or [[0.2, 0.2], [0.8, 0.8]], 2, 2),
            "target": d.get("target") if d.get("target") in known else "area",
            "warn": num(d.get("warn", 1.15), 0.5, 10), "reject": num(d.get("reject", 1.4), 0.5, 10),
            "votes": min(num(d.get("votes", 2), 1, 30, int), window), "window": window,
            "learn_frames": num(d.get("learn_frames", 40), 10, 500, int)}


def clean_settings(d):
    """Validate settings posted from the browser; missing fields keep their current value."""
    s = json.loads(json.dumps(settings))
    model_path = str(d.get("custom_model") or s["custom_model"]).strip()
    normalize_class = str.lower if "world" in os.path.basename(model_path).lower() else lambda value: value
    s["custom_classes"] = list(dict.fromkeys(
        x for x in (normalize_class(str(v).strip())[:80] for v in d.get("custom_classes", s["custom_classes"])) if x))[:200]
    s["custom_model"] = model_path
    s["custom_confidence"] = num(d.get("custom_confidence", s["custom_confidence"]), 0.01, 0.95)
    known = COCO + s["custom_classes"]
    cams = []
    for c in d.get("cameras", s["cameras"]):
        src = str(c.get("source", "")).strip()
        if not src:
            continue
        cams.append({
            "id": re.sub(r"\W", "", str(c.get("id", ""))) or uuid.uuid4().hex[:6],
            "name": str(c.get("name") or "Camera")[:40],
            "location": str(c.get("location") or "")[:60],
            "group": str(c.get("group") or "").strip()[:60],
            "conn": {k: str(v)[:300] for k, v in (c.get("conn") or {}).items() if k in CONN_KEYS},
            "source": src,
            "enabled": bool(c.get("enabled", True)),
            "features": {k: bool(c.get("features", {}).get(k, v)) for k, v in FEATURES.items()},
            "zones": [{"name": str(z.get("name") or "Zone")[:30], "points": points(z["points"], 3, 30),
                       "classes": [x for x in z.get("classes", []) if x in known],
                       "dwell": num(z.get("dwell", 0), 0, 3600, int)} for z in c.get("zones", [])],
            "lines": [{"name": str(ln.get("name") or "Line")[:30], "points": points(ln["points"], 2, 2),
                       "classes": [x for x in ln.get("classes", []) if x in known]} for ln in c.get("lines", [])],
            "inspection": clean_inspection(c.get("inspection") or {}, known),
        })
    s["cameras"] = cams
    for k in ("detect", "pose", "faces"):
        s["models"][k] = str(d.get("models", {}).get(k) or s["models"][k]).strip()
    s["confidence"] = num(d.get("confidence", s["confidence"]), 0.05, 0.95)
    s["max_fps"] = num(d.get("max_fps", s["max_fps"]), 1, 30, int)
    s["alert_classes"] = [c for c in d.get("alert_classes", s["alert_classes"]) if c in known]
    s["alert_cooldown"] = num(d.get("alert_cooldown", s["alert_cooldown"]), 1, 3600, int)
    s["motion_enabled"] = bool(d.get("motion_enabled", s["motion_enabled"]))
    s["motion_sensitivity"] = num(d.get("motion_sensitivity", s["motion_sensitivity"]), 0.1, 50)
    s["record_clips"] = bool(d.get("record_clips", s["record_clips"]))
    s["clip_seconds"] = num(d.get("clip_seconds", s["clip_seconds"]), 3, 120, int)
    s["telegram_chat_id"] = str(d.get("telegram_chat_id", s["telegram_chat_id"])).strip()
    s["username"] = str(d.get("username") or s["username"])
    for k in SECRETS:  # blank = keep the stored secret
        if d.get(k):
            s[k] = str(d[k]).strip()
    return s


# ---------------------------------------------------------------- Hailo models

class Hailo:
    """One VDevice for all models; models load on first use."""

    def __init__(self):
        from hailo_platform import HailoSchedulingAlgorithm, VDevice
        p = VDevice.create_params()
        p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        p.group_id = "SHARED"
        try:  # go through hailort.service when it runs, so other Hailo apps can share the chip
            p.multi_process_service = True
            self.device = VDevice(p)
        except Exception:
            p.multi_process_service = False
            self.device = VDevice(p)
        self.models = {}
        self.lock = threading.Lock()

    def run(self, hef, img):
        from hailo_platform import FormatType
        with self.lock:
            if hef not in self.models:
                m = self.device.create_infer_model(hef)
                for o in m.outputs:
                    o.set_format_type(FormatType.FLOAT32)
                ctx = m.configure()
                self.models[hef] = (m, ctx, ctx.__enter__())  # keep ctx referenced or HailoRT tears it down
            m, _, cfg = self.models[hef]
            b = cfg.create_bindings(output_buffers={o.name: np.empty(o.shape, np.float32) for o in m.outputs})
            b.input().set_buffer(img)
            cfg.run([b], 5000)
            return {o.name: b.output(o.name).get_buffer() for o in m.outputs}

    label = "Hailo"

    def objects(self, f, conf):
        return detect_objects(*letterbox(f), conf)

    def poses(self, f, conf):
        return detect_poses(*letterbox(f), conf)

    def faces(self, f):
        return detect_faces(*letterbox(f))


class CustomDetector:
    """Extra objects the 80-class model doesn't know, e.g. "egg". Runs on CPU with Ultralytics.
    A YOLO-World model (*world*.pt) finds them zero-shot from their names - no training.
    Any other .pt (exported from Roboflow or trained with Ultralytics) uses its own trained classes."""

    def __init__(self, path, classes):
        from ultralytics import YOLO
        self.model = YOLO(path)
        if "world" in os.path.basename(path).lower():
            self.model.set_classes(list(classes))
        self.key, self.lock = (path, tuple(classes)), threading.Lock()

    def objects(self, f, conf):
        with self.lock:
            b = self.model(f, conf=conf, verbose=False)[0].boxes
        names = self.model.names
        return [(names[int(c)], float(sc), *map(int, xy)) for xy, sc, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]


custom_det, custom_error, custom_lock = None, "", threading.Lock()


def get_custom():
    """The custom-object detector for the current settings, (re)loaded when they change."""
    global custom_det, custom_error
    key = (settings["custom_model"], tuple(settings["custom_classes"]))
    with custom_lock:
        if (custom_det is None or custom_det.key != key) and custom_error != key:
            try:
                custom_det, custom_error = CustomDetector(*key), ""
            except Exception as e:
                custom_det, custom_error = None, key  # remember the failure so it isn't retried every frame
                print("custom objects unavailable:", e)
    return custom_det if custom_det and custom_det.key == key else None


class CpuAI:
    """shortcut: CPU fallback for PCs without a Hailo (demo / development); a few FPS only. Needs: pip install ultralytics"""
    label = "CPU"

    def __init__(self):
        from ultralytics import YOLO
        self.det, self.pose = YOLO("yolov8n.pt"), YOLO("yolov8n-pose.pt")
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.lock = threading.Lock()

    def objects(self, f, conf):
        with self.lock:
            b = self.det(f, conf=conf, verbose=False)[0].boxes
        return [(COCO[int(c)], float(sc), *map(int, xy)) for xy, sc, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]

    def poses(self, f, conf):
        with self.lock:
            r = self.pose(f, conf=conf, verbose=False)[0]
        if r.keypoints is None:
            return []
        return list(zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()))

    def faces(self, f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return [(x, y, x + w, y + h) for x, y, w, h in self.face.detectMultiScale(g, 1.1, 5, minSize=(40, 40))]


def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    s = size / max(h, w)
    img = np.zeros((size, size, 3), np.uint8)  # pad bottom/right so model coords map back with one scale
    img[:int(h * s), :int(w * s)] = cv2.cvtColor(cv2.resize(frame, (int(w * s), int(h * s))), cv2.COLOR_BGR2RGB)
    return img, s


def detect_objects(img, s, conf):
    out = next(iter(ai.run(settings["models"]["detect"], img).values()))  # on-chip NMS: per-class [y1,x1,y2,x2,score]
    k = img.shape[0] / s
    return [(COCO[c], float(sc), int(x1 * k), int(y1 * k), int(x2 * k), int(y2 * k))
            for c, arr in enumerate(out) for y1, x1, y2, x2, sc in arr if sc >= conf]


def nms(boxes, scores, conf):
    idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                           scores.tolist(), conf, 0.5)
    return np.array(idx, int).flatten()


def detect_poses(img, s, conf):
    """YOLOv8-pose decode (DFL boxes + 17 keypoints) for the 3 raw output scales."""
    by = {(a.shape[0], a.shape[2]): a for a in ai.run(settings["models"]["pose"], img).values()}
    boxes, scores, kpts = [], [], []
    for g in (80, 40, 20):
        stride = img.shape[0] // g
        sc = by[(g, 1)].reshape(-1)
        keep = sc >= conf
        if not keep.any():
            continue
        gy, gx = np.divmod(np.nonzero(keep)[0], g)
        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
        d = by[(g, 64)].reshape(-1, 4, 16)[keep]
        e = np.exp(d - d.max(-1, keepdims=True))
        dist = (e / e.sum(-1, keepdims=True)) @ np.arange(16) * stride
        boxes.append(np.stack([cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]], 1))
        kp = by[(g, 51)].reshape(-1, 17, 3)[keep].copy()
        kp[..., 0] = (kp[..., 0] * 2 - 0.5) * stride + cx[:, None]
        kp[..., 1] = (kp[..., 1] * 2 - 0.5) * stride + cy[:, None]
        kp[..., 2] = 1 / (1 + np.exp(-kp[..., 2]))
        scores.append(sc[keep])
        kpts.append(kp)
    if not boxes:
        return []
    boxes, scores, kpts = np.concatenate(boxes) / s, np.concatenate(scores), np.concatenate(kpts)
    kpts[..., :2] /= s
    return [(boxes[i], kpts[i]) for i in nms(boxes, scores, conf)]


def detect_faces(img, s, conf=0.5):
    """SCRFD decode: 2 anchors per grid cell, distances in stride units."""
    by = {(a.shape[0], a.shape[2]): a for a in ai.run(settings["models"]["faces"], img).values()}
    boxes, scores = [], []
    for g in (80, 40, 20):
        stride = img.shape[0] // g
        sc = by[(g, 2)].reshape(-1)
        keep = sc >= conf
        if not keep.any():
            continue
        gy, gx = np.divmod(np.nonzero(keep)[0] // 2, g)
        cx, cy = gx * stride, gy * stride
        d = by[(g, 8)].reshape(-1, 4)[keep] * stride
        boxes.append(np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1))
        scores.append(sc[keep])
    if not boxes:
        return []
    boxes, scores = np.concatenate(boxes) / s, np.concatenate(scores)
    return [boxes[i].astype(int) for i in nms(boxes, scores, conf)]


def is_fallen(box, kp):
    # shortcut: posture heuristic (torso near horizontal + wide box); swap for a trained fall model if it misfires
    x1, y1, x2, y2 = box
    sh, hp = kp[[5, 6]], kp[[11, 12]]
    if min(sh[:, 2].min(), hp[:, 2].min()) < 0.3:
        return False
    dx, dy = hp[:, :2].mean(0) - sh[:, :2].mean(0)
    return abs(dx) > 1.5 * abs(dy) and (x2 - x1) > (y2 - y1)


# ---------------------------------------------------------------- quality inspection (anomaly detection)

class AnomalyModel:
    """PatchCore-style anomaly detection: learns patch features of GOOD images only, then a patch's
    anomaly = distance to its nearest good patch. Scores are scaled so 1.0 = the worst seen on good images.
    Runs on CPU with PyTorch (ResNet-18 backbone)."""
    SIZE = 128  # per-object crops (items are small); a whole area keeps its aspect ratio (see area_size)
    _backbone, _lock = None, threading.Lock()

    def __init__(self, bank, norm, meta, info):
        self.bank, self.norm, self.meta, self.info = bank, norm, meta, info

    @classmethod
    def features(cls, crops, size):
        import torch
        import torch.nn.functional as F
        import torchvision
        with cls._lock:
            if cls._backbone is None:
                cls._backbone = torchvision.models.resnet18(weights="DEFAULT").eval()
            m = cls._backbone
            x = np.stack([cv2.resize(c, size)[:, :, ::-1] for c in crops]).astype(np.float32) / 255
            x = torch.from_numpy(x).permute(0, 3, 1, 2)
            x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            with torch.no_grad():
                l2 = m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))
                l3 = m.layer3(l2)
                f = torch.cat([l2, F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)], 1)
                return F.avg_pool2d(f, 3, 1, 1)  # neighbourhood context, as in PatchCore

    def distances(self, f):
        import torch
        b, c, h, w = f.shape
        q = f.permute(0, 2, 3, 1).reshape(-1, c)
        d = torch.cat([torch.cdist(q[i:i + 4096], self.bank).min(1).values for i in range(0, len(q), 4096)])
        return d.reshape(b, h, w).numpy()

    @classmethod
    def fit(cls, crops, meta, size):
        import torch
        cal = crops[::5]  # every 5th good image is held back to calibrate what "normal" scores look like
        train = [c for i, c in enumerate(crops) if i % 5] or crops
        feats = torch.cat([cls.features(train[i:i + 16], size) for i in range(0, len(train), 16)])
        bank = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        if len(bank) > 20000:  # shortcut: random subsample instead of greedy coreset; fine for one product
            bank = bank[torch.randperm(len(bank))[:20000]]
        model = cls(bank, 1.0, meta, {"images": len(crops), "time": time.time(), "size": list(size)})
        model.norm = max(float(model.distances(cls.features(cal, size)).max()), 1e-6)
        return model

    def score(self, crops):
        maps = self.distances(self.features(crops, tuple(self.info["size"]))) / self.norm
        return [(float(m.max()), m) for m in maps]

    def save(self, cid):
        import torch
        os.makedirs(INSPECT_DIR, exist_ok=True)
        torch.save({"bank": self.bank, "norm": self.norm, "meta": self.meta, "info": self.info},
                   os.path.join(INSPECT_DIR, cid + ".pt"))

    @classmethod
    def load(cls, cid):
        p = os.path.join(INSPECT_DIR, cid + ".pt")
        if not os.path.exists(p):
            return None
        import torch
        d = torch.load(p)
        return cls(d["bank"], d["norm"], d["meta"], d["info"])


def area_size(w, h, long=640):
    """Model input size for a whole inspection area: keep aspect so small defects survive the resize."""
    s = long / max(w, h)
    return max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)


SEVERITY_COLOR = {"ok": (60, 170, 60), "warn": (0, 165, 255), "reject": (0, 0, 255)}


# ---------------------------------------------------------------- tracking & geometry

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class Tracker:
    # shortcut: greedy IoU matching; fast objects at low FPS may get a new ID - use ByteTrack if counts drift
    def __init__(self):
        self.next_id, self.tracks = 1, {}

    def update(self, dets):
        now, free, out = time.time(), set(self.tracks), []
        for label, score, *box in sorted(dets, key=lambda d: -d[1]):
            best, best_iou = None, 0.25
            for tid in free:
                t = self.tracks[tid]
                v = iou(t["box"], box) if t["label"] == label else 0
                if v > best_iou:
                    best, best_iou = tid, v
            new = best is None
            if new:
                best, self.next_id = self.next_id, self.next_id + 1
                self.tracks[best] = {"label": label, "first": now, "trail": deque(maxlen=40), "zones": {}, "done": set()}
            free.discard(best)
            t = self.tracks[best]
            t.update(box=box, score=score, seen=now)
            t["trail"].append(((box[0] + box[2]) // 2, box[3]))  # bottom-centre = where the object stands
            out.append((best, t, new))
        for tid in free:
            if now - self.tracks[tid]["seen"] > 5:  # survive detection flicker so "unique today" isn't inflated
                del self.tracks[tid]
        return out


def side(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed(p1, p2, a, b):
    return side(a, b, p1) * side(a, b, p2) < 0 and side(p1, p2, a) * side(p1, p2, b) < 0


def tag(img, text, x, y, color, fs):
    (tw, th), _ = cv2.getTextSize(text, FONT, fs, 1)
    x, y = max(0, int(x)), max(int(y), th + 4)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- stats

def today():
    return time.strftime("%Y-%m-%d")


def load_stats():
    try:
        with open(STATS_FILE) as f:
            st = json.load(f)
        if st.get("date") == today():
            return st
    except (OSError, ValueError):
        pass
    return {"date": today(), "unique": {}, "lines": {}, "kinds": {}}


stats, stats_lock = load_stats(), threading.Lock()


def bump(section, key, sub=None):
    global stats
    with stats_lock:
        if stats["date"] != today():
            stats = {"date": today(), "unique": {}, "lines": {}, "kinds": {}}
        d = stats[section]
        if sub:
            d = d.setdefault(key, {"in": 0, "out": 0})
            key = sub
        d[key] = d.get(key, 0) + 1


def stats_saver():
    while True:
        time.sleep(30)
        with stats_lock:
            data = json.dumps(stats)
        with open(STATS_FILE, "w") as f:
            f.write(data)


# ---------------------------------------------------------------- events, clips, notifications

def load_events():
    try:
        with open(EVENTS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


events, ev_lock, last_alert = deque(load_events(), maxlen=500), threading.Lock(), {}


def save_events():
    with open(EVENTS_FILE, "w") as f:
        json.dump(list(events), f)


def remove_files(ev):
    for k in ("image", "clip"):
        if ev.get(k):
            try:
                os.remove(os.path.join(EVENTS_DIR, ev[k]))
            except OSError:
                pass


def notify(text, jpg):
    tok, chat = settings["telegram_token"], settings["telegram_chat_id"]
    if not (tok and chat):
        return

    def send():
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendPhoto", data={"chat_id": chat, "caption": text},
                          files={"photo": ("event.jpg", jpg)}, timeout=15)
        except requests.RequestException as e:
            print("Telegram failed:", e)

    threading.Thread(target=send, daemon=True).start()


def sensor_event(signal):
    """A digital input went active: raise an event (with the first live camera's frame if any)."""
    cam = next((c for c in cameras.values() if c.online and c.jpeg), None)
    if cam:
        add_event(cam, "input", f"{signal} input triggered", cam.jpeg, force=True)
    else:
        peripherals.on_alert(signal)  # no camera online, still sound the alarm


def add_event(cam, kind, label, jpg, force=False):
    now = time.time()
    key = (cam.id, kind, label)
    if not force and now - last_alert.get(key, 0) < settings["alert_cooldown"]:
        return
    last_alert[key] = now
    base = f"{cam.id}_{int(now * 1000)}_{kind}"
    with open(os.path.join(EVENTS_DIR, base + ".jpg"), "wb") as f:
        f.write(jpg)
    ev = {"id": base, "time": now, "camera": cam.name, "camera_id": cam.id, "type": kind, "label": label,
          "image": base + ".jpg"}
    with ev_lock:
        if len(events) == events.maxlen:
            remove_files(events[0])
        events.append(ev)
        save_events()
    bump("kinds", kind)
    if kind not in QUIET:
        notify(f"{cam.name}: {label}", jpg)
        peripherals.on_alert(label)  # buzzer + siren/relay on the IRIV board; no-op elsewhere
        if settings["record_clips"] and cam.clip is None and shutil.disk_usage(EVENTS_DIR).free > 1 << 30:
            cam.start_clip(base, ev)


# ---------------------------------------------------------------- cameras

def open_capture(src):
    if src.isdigit():  # USB webcam index; DirectShow opens in ~1 s on Windows vs ~8 s for the default
        return cv2.VideoCapture(int(src), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if "appsink" in src:  # GStreamer pipeline, e.g. a CSI / Pi camera via libcamera (needs OpenCV built with GStreamer)
        return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(src, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])


class Camera:
    def __init__(self, cfg):
        self.cfg, self.id = cfg, cfg["id"]
        self.frame = self.jpeg = self.raw = self.heat = self.clip = self.overlay = None
        self.paused = False  # AI paused from the dashboard (video keeps streaming)
        self.collecting = None  # [saved, wanted] while saving training photos
        self.online, self.motion, self.fps, self.fs = False, False, 0.0, 0.5
        self.last_frame, self.last_connected, self.reconnects = None, None, 0
        self.counts, self.zone_counts, self.fall_since = {}, {}, None
        self.last_dets = []  # most recent object detections, reused as suggested boxes when capturing a training photo
        self.prebuf = deque(maxlen=45)  # ~3 s of JPEGs so clips start before the event
        self.tracker = Tracker()
        self.learning, self.insp_hist, self.insp_flagged = None, {}, set()
        try:
            self.anomaly = AnomalyModel.load(self.id)
        except Exception as e:
            self.anomaly = None
            print(f"[{self.id}] could not load inspection model: {e}")
        self.insp = {"state": "ready" if self.anomaly else "not trained", "score": None, "severity": None}
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()
        threading.Thread(target=self._process, daemon=True).start()

    @property
    def name(self):
        return self.cfg["name"]

    def _grab(self):
        # Separate reader thread keeps only the newest frame, so RTSP never lags behind live.
        is_file = os.path.isfile(self.cfg["source"])  # video files loop at their own speed (demo / testing)
        shown = 0.0
        while self.running:
            cap = open_capture(self.cfg["source"])
            if cap.isOpened():
                self.last_connected = time.time()
            delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if is_file else 0
            while self.running and cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                if time.time() - shown >= 1 / 15:  # smooth live view (~15 fps), independent of how fast the AI runs
                    shown = time.time()
                    self._show(f)  # before handing f to the AI thread, which draws on it
                self.frame, self.online = f, True
                self.last_frame = time.time()
                time.sleep(delay)
            cap.release()
            if self.running and not is_file:
                self.online = False
                self.reconnects += 1
                time.sleep(3)

    def _show(self, f):
        """Live view frame: the newest camera picture with the AI's latest drawings pasted on top."""
        ov = self.overlay
        if ov is None or ov[0].shape != f.shape:
            return  # no AI frame yet, or privacy blur is on: the AI loop provides the view itself
        img, mask = ov
        f = f.copy()
        f[mask] = img[mask]
        self.jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def start_clip(self, base, ev):
        h, w = self.raw.shape[:2]
        tmp = os.path.join(EVENTS_DIR, base + ".tmp.mp4")
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"avc1"), max(1.0, min(self.fps, 30.0)), (w, h))
        if not writer.isOpened():
            return
        for j in self.prebuf:
            fr = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
            writer.write(cv2.resize(fr, (w, h)) if fr.shape[:2] != (h, w) else fr)
        self.clip = {"writer": writer, "size": (w, h), "tmp": tmp, "base": base, "ev": ev,
                     "end": time.time() + settings["clip_seconds"]}

    def _write_clip(self, f):
        c = self.clip
        c["writer"].write(cv2.resize(f, c["size"]) if f.shape[1::-1] != c["size"] else f)
        if time.time() >= c["end"] or not self.running:
            c["writer"].release()
            self.clip = None
            try:
                os.replace(c["tmp"], os.path.join(EVENTS_DIR, c["base"] + ".mp4"))
            except OSError:  # events were cleared while recording
                return
            with ev_lock:
                c["ev"]["clip"] = c["base"] + ".mp4"
                save_events()

    def _process(self):
        bg = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
        warmup, last = 30, time.time()
        while self.running:
            f = self.frame
            if f is None:
                time.sleep(0.01)
                continue
            self.frame = None
            t0, s, cfg = time.time(), settings, self.cfg
            if self.paused:
                self.raw, pending, self.counts, self.zone_counts = f.copy(), [], {}, {}
                self.fs = max(0.45, f.shape[1] / 1400)
                tag(f, "AI PAUSED", 8, 30, (90, 90, 90), self.fs)
            else:
                try:
                    pending = self._analyze(f, s, cfg)
                except Exception as e:  # keep streaming even if a model fails
                    print(f"[{self.id}] AI error: {e}")
                    pending = []

            mask = bg.apply(cv2.resize(f, (320, max(1, 320 * f.shape[0] // f.shape[1]))))
            warmup = max(0, warmup - 1)  # background model needs a few frames before motion means anything
            self.motion = bool(s["motion_enabled"] and not warmup and not self.paused
                               and cv2.countNonZero(mask) * 100 / mask.size >= s["motion_sensitivity"])
            if self.motion:
                pending.append(("motion", "motion"))
                tag(f, "MOTION", 8, 30, (0, 120, 255), self.fs)
            cv2.putText(f, f"{self.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}", (8, f.shape[0] - 10),
                        FONT, self.fs, (255, 255, 255), 1, cv2.LINE_AA)

            ai_jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.prebuf.append(ai_jpeg)
            # smooth view pastes these drawings on newer frames; not with face blur (a face could show) or when off
            if cfg["features"]["blur_faces"] or not cfg["features"]["smooth_view"]:
                self.overlay, self.jpeg = None, ai_jpeg
            else:
                self.overlay = (f, np.any(f != self.raw, axis=2))
                if self.jpeg is None:
                    self.jpeg = ai_jpeg
            for kind, label in pending:
                add_event(self, kind, label, ai_jpeg)
            if self.clip:
                self._write_clip(f)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
            time.sleep(max(0, 1 / s["max_fps"] - (now - t0)))
        if self.clip:
            self._write_clip(self.raw)

    def _analyze(self, f, s, cfg):
        """Run every enabled AI feature on frame f, draw the results on it, return events to raise."""
        h, w = f.shape[:2]
        self.fs = fs = max(0.45, w / 1400)
        self.raw = f.copy()
        feat, pending, now = cfg["features"], [], time.time()
        if self.heat is None or self.heat.shape != (h // 8, w // 8):
            self.heat = np.zeros((h // 8, w // 8), np.float32)
        if not ai:
            return pending

        # faces first so the privacy blur is applied before anything is drawn on top
        faces = ai.faces(f) if feat["faces"] or feat["blur_faces"] else []
        for x1, y1, x2, y2 in faces:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if feat["blur_faces"]:
                roi = f[y1:y2, x1:x2]
                f[y1:y2, x1:x2] = cv2.resize(cv2.resize(roi, (8, 8)), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            else:
                cv2.rectangle(f, (x1, y1), (x2, y2), (255, 200, 0), 2)

        # heatmap of where things stand
        self.heat *= 0.999
        if feat["heatmap"]:
            hm = cv2.resize(cv2.GaussianBlur(self.heat, (0, 0), 3), (w, h))
            n = (hm / (hm.max() + 1e-6) * 255).astype(np.uint8)
            m = n > 25
            f[m] = (f[m] * 0.5 + cv2.applyColorMap(n, cv2.COLORMAP_JET)[m] * 0.5).astype(np.uint8)

        # zones
        zones = [(z, np.int32([[x * w, y * h] for x, y in z["points"]])) for z in cfg["zones"]]
        zone_counts = {z["name"]: 0 for z in cfg["zones"]}
        lines = [(ln, [(int(x * w), int(y * h)) for x, y in ln["points"]]) for ln in cfg["lines"]]

        # objects + tracking
        counts = {}
        dets = ai.objects(f, s["confidence"]) if feat["objects"] else []
        if feat["custom"] and s["custom_classes"] and get_custom():
            dets += get_custom().objects(f, s["custom_confidence"])
        self.last_dets = dets  # keep for on-demand capture (suggested boxes), matched to self.raw of this frame
        try:
            training_service.sample(self.id, self.raw, dets)
        except Exception:
            pass  # collection must never interrupt monitoring; library capacity is shown in Training
        tracked = self.tracker.update(dets)
        for tid, t, new in tracked:
            label, (x1, y1, x2, y2), foot = t["label"], t["box"], t["trail"][-1]
            counts[label] = counts.get(label, 0) + 1
            self.heat[min(foot[1] // 8, h // 8 - 1), min(foot[0] // 8, w // 8 - 1)] += 1
            if new:
                bump("unique", label)
                if label in s["alert_classes"]:
                    pending.append(("object", label))
            for z, poly in zones:
                if z["classes"] and label not in z["classes"]:
                    continue
                if cv2.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0:
                    zone_counts[z["name"]] += 1
                    since = t["zones"].setdefault(z["name"], now)
                    if ("in", z["name"]) not in t["done"]:
                        t["done"].add(("in", z["name"]))
                        pending.append(("intrusion", f"{label} entered {z['name']}"))
                    if z["dwell"] and now - since >= z["dwell"] and ("dwell", z["name"]) not in t["done"]:
                        t["done"].add(("dwell", z["name"]))
                        pending.append(("loitering", f"{label} in {z['name']} for {z['dwell']}s"))
                else:
                    t["zones"].pop(z["name"], None)
            if len(t["trail"]) >= 2:
                for ln, (a, b) in lines:
                    if (not ln["classes"] or label in ln["classes"]) and crossed(t["trail"][-2], foot, a, b):
                        bump("lines", f"{self.id}|{ln['name']}", "in" if side(a, b, foot) > 0 else "out")
            color = (40, 40, 230) if label in s["alert_classes"] else (60, 170, 60)
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)
            tag(f, f"{label} #{tid} {t['score']:.0%}", x1, y1, color, fs)
            if feat["trails"] and len(t["trail"]) > 1:
                cv2.polylines(f, [np.int32(t["trail"])], False, color, 2)
        if faces:
            counts["face"] = len(faces)

        # pose + fall detection
        if feat["pose"]:
            fallen = False
            for box, kp in ai.poses(f, max(0.4, s["confidence"])):
                for i, j in SKELETON:
                    if kp[i, 2] > 0.4 and kp[j, 2] > 0.4:
                        cv2.line(f, tuple(map(int, kp[i, :2])), tuple(map(int, kp[j, :2])), (0, 220, 255), 2)
                if is_fallen(box, kp):
                    fallen = True
                    tag(f, "FALL?", box[0], box[3], (0, 0, 255), fs * 1.3)
            self.fall_since = (self.fall_since or now) if fallen else None
            if self.fall_since and now - self.fall_since >= 2:  # must stay down 2 s to count
                pending.append(("fall", "person fell"))

        for z, poly in zones:
            busy = zone_counts[z["name"]] > 0
            cv2.polylines(f, [poly], True, (0, 0, 255) if busy else (255, 160, 0), 2)
            tag(f, f"{z['name']}: {zone_counts[z['name']]}", *poly[0], (0, 0, 255) if busy else (180, 110, 0), fs)
        with stats_lock:
            line_stats = dict(stats["lines"])
        for ln, (a, b) in lines:
            c = line_stats.get(f"{self.id}|{ln['name']}", {"in": 0, "out": 0})
            cv2.line(f, a, b, (255, 0, 255), 2)
            tag(f, f"{ln['name']}  in {c['in']} / out {c['out']}", *a, (150, 0, 150), fs)

        if cfg["inspection"]["enabled"]:
            self._inspect(f, cfg["inspection"], tracked, pending, fs)
        self.counts, self.zone_counts = counts, zone_counts
        return pending

    def _inspect(self, f, insp, tracked, pending, fs):
        """Score the inspection area (or each tracked object of the target class) against the learned good images."""
        h, w = f.shape[:2]
        (ax, ay), (bx, by) = insp["roi"]
        rx1, ry1, rx2, ry2 = int(min(ax, bx) * w), int(min(ay, by) * h), int(max(ax, bx) * w), int(max(ay, by) * h)
        if insp["target"] == "area":
            items = [("area", (rx1, ry1, rx2, ry2))]
        else:
            items = [(tid, t["box"]) for tid, t, _ in tracked if t["label"] == insp["target"]
                     and rx1 <= (t["box"][0] + t["box"][2]) / 2 <= rx2 and ry1 <= (t["box"][1] + t["box"][3]) / 2 <= ry2]
        items = [(k, (max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3]))) for k, b in items]
        items = [(k, b) for k, b in items if b[2] - b[0] >= 16 and b[3] - b[1] >= 16]
        crops = [f[b[1]:b[3], b[0]:b[2]].copy() for _, b in items]
        meta = {"roi": insp["roi"], "target": insp["target"]}
        size = area_size(rx2 - rx1, ry2 - ry1) if insp["target"] == "area" else (AnomalyModel.SIZE, AnomalyModel.SIZE)

        if self.learning is not None:
            self.learning.extend(crops)
            n, goal = len(self.learning), insp["learn_frames"]
            cv2.rectangle(f, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2)
            tag(f, f"LEARNING GOOD {min(n, goal)}/{goal}", rx1, ry1, (0, 140, 200), fs)
            self.insp.update(state=f"learning {min(n, goal)}/{goal}", score=None, severity=None)
            if n >= goal:
                good, self.learning = self.learning, None
                self.insp["state"] = "training"
                threading.Thread(target=self._train, args=(good, meta, size), daemon=True).start()
            return

        cv2.rectangle(f, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        if self.anomaly is None or self.anomaly.meta != meta:
            busy = self.insp["state"] == "training" or self.insp["state"].startswith("learning failed")
            state = self.insp["state"] if busy else "not trained" if self.anomaly is None else "area changed - learn again"
            self.insp.update(state=state, score=None, severity=None)
            tag(f, f"INSPECTION: {state.upper()}", rx1, ry1, (120, 120, 0), fs)
            return
        if not crops:
            self.insp.update(state="ready", score=None, severity=None)
            tag(f, "INSPECTION: WAITING FOR ITEMS", rx1, ry1, (120, 120, 0), fs)
            return

        worst, keys = (0.0, "ok"), set()
        for (key, (x1, y1, x2, y2)), (sc, amap) in zip(items, self.anomaly.score(crops)):
            keys.add(key)
            sev = "reject" if sc >= insp["reject"] else "warn" if sc >= insp["warn"] else "ok"
            hist = self.insp_hist.setdefault(key, deque(maxlen=insp["window"]))
            hist.append(sev == "reject")
            final = "reject" if sum(hist) >= insp["votes"] else "warn" if sev != "ok" else "ok"
            color = SEVERITY_COLOR[final]

            # heatmap only where it looks off
            m = cv2.GaussianBlur(cv2.resize(amap, (x2 - x1, y2 - y1)), (0, 0), 4)
            hot = m >= insp["warn"]
            if hot.any():
                roi = f[y1:y2, x1:x2]
                n = np.clip(m / insp["reject"] * 200, 0, 255).astype(np.uint8)
                roi[hot] = (roi[hot] * 0.45 + cv2.applyColorMap(n, cv2.COLORMAP_JET)[hot] * 0.55).astype(np.uint8)

            if key == "area":  # outline each anomalous spot inside the area
                if final != "ok":
                    mask = cv2.dilate(hot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))  # ring around the spot
                    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                        if cv2.contourArea(c) >= 40:
                            c = c + np.int32([x1, y1])
                            cv2.drawContours(f, [c], -1, color, 2)
                            cx_, cy_, _, _ = cv2.boundingRect(c)
                            tag(f, "ANOMALY", cx_, cy_ - 4, color, fs)
                tag(f, f"INSPECTION {final.upper()} {sc:.2f}", x1, y1, color, fs)
            else:
                cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)
                tag(f, f"{final.upper()} {sc:.2f}", x1, y2 + 22, color, fs)

            if final == "reject" and key not in self.insp_flagged:
                self.insp_flagged.add(key)
                what = "item" if key == "area" else f"{insp['target']} #{key}"
                pending.append(("anomaly", f"{what} rejected (score {sc:.2f})"))
            elif final == "ok":
                self.insp_flagged.discard(key)  # the next bad item raises a new alert
            worst = max(worst, (sc, final))
        self.insp_hist = {k: v for k, v in self.insp_hist.items() if k in keys}
        self.insp.update(state="ready", score=round(worst[0], 2), severity=worst[1])

    def _train(self, crops, meta, size):
        try:
            model = AnomalyModel.fit(crops, meta, size)
            model.save(self.id)
            self.anomaly, self.insp_hist, self.insp_flagged = model, {}, set()
            self.insp.update(state="ready")
        except Exception as e:
            self.insp.update(state=f"learning failed: {e}")
            print(f"[{self.id}] inspection learning failed: {e}")


settings = load_settings()
ai, ai_error, cameras = None, "", {}
peripherals = hardware.Peripherals(False)  # real board peripherals attached at startup (see __main__)


def start_ai():
    global ai, ai_error
    try:
        ai, ai_error = Hailo(), ""
    except Exception as e:
        print("Hailo unavailable, trying CPU:", e)
        try:
            ai, ai_error = CpuAI(), ""
        except Exception as e2:
            ai, ai_error = None, f"No Hailo ({e}) and no CPU AI ({e2})"
            print(ai_error)


def sync_cameras():
    """Restart only cameras whose source changed; others pick up new settings live."""
    wanted = {c["id"]: c for c in settings["cameras"] if c["enabled"]}
    for cid in list(cameras):
        if cid not in wanted or wanted[cid]["source"] != cameras[cid].cfg["source"]:
            cameras.pop(cid).running = False
    for cid, c in wanted.items():
        if cid in cameras:
            cameras[cid].cfg = c
        else:
            cameras[cid] = Camera(c)


# ---------------------------------------------------------------- web

app = Flask(__name__)
from training_service import register_training
training_service = register_training(app, globals())


@app.before_request
def require_login():
    a = request.authorization
    if not (a and hmac.compare_digest(a.username or "", settings["username"])
            and hmac.compare_digest(a.password or "", settings["password"])):
        return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="AI Camera"'})


@app.route("/")
def index():
    r = send_from_directory(BASE, "index.html")
    r.headers["Cache-Control"] = "no-cache"  # always revalidate so UI/JS updates load without a hard refresh
    return r


@app.route("/workspace/<name>")
def workspace_asset(name):
    if name not in {"workspace.js", "workspace.css"}:
        abort(404)
    r = send_from_directory(BASE, name)
    r.headers["Cache-Control"] = "no-cache"
    return r


@app.route("/video/<cid>")
def video(cid):
    cam = cameras.get(cid) or abort(404)

    def gen():
        sent = None
        while cam.running:
            j = cam.jpeg
            if j is not None and j is not sent:
                sent = j
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/frame/<cid>")
def frame(cid):
    """Newest live-view picture. The dashboard polls this instead of an endless MJPEG stream: browsers allow
    only ~6 open connections per server, and long streams from several tabs used them up (tiles went black)."""
    cam = cameras.get(cid) or abort(404)
    if cam.jpeg is None or not cam.online or not cam.last_frame or time.time() - cam.last_frame > 15:
        abort(404)
    return Response(cam.jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/collect/<cid>", methods=["GET", "POST", "DELETE"])
def collect(cid):
    """Save clean camera photos for training a custom detector (label them on Roboflow)."""
    cam = cameras.get(cid) or abort(404)
    folder = os.path.join(TRAIN_DIR, cid)
    if request.method == "DELETE":
        cam.collecting = None
        shutil.rmtree(folder, ignore_errors=True)
    elif request.method == "POST":
        if cam.collecting:
            return jsonify(error="Already collecting photos"), 409
        d = request.get_json(force=True) or {}
        want, every = num(d.get("count", 100), 1, 2000, int), num(d.get("every", 2), 0.5, 60)
        cam.collecting = [0, want]
        os.makedirs(folder, exist_ok=True)

        def run():
            prog = cam.collecting
            while cam.running and cam.collecting is prog and prog[0] < want:
                if cam.raw is not None:
                    cv2.imwrite(os.path.join(folder, f"{cid}_{int(time.time() * 1000)}.jpg"), cam.raw)
                    prog[0] += 1
                time.sleep(every)
            if cam.collecting is prog:
                cam.collecting = None

        threading.Thread(target=run, daemon=True).start()
    return jsonify(saved=len(glob.glob(os.path.join(folder, "*.jpg"))), progress=cam.collecting)


@app.route("/training/<cid>.zip")
def training_zip(cid):
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, os.path.basename(cid), "*.jpg")))
    if not files:
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # JPEGs are already compressed
        for p in files:
            z.write(p, os.path.basename(p))
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={cid}_training_photos.zip"})


@app.route("/snapshot/<cid>")
def snapshot(cid):
    cam = cameras.get(cid) or abort(404)
    if request.args.get("raw") and cam.raw is not None:  # clean frame for the zone editor
        return Response(cv2.imencode(".jpg", cam.raw)[1].tobytes(), mimetype="image/jpeg")
    if cam.jpeg is None:
        abort(404)
    return Response(cam.jpeg, mimetype="image/jpeg", headers={
        "Content-Disposition": f"attachment; filename={cid}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"})


@app.route("/api/record/<cid>", methods=["POST"])
def record(cid):
    cam = cameras.get(cid) or abort(404)
    if cam.jpeg is None:
        return jsonify(error="Camera has no picture yet"), 409
    if cam.clip:
        return jsonify(error="Already recording"), 409
    add_event(cam, "manual", "manual recording", cam.jpeg, force=True)
    return jsonify(ok=True)


@app.route("/api/pause/<cid>", methods=["POST"])
def pause(cid):
    """Pause / resume the AI on one camera; the video keeps streaming."""
    cam = cameras.get(cid) or abort(404)
    cam.paused = bool((request.get_json(force=True) or {}).get("paused"))
    if cam.paused:
        cam.counts, cam.zone_counts, cam.motion = {}, {}, False
    return jsonify(paused=cam.paused)


def active_model_name():
    """Friendly name of the active custom detector, or '' when it's the base/no custom model."""
    base = os.path.basename(str(settings.get("custom_model", "")).replace("\\", "/"))
    for m in training_service.state.get("models", {}).values():
        if os.path.basename(m["file"]) == base:
            return m["name"]
    return ""


@app.route("/api/hardware")
def api_hardware():
    return jsonify(hardware.info(peripherals))


@app.route("/api/hardware/output", methods=["POST"])
def api_hardware_output():
    d = request.get_json(force=True)
    peripherals.set_output(str(d.get("name", "")), bool(d.get("on")))  # no-op off-board
    return jsonify(hardware.info(peripherals))


@app.route("/api/status")
def status():
    with stats_lock:
        st = json.loads(json.dumps(stats))
    return jsonify(ai_ok=ai is not None, ai_backend=ai.label if ai else "", ai_error=ai_error, stats=st,
                   custom_model=active_model_name(), custom_classes=settings["custom_classes"], cameras=[
        {"id": c.id, "name": c.name, "online": c.online, "fps": round(c.fps, 1), "motion": c.motion,
         "group": c.cfg.get("group", ""), "last_frame": c.last_frame, "last_connected": c.last_connected,
         "reconnects": c.reconnects,
         "recording": c.clip is not None, "counts": c.counts, "zones": c.zone_counts,
         "features": c.cfg["features"], "paused": c.paused,
         "inspection": {**c.insp, "enabled": c.cfg["inspection"]["enabled"],
                        "trained": c.anomaly.info if c.anomaly else None}} for c in list(cameras.values())])


@app.route("/api/events", methods=["GET", "DELETE"])
def api_events():
    try:
        limit = num(request.args.get("limit", 200), 1, 500, int)
        after = float(request.args.get("after", 0))
        before = float(request.args.get("before", "inf"))
    except (ValueError, TypeError):
        return jsonify(error="Invalid event date or limit"), 400
    with ev_lock:
        if request.method == "DELETE":
            for p in glob.glob(os.path.join(EVENTS_DIR, "*.jpg")) + glob.glob(os.path.join(EVENTS_DIR, "*.mp4")):
                os.remove(p)
            events.clear()
            save_events()
        cam, kind = request.args.get("camera"), request.args.get("type")
        review, severity = request.args.get("review"), request.args.get("severity")
        out = [{**e, "severity": event_severity(e), "reviewed": e.get("reviewed", False)} for e in reversed(events)
               if (not cam or e["camera_id"] == cam) and (not kind or e["type"] == kind)
               and after <= e["time"] < before
               and (not review or bool(e.get("reviewed")) == (review == "reviewed"))
               and (not severity or event_severity(e) == severity)]
    return jsonify(out[:limit])


def event_severity(event):
    return "high" if event["type"] in {"fall", "intrusion"} else "medium" if event["type"] in {"anomaly", "loitering"} else "low"


@app.route("/api/events/<eid>/review", methods=["POST"])
def review_event(eid):
    data = request.get_json(silent=True) or {}
    if data.get("outcome", "confirmed") not in {"confirmed", "false_alarm", "unreviewed"}:
        return jsonify(error="Choose confirmed, false alarm or unreviewed"), 400
    with ev_lock:
        event = next((e for e in events if e["id"] == eid), None)
        if event is None:
            abort(404)
        event.update(reviewed=data.get("outcome") != "unreviewed", outcome=data.get("outcome", "confirmed"),
                     review_note=str(data.get("note", ""))[:1000], reviewed_at=time.time())
        save_events()
    return jsonify(ok=True)


@app.route("/events/<path:name>")
def event_file(name):
    return send_from_directory(EVENTS_DIR, name)


@app.route("/api/analytics")
def analytics():
    now = time.time()
    start = (now // 3600 - 23) * 3600
    hours = [{"time": start + i * 3600, "count": 0} for i in range(24)]
    with ev_lock:
        for e in events:
            i = int((e["time"] - start) // 3600)
            if 0 <= i < 24 and e["type"] not in QUIET:
                hours[i]["count"] += 1
    return jsonify(hours=hours)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global settings
    if request.method == "POST":
        try:
            new = clean_settings(request.get_json(force=True))
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
            return jsonify(error=f"Invalid settings: {e}"), 400
        settings = new
        save_settings(settings)
        if ai is None:
            start_ai()
        sync_cameras()
    safe = {k: v for k, v in settings.items() if k not in SECRETS}
    return jsonify({**safe, "telegram_set": bool(settings["telegram_token"]),
                    "classes": COCO + [c for c in settings["custom_classes"] if c not in COCO],
                    "custom_ok": custom_det is not None and custom_det.key[0] == settings["custom_model"],
                    "custom_failed": bool(custom_error)})


@app.route("/api/inspect/<cid>", methods=["POST", "DELETE"])
def api_inspect(cid):
    """POST: start learning good images from the live camera. DELETE: forget the learned model."""
    cam = cameras.get(cid) or abort(404)
    if request.method == "DELETE":
        cam.anomaly, cam.learning = None, None
        cam.insp.update(state="not trained", score=None, severity=None)
        try:
            os.remove(os.path.join(INSPECT_DIR, cid + ".pt"))
        except OSError:
            pass
        return jsonify(ok=True)
    if not cam.cfg["inspection"]["enabled"]:
        return jsonify(error="Turn on inspection for this camera and save first"), 409
    cam.learning = []
    return jsonify(ok=True)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Find devices with an open RTSP port (554) on this computer's /24 network."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
        u.connect(("10.255.255.255", 1))  # sends nothing; just picks the LAN interface
        prefix = u.getsockname()[0].rsplit(".", 1)[0]

    def probe(ip):
        try:
            with socket.create_connection((ip, 554), timeout=0.6):
                return ip
        except OSError:
            return None

    with ThreadPoolExecutor(64) as ex:
        found = [ip for ip in ex.map(probe, [f"{prefix}.{i}" for i in range(1, 255)]) if ip]
    return jsonify(network=prefix + ".0/24", devices=found)


@app.route("/api/test", methods=["POST"])
def api_test():
    src = str((request.get_json(force=True) or {}).get("source", "")).strip()
    if not src:
        return jsonify(ok=False, error="Source is empty")
    # Probe in a child process: a misbehaving camera driver can hang or crash, and must not take the server down.
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe", src],
                             capture_output=True, text=True, timeout=25).stdout.split()
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error="No picture within 25 seconds")
    if out[-3:-2] != ["OK"]:
        return jsonify(ok=False, error="Could not read a frame from this source")
    return jsonify(ok=True, width=int(out[-2]), height=int(out[-1]))


@app.route("/api/test_telegram", methods=["POST"])
def api_test_telegram():
    if not (settings["telegram_token"] and settings["telegram_chat_id"]):
        return jsonify(ok=False, error="Save a bot token and chat ID first")
    try:
        r = requests.post(f"https://api.telegram.org/bot{settings['telegram_token']}/sendMessage",
                          data={"chat_id": settings["telegram_chat_id"], "text": "AI Camera test message ✅"}, timeout=15)
        return jsonify(ok=r.ok, error=None if r.ok else r.json().get("description", r.text))
    except requests.RequestException as e:
        return jsonify(ok=False, error=str(e))


def probe(src):
    cap = open_capture(src)
    ok, f = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    print(f"OK {f.shape[1]} {f.shape[0]}" if ok else "FAIL")


if __name__ == "__main__" and sys.argv[1:2] == ["--probe"]:
    probe(sys.argv[2])
elif __name__ == "__main__":
    start_ai()
    peripherals = hardware.Peripherals(ai is not None and getattr(ai, "label", "") == "Hailo")
    peripherals.on_input = sensor_event  # digital inputs (DI0/DI1) raise events like a camera detection
    atexit.register(peripherals.cleanup)
    threading.Thread(target=hardware.status_loop, args=(peripherals, cameras), daemon=True).start()
    sync_cameras()
    threading.Thread(target=stats_saver, daemon=True).start()
    app.run(host="0.0.0.0", port=8080, threaded=True)
