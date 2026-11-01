"""Make synthetic egg-line videos for trying Quality inspection without a real line:
  demo-videos/egg_line_good.mp4  - only good eggs (use it to "Learn good")
  demo-videos/egg_line.mp4       - about 1 in 30 eggs has a defect (spots, crack, dirt, calcium patch)
Run: python make_demo_eggs.py
"""
import os
import cv2
import numpy as np

W, H, FPS, SPEED, GAP = 1280, 720, 25, 2, 150
LANES = [250, 370, 490, 610]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-videos")


def egg(seed, defect):
    rng = np.random.default_rng(seed)
    w, h = 130, 104
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = rng.uniform(48, 54), rng.uniform(38, 42)
    alpha = np.zeros((h, w), np.float32)
    cv2.ellipse(alpha, (w // 2, h // 2), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    shade = 1.2 - 0.6 * np.sqrt(((xx - w * 0.4) / rx) ** 2 + ((yy - h * 0.36) / ry) ** 2)  # light from top-left
    img = np.array([62, 118, 188], np.float32) * rng.uniform(0.9, 1.08) * np.clip(shade, 0.5, 1.3)[..., None]
    img += rng.normal(0, 3, (h, w, 1))  # shell texture
    cx, cy = w // 2, h // 2
    if defect == "spots":
        for _ in range(rng.integers(4, 9)):
            cv2.circle(img, (int(cx + rng.uniform(-30, 30)), int(cy + rng.uniform(-22, 22))),
                       int(rng.integers(2, 6)), (28, 40, 62), -1)
    elif defect == "crack":
        p = np.array([cx + rng.uniform(-30, -10), cy + rng.uniform(-20, 0)])
        for _ in range(6):
            q = p + rng.uniform([6, -8], [14, 10])
            cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), (30, 45, 70), 2, cv2.LINE_AA)
            p = q
    elif defect == "dirt":
        blob = np.zeros((h, w), np.float32)
        cv2.ellipse(blob, (int(cx + rng.uniform(-15, 15)), int(cy + rng.uniform(-10, 10))), (16, 9),
                    rng.uniform(0, 180), 0, 360, 1, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 3)[..., None]
        img = img * (1 - 0.8 * blob) + np.array([55, 75, 70], np.float32) * 0.8 * blob
    elif defect == "calcium":
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (int(cx + rng.uniform(-15, 15)), int(cy + rng.uniform(-10, 10))), 11, 1, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 3)[..., None]
        img = img * (1 - 0.85 * blob) + np.array([205, 215, 225], np.float32) * 0.85 * blob
    return np.clip(img, 0, 255), cv2.GaussianBlur(alpha, (5, 5), 0)[..., None]


def background(t):
    f = np.full((H, W, 3), (78, 58, 44), np.uint8)  # blue-grey belt
    for x in range(-(t * SPEED) % 40, W, 40):  # moving slats
        cv2.line(f, (x, 190), (x, 680), (68, 50, 38), 2)
    for y in [190] + [(a + b) // 2 for a, b in zip(LANES, LANES[1:])] + [680]:  # lane rails
        cv2.rectangle(f, (0, y - 5), (W, y + 5), (170, 170, 165), -1)
    cv2.rectangle(f, (0, 0), (W, 184), (40, 40, 44), -1)
    return f


def render(path, seconds, defect_rate):
    os.makedirs(OUT, exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cache, kinds = {}, ["spots", "crack", "dirt", "calcium"]
    for t in range(seconds * FPS):
        f = background(t).astype(np.float32)
        for li, y in enumerate(LANES):
            shift = t * SPEED + li * 37
            for j in range(shift // GAP - 1, (shift + W) // GAP + 2):
                x = j * GAP - shift
                key = (li, j)
                if key not in cache:
                    r = np.random.default_rng(hash(key) & 0xFFFFFFFF)
                    bad = r.random() < defect_rate
                    cache[key] = egg(hash(key) & 0xFFFF, kinds[r.integers(4)] if bad else None)
                img, a = cache[key]
                h, w = img.shape[:2]
                x0, y0 = x - w // 2, y - h // 2
                sx0, sx1 = max(0, -x0), min(w, W - x0)
                if sx1 <= sx0:
                    continue
                region = f[y0:y0 + h, x0 + sx0:x0 + sx1]
                sh = a[:, sx0:sx1] * 0.35  # soft shadow
                region[6:] *= 1 - sh[:-6]
                region[:] = region * (1 - a[:, sx0:sx1]) + img[:, sx0:sx1] * a[:, sx0:sx1]
        out.write(f.astype(np.uint8))
    out.release()


if __name__ == "__main__":
    render(os.path.join(OUT, "egg_line_good.mp4"), 20, 0.0)
    render(os.path.join(OUT, "egg_line.mp4"), 40, 0.03)
    print("wrote", OUT)
