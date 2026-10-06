"""Rebuilds docs/before-after.gif: a synthetic shaky clip, stabilized, side by side.

    python docs/make_demo.py

Needs ffmpeg on PATH. The scene is generated here, so the demo contains no third-party footage.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "skills" / "point-stabilizer" / "scripts" / "stabilize.py"
OUT = ROOT / "docs" / "before-after.gif"
W, H, FPS, SECONDS = 640, 360, 30, 6
OX, OY = 200, 150


def make_world():
    rng = np.random.default_rng(3)
    canvas = np.full((900, 1400, 3), 236, np.uint8)
    canvas = cv2.GaussianBlur(rng.integers(205, 245, (900, 1400, 3), dtype=np.uint8), (0, 0), 3.0)
    palette = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (156, 39, 176), (0, 172, 193)]
    for _ in range(70):
        colour = palette[int(rng.integers(0, len(palette)))]
        x, y = int(rng.integers(0, 1400)), int(rng.integers(0, 900))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + int(rng.integers(30, 120)), y + int(rng.integers(30, 90))), colour, -1)
        else:
            cv2.circle(canvas, (x, y), int(rng.integers(15, 50)), colour, -1)
    cv2.rectangle(canvas, (OX + 60, OY + 40), (OX + 580, OY + 320), (255, 255, 255), -1)          # a "card"
    cv2.rectangle(canvas, (OX + 60, OY + 40), (OX + 580, OY + 320), (60, 60, 60), 3)
    cv2.putText(canvas, "POINT", (OX + 100, OY + 150), cv2.FONT_HERSHEY_DUPLEX, 2.6, (40, 40, 40), 5, cv2.LINE_AA)
    cv2.putText(canvas, "STABILIZER", (OX + 100, OY + 240), cv2.FONT_HERSHEY_DUPLEX, 2.0, (40, 40, 40), 4, cv2.LINE_AA)
    return canvas


def shake(n):
    rng = np.random.default_rng(11)
    t = np.arange(n) / FPS
    jitter = np.zeros((n, 2))
    for i in range(1, n):
        jitter[i] = 0.9 * jitter[i - 1] + rng.normal(0, 1.0, 2)
    tx = 11 * np.sin(2 * np.pi * 0.6 * t) + 6 * np.sin(2 * np.pi * 1.9 * t + 1) + jitter[:, 0]
    ty = 8 * np.sin(2 * np.pi * 0.45 * t + 2) + 4 * np.sin(2 * np.pi * 2.3 * t) + jitter[:, 1]
    return tx, ty


def write_video(path, frames):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{frames[0].shape[1]}x{frames[0].shape[0]}", "-framerate", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def label(frame, text, point):
    f = cv2.resize(frame, (W // 2 + 40, (H // 2 + 20)), interpolation=cv2.INTER_AREA)
    cx, cy = int(point[0] * f.shape[1] / W), int(point[1] * f.shape[0] / H)
    cv2.rectangle(f, (0, 0), (92, 22), (0, 0, 0), -1)
    cv2.putText(f, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for colour, thick in (((0, 0, 0), 3), ((0, 140, 255), 1)):          # the same screen position in both halves
        cv2.circle(f, (cx, cy), 8, colour, thick, cv2.LINE_AA)
        cv2.line(f, (cx - 15, cy), (cx + 15, cy), colour, thick, cv2.LINE_AA)
        cv2.line(f, (cx, cy - 15), (cx, cy + 15), colour, thick, cv2.LINE_AA)
    return f


def main():
    n = FPS * SECONDS
    world = make_world()
    tx, ty = shake(n)
    shaky = [cv2.warpAffine(world, np.float32([[1, 0, tx[i] - OX], [0, 1, ty[i] - OY]]), (W, H),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        write_video(tmp / "shaky.mp4", shaky)
        run = subprocess.run([sys.executable, str(ENGINE), str(tmp / "shaky.mp4"), "--point", "auto",
                              "-o", str(tmp / "stable.mp4")], capture_output=True, text=True, check=True)
        point = [float(v) for v in re.search(r"Locked point: \(([\d.]+), ([\d.]+)\)", run.stdout).groups()]
        stable = read_video(tmp / "stable.mp4")
        gap = np.full((H // 2 + 20, 6, 3), 255, np.uint8)
        side = [np.hstack([label(a, "BEFORE", point), gap, label(b, "AFTER", point)]) for a, b in zip(shaky, stable)]
        write_video(tmp / "side.mp4", side)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "side.mp4"), "-vf",
                        "fps=12,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(OUT)], check=True)
    print(f"locked point {point}; wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
