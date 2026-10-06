#!/usr/bin/env python3
"""Point stabilizer: lock a video onto a point you click on.

    python stabilize.py video.mp4

1. The first frame opens in a window - click the point to lock onto,
   press Enter/Space to confirm (Esc/q cancels).
2. The point is tracked through every frame with pyramidal Lucas-Kanade
   (cv2.calcOpticalFlowPyrLK); each frame is shifted so the point stays at
   the position you clicked.
3. If the track is lost, it is re-seeded near the last known position.
4. The result is encoded to H.264 mp4 and the original audio is muxed in
   with ffmpeg.

Dependencies: opencv-python, numpy (+ the ffmpeg binary).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

WINDOW = "Click the point to stabilize"

# Lucas-Kanade settings: a big window and 5 pyramid levels survive fast shake.
LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FB_MAX_ERR = 1.5        # px: forward-backward check, above this the track is considered lost
SNAP_RADIUS = 10        # px: the click is snapped to a corner at most this far away
TPL_HALF = 15           # appearance template is (2*15+1)^2 px around the point
MATCH_MIN = 0.8         # min normalized correlation to accept a re-found point
SEARCH_RADIUS = 60      # px: where to look for the lost point around its last position
RESEED_AFTER = 12       # lost frames after which brand-new corners are seeded nearby
REINIT_RADIUS = 40      # px: corner search radius when re-seeding, grows on repeated failures
REINIT_GROWTH = 1.5

# Audio codecs that can be copied into an mp4 as-is (anything else becomes AAC).
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}

BORDER_MODES = {
    "black": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
}


# --------------------------------------------------------------------- utils

def log(msg=""):
    print(msg, flush=True)


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def find_ffmpeg():
    """Bundled copy (PyInstaller / next to the script) first, then PATH."""
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    folders = []
    if getattr(sys, "frozen", False):
        folders.append(getattr(sys, "_MEIPASS", ""))
        folders.append(os.path.dirname(sys.executable))
    folders.append(os.path.dirname(os.path.abspath(__file__)))
    for folder in folders:
        candidate = os.path.join(folder, name)
        if folder and os.path.isfile(candidate):
            return candidate
    return shutil.which("ffmpeg")


def audio_codec(ffmpeg, path):
    """Codec of the first audio stream of `path`, or None if it has no audio."""
    info = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                          capture_output=True).stderr.decode(errors="replace")
    match = re.search(r"Stream #\S+.*?: Audio: (\w+)", info)
    return match.group(1) if match else None


class Progress:
    """Single-line console progress bar (no dependencies)."""

    def __init__(self, total, width=30):
        self.total = total if total and total > 0 else 0
        self.width = width
        self.start = time.time()
        self.last_draw = 0.0

    def update(self, done, note="", force=False):
        now = time.time()
        if not force and now - self.last_draw < 0.1:
            return
        self.last_draw = now
        elapsed = max(now - self.start, 1e-6)
        speed = done / elapsed
        if self.total:
            frac = min(done / self.total, 1.0)
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            eta = (self.total - done) / speed if speed > 0 else 0
            line = (f"\r[{bar}] {frac * 100:5.1f}%  {done}/{self.total}  "
                    f"{speed:5.1f} fps  ETA {fmt_time(eta)}")
        else:
            line = f"\r{done} frames  {speed:5.1f} fps  {fmt_time(elapsed)}"
        sys.stdout.write(f"{line}  {note:<22}")
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# ------------------------------------------------------------- point picking

SUGGEST_FRAMES = 40         # frames sampled from the start of the video to find still details
SUGGEST_STRIDE = 2
SUGGEST_MAX_SIDE = 960      # analysis resolution, keeps long 4K clips cheap
GOOD_QUALITY = 0.02         # corner strength relative to the strongest corner in the frame


def sample_frames(src):
    """First SUGGEST_FRAMES sampled frames as gray images, downscaled. Returns (frames, scale)."""
    cap = cv2.VideoCapture(src)
    frames, i, scale = [], 0, 1.0
    while len(frames) < SUGGEST_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if i % SUGGEST_STRIDE == 0:
            if not frames:
                scale = min(1.0, SUGGEST_MAX_SIDE / max(frame.shape[:2]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray if scale == 1.0 else cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    return frames, scale


def suggest_points(frames, scale, max_points=4):
    """Details that are still in the world: corners that move exactly like the background.

    The camera moves everything together, so the motion shared by most corners
    (shift, rotation, zoom) is the camera. Corners that follow it all the time
    are still objects; people, hands or anything else that moves on its own
    deviate and are dropped. Returns an (N, 2) array in full-resolution pixels,
    best first.
    """
    empty = np.zeros((0, 2), np.float32)
    if len(frames) < 4:
        return empty
    h, w = frames[0].shape[:2]
    corners = cv2.goodFeaturesToTrack(
        frames[0], maxCorners=400, qualityLevel=0.02, minDistance=10, blockSize=7)
    if corners is None:
        return empty
    pts = corners.reshape(-1, 2)                       # strongest corner first
    margin_x, margin_y = 0.06 * w, 0.06 * h
    inside = ((pts[:, 0] > margin_x) & (pts[:, 0] < w - margin_x) &
              (pts[:, 1] > margin_y) & (pts[:, 1] < h - margin_y))
    pts = pts[inside]
    if len(pts) < 8:
        return empty

    track, alive, cur = [pts], np.ones(len(pts), bool), pts.copy()
    for prev, nxt in zip(frames, frames[1:]):
        cur, ok = _lk_checked(prev, nxt, cur)
        alive &= ok
        track.append(cur.copy())
    track = np.stack(track)[:, alive]                  # (frames, N, 2)
    pts = pts[alive]
    if len(pts) < 8:
        return empty

    # Fit the camera (shift + rotation + zoom) to all corners on every frame, robust
    # to outliers; a corner that stays on that model all the time is a still detail.
    first = np.ascontiguousarray(track[0], np.float32)
    worst = np.zeros(len(pts))
    for k in range(1, len(track)):
        model, _ = cv2.estimateAffinePartial2D(
            first.reshape(-1, 1, 2), np.ascontiguousarray(track[k], np.float32).reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if model is None:
            return empty
        predicted = first @ model[:, :2].T + model[:, 2]
        worst = np.maximum(worst, np.linalg.norm(predicted - track[k], axis=1))

    consistent = list(np.flatnonzero(worst <= 1.5))    # follow the camera within 1.5 px
    if not consistent and worst.min() <= 4.0:
        consistent = [int(np.argmin(worst))]           # nothing perfect: take the best one
    # `pts` is ordered by corner strength, so the smallest index is the sharpest detail
    chosen = sorted(consistent)
    picked = []
    min_gap = 0.15 * min(w, h)
    for i in chosen:
        if all(np.linalg.norm(pts[i] - pts[j]) > min_gap for j in picked):
            picked.append(i)
        if len(picked) == max_points:
            break
    return (pts[picked] / scale).astype(np.float32)


def fit_text(canvas, text, y, color=(255, 255, 255), size=0.55, x=8):
    """putText that shrinks to fit the width of the canvas."""
    while size > 0.3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] > canvas.shape[1] - 2 * x:
        size -= 0.03
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def pick_point(frame, suggestions=None):
    """Show `frame`, let the user click a point or accept a suggested one.
    Returns (x, y) in frame pixels or None."""
    suggestions = np.zeros((0, 2), np.float32) if suggestions is None else suggestions
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / w, 720 / h)
    view = frame if scale == 1.0 else cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    eig = cv2.cornerMinEigenVal(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), 7)
    peak = float(eig.max()) or 1.0

    def quality(pt):
        """Strength of the best corner within 8 px of the clicked place (0..1)."""
        x, y = int(pt[0] * scale), int(pt[1] * scale)
        return float(eig[max(0, y - 8):y + 9, max(0, x - 8):x + 9].max()) / peak

    state = {"pt": None, "q": 0.0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x / scale, y / scale)
            for s in suggestions:                  # a click on a green marker takes exactly that one
                if np.hypot(s[0] * scale - x, s[1] * scale - y) <= 18:
                    pt = (float(s[0]), float(s[1]))
                    break
            state["pt"] = pt
            state["q"] = quality(pt)

    def cross(canvas, pt, color, radius=10, arm=16):
        cx, cy = int(round(pt[0] * scale)), int(round(pt[1] * scale))
        for col, thick in (((0, 0, 0), 3), (color, 1)):
            cv2.circle(canvas, (cx, cy), radius, col, thick, cv2.LINE_AA)
            cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), col, thick, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), col, thick, cv2.LINE_AA)
        return cx, cy

    def label(canvas, cx, cy, text, color):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = int(min(max(4, cx - size[0] // 2), canvas.shape[1] - size[0] - 4))
        y = cy - 22 if cy > 60 else cy + 36
        cv2.rectangle(canvas, (x - 3, y - size[1] - 4), (x + size[0] + 3, y + 5), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    GREEN, ORANGE, YELLOW = (80, 230, 80), (0, 165, 255), (0, 255, 255)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    seen_visible = False
    result = None
    while True:
        canvas = view.copy()
        if state["pt"] is None and len(suggestions):
            hint = "Click a still detail, or press Enter to use the green one | Esc = cancel"
        elif state["pt"] is None:
            hint = "Click the point | Enter = OK | Esc = cancel"
        else:
            hint = "Enter = OK | click again to change | Esc = cancel"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        fit_text(canvas, hint, 20)

        for k, s in enumerate(suggestions):        # suggestions stay visible so you can switch
            if k == 0 and state["pt"] is None:
                cx, cy = cross(canvas, s, GREEN)
                label(canvas, cx, cy, "suggested: still detail", GREEN)
            else:
                cross(canvas, s, GREEN, radius=6, arm=0)
        if state["pt"] is not None:
            good = state["q"] >= GOOD_QUALITY
            cx, cy = cross(canvas, state["pt"], YELLOW if good else ORANGE)
            label(canvas, cx, cy, "good point" if good else "weak point: pick a sharper detail",
                  GREEN if good else ORANGE)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10, 32):
            if state["pt"] is not None:
                result = state["pt"]
                break
            if len(suggestions):
                result = (float(suggestions[0][0]), float(suggestions[0][1]))
                break
        if key in (27, ord("q")):
            break
        # Window closed with the mouse. Some backends report -1 / 0 before the
        # first paint, so only trust it after we have seen the window visible.
        visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        if visible >= 1:
            seen_visible = True
        elif seen_visible:
            break

    cv2.destroyAllWindows()
    for _ in range(4):          # let the GUI backend actually close the window (macOS)
        cv2.waitKey(1)
    return result


# ------------------------------------------------------------------ tracking

def _lk_checked(prev, cur, pts):
    """LK prev -> cur with a backward check. pts: (N,2). Returns (new_pts (N,2), ok mask (N,))."""
    h, w = cur.shape[:2]
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **LK_PARAMS)
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK_PARAMS)
    fb_err = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
    p1 = p1.reshape(-1, 2)
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < FB_MAX_ERR) & inside
    return p1, ok


class PointTracker:
    """Tracks one point with LK optical flow and survives losing it.

    Two positions are kept:
      feat - the image feature LK really follows;
      pos  - the "virtual" point we report = feat + bias (starts exactly at the click).

    When `feat` is lost (blur, flash, something passing over it) the position
    is held and the point is searched for again near its last known position:
      1. by appearance - template matching against the patch from the first
         frame and from the last good frame, refined to sub-pixel with LK;
      2. if it stays lost for RESEED_AFTER frames - by seeding brand-new
         corners nearby; their median motion carries `pos` forward and `bias`
         is recomputed, so the stabilization continues without a jump.
    """

    def __init__(self, gray, click):
        self.prev = gray            # last frame on which the point was known
        self.pos = np.array(click, np.float32)
        self.feat = self.pos.copy()
        self.bias = np.zeros(2, np.float32)
        self.fails = 0              # consecutive frames without a good track
        self.lost_events = 0        # how many times the track was lost
        self.recovered = 0          # ... and found again by appearance
        self.reseeds = 0            # ... and replaced by new corners
        self.held_frames = 0        # frames where the position was simply held

        # Snap the click to the closest strong corner, but keep `pos` at the click.
        corners = self._corners(gray, self.feat, SNAP_RADIUS)
        if len(corners):
            self.feat = corners[0]
            self.bias = self.pos - self.feat
        self.first, self.first_feat = gray, self.feat.copy()   # first frame, never changes
        self.first_patch = self._patch(gray, self.feat)

    @staticmethod
    def _corners(gray, center, radius):
        """Corners within `radius` of `center`, closest first. Returns (N,2) float32."""
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), int(radius), 255, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=60, qualityLevel=0.01, minDistance=3, blockSize=7, mask=mask)
        if corners is None:
            return np.zeros((0, 2), np.float32)
        corners = corners.reshape(-1, 2).astype(np.float32)
        order = np.argsort(np.linalg.norm(corners - np.asarray(center, np.float32), axis=1))
        return corners[order]

    def _reseed(self, gray):
        """Seed corners near the last known position and carry the point over to `gray`.
        Returns True on success."""
        radius = min(REINIT_RADIUS * REINIT_GROWTH ** max(0, self.fails - RESEED_AFTER),
                     0.25 * max(gray.shape[:2]))
        h, w = gray.shape[:2]
        center = np.clip(self.feat, 0, [w - 1, h - 1])
        seeds = self._corners(self.prev, center, radius)
        if len(seeds) == 0:
            return False
        moved, ok = _lk_checked(self.prev, gray, seeds)
        if not ok.any():
            return False
        seeds, moved = seeds[ok], moved[ok]
        disp = moved - seeds
        median = np.median(disp, axis=0)
        inliers = np.linalg.norm(disp - median, axis=1) < 2.0    # drop corners on moving junk
        if not inliers.any():
            return False
        seeds, moved, disp = seeds[inliers], moved[inliers], disp[inliers]
        # seeds are sorted closest-first, so index 0 is the closest consistent corner
        self.feat = moved[0]
        self.pos = self.pos + np.median(disp, axis=0)
        self.bias = self.pos - self.feat
        # We follow a different image point now: its appearance is the new reference.
        self.first, self.first_feat = gray, self.feat.copy()
        self.first_patch = self._patch(gray, self.feat)
        return True

    @staticmethod
    def _patch(img, pt):
        """Square patch around `pt`, or None if it is not fully inside / has no texture."""
        x, y = int(round(pt[0])), int(round(pt[1]))
        h, w = img.shape[:2]
        if x - TPL_HALF < 0 or y - TPL_HALF < 0 or x + TPL_HALF >= w or y + TPL_HALF >= h:
            return None
        patch = img[y - TPL_HALF:y + TPL_HALF + 1, x - TPL_HALF:x + TPL_HALF + 1]
        return patch if patch.std() > 4 else None

    def _reacquire(self, gray):
        """Look for the same image point by appearance near its last position."""
        h, w = gray.shape[:2]
        radius = int(min(SEARCH_RADIUS, 0.3 * max(h, w)))
        fx, fy = int(self.feat[0]), int(self.feat[1])
        x0, x1 = max(0, fx - radius - TPL_HALF), min(w, fx + radius + TPL_HALF + 1)
        y0, y1 = max(0, fy - radius - TPL_HALF), min(h, fy + radius + TPL_HALF + 1)
        region = gray[y0:y1, x0:x1]

        # First-frame template first: it never drifts. Then the last good frame's.
        sources = ((self.first_patch, self.first, self.first_feat),
                   (self._patch(self.prev, self.feat), self.prev, self.feat))
        for tpl, ref, ref_pt in sources:
            if tpl is None or region.shape[0] < tpl.shape[0] or region.shape[1] < tpl.shape[1]:
                continue
            res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < MATCH_MIN:
                continue
            guess = np.array([x0 + loc[0] + TPL_HALF, y0 + loc[1] + TPL_HALF], np.float32)
            found = self._verify(ref, gray, ref_pt, guess, score)
            if found is not None:
                self.feat = found
                self.pos = self.feat + self.bias    # same image point, so bias is unchanged
                return True
        return False

    @staticmethod
    def _verify(ref, gray, ref_pt, guess, score):
        """Refine a template match to sub-pixel with LK and make sure it is geometrically
        consistent (forward-backward). Look-alikes such as a similar edge fail this."""
        p0 = ref_pt.reshape(1, 1, 2).astype(np.float32)
        init = guess.reshape(1, 1, 2).copy()
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            ref, gray, p0, init, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK_PARAMS)
        if st[0, 0] == 1 and np.linalg.norm(p1 - init) < 3:
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(
                gray, ref, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK_PARAMS)
            if st_b[0, 0] == 1 and np.linalg.norm(back - p0) < FB_MAX_ERR:
                return p1.reshape(2)
        return guess if score >= 0.95 else None     # a near-perfect match needs no proof

    def update(self, gray):
        """Feed the next frame. Returns (virtual position, status)."""
        moved, ok = _lk_checked(self.prev, gray, self.feat.reshape(1, 2))
        if ok[0]:
            self.feat = moved[0]
            self.pos = self.feat + self.bias
            self.prev = gray
            self.fails = 0
            return self.pos, "track"

        if self.fails == 0:
            self.lost_events += 1
        self.fails += 1
        if self._reacquire(gray):
            self.prev = gray
            self.recovered += 1
            self.fails = 0
            return self.pos, "RECOVERED"
        if self.fails >= RESEED_AFTER and self._reseed(gray):
            self.prev = gray
            self.reseeds += 1
            self.fails = 0
            return self.pos, "RESEEDED"

        # Nothing usable on this frame: hold the position, keep the last good
        # frame as the reference and try again on the next one.
        self.held_frames += 1
        return self.pos, "HELD"


# ------------------------------------------------------------------- output

class FFmpegSink:
    """Pipes BGR frames into ffmpeg (libx264), then muxes the original audio."""

    def __init__(self, ffmpeg, src, dst, w, h, fps, crf):
        self.ffmpeg, self.src, self.dst, self.fps = ffmpeg, src, dst, fps
        self.frames = 0
        self.tmp = os.path.splitext(dst)[0] + ".video_only.tmp.mp4"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-framerate", f"{fps:.6f}", "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.tmp,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.frames += 1
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError("ffmpeg stopped unexpectedly:\n" + self._stderr())

    def _stderr(self):
        try:
            return self.proc.stderr.read().decode(errors="replace")
        except Exception:
            return ""

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode video:\n" + self._stderr())
        self._mux_audio()

    def abort(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _run_mux(self, audio_args):
        cmd = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", self.tmp, "-i", self.src,
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", *audio_args,
            # The video is the master: never cut it because the audio is a hair shorter,
            # but trim audio that runs longer than the picture.
            "-t", f"{self.frames / self.fps:.6f}", "-movflags", "+faststart", self.dst,
        ]
        return subprocess.run(cmd, capture_output=True)

    def _mux_audio(self):
        # Keep the original audio untouched when mp4 can hold it; otherwise AAC.
        codec = audio_codec(self.ffmpeg, self.src)
        aac = ["-c:a", "aac", "-b:a", "192k"]
        if codec in MP4_AUDIO:
            res = self._run_mux(["-c:a", "copy"])
            if res.returncode != 0:
                log("  audio could not be copied as-is, re-encoding to AAC")
                res = self._run_mux(aac)
        else:
            if codec:
                log(f"  original audio is {codec}, converting to AAC for mp4")
            res = self._run_mux(aac)
        if res.returncode != 0:
            raise RuntimeError("ffmpeg failed to add audio:\n" + res.stderr.decode(errors="replace"))
        os.remove(self.tmp)


class CvSink:
    """Fallback when ffmpeg is missing: mp4v through OpenCV, no audio."""

    def __init__(self, dst, w, h, fps):
        self.writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError("OpenCV could not open the output file for writing")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def abort(self):
        self.writer.release()


# --------------------------------------------------------------------- main

def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Stabilize a video around a point you click on the first frame.")
    ap.add_argument("video", nargs="?", help="input video")
    ap.add_argument("-o", "--output", help="output mp4 (default: <name>_stabilized.mp4)")
    ap.add_argument("--point", metavar="X,Y|auto",
                    help="skip the window: a point (pixels in the first frame), or 'auto' to let "
                         "the program pick a still detail by itself")
    ap.add_argument("--border", choices=BORDER_MODES, default="black",
                    help="how to fill the edges revealed by the shift (default: black)")
    ap.add_argument("--zoom", default="auto", metavar="auto|N",
                    help="'auto' (default) zooms in just enough to hide every black edge; "
                         "a number such as 1.15 sets it by hand; 1 turns zoom off")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 quality, lower = better/bigger (default: 18)")
    ap.add_argument("--preview", action="store_true",
                    help="show the stabilized frames and a progress bar in a window "
                         "(always on when started without a video argument)")
    return ap.parse_args(argv)


class Cancelled(Exception):
    pass


class Preview:
    """Progress for the double-click app (no console): the stabilized frames in a
    window with a bar under them. Esc cancels."""

    WIN = "Point Stabilizer - Esc to cancel"

    def __init__(self, w, h):
        self.scale = min(1.0, 960 / w, 600 / h)
        self.start = time.time()
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)

    def _draw(self, frame, frac, text):
        view = frame if self.scale == 1.0 else cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        view = view.copy()
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, h - 46), (w, h), (0, 0, 0), -1)
        cv2.putText(view, text, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(view, (10, h - 18), (w - 10, h - 8), (90, 90, 90), 1)
        cv2.rectangle(view, (10, h - 18), (10 + int((w - 20) * frac), h - 8), (255, 255, 255), -1)
        cv2.imshow(self.WIN, view)

    def show(self, frame, done, total, note=""):
        """Draw one frame + progress. Returns False if the user pressed Esc."""
        frac = min(done / total, 1.0) if total else 0.0
        elapsed = max(time.time() - self.start, 1e-6)
        eta = (total - done) * elapsed / done if total and done else 0
        text = f"{frac * 100:.0f}%   {done}/{total}   ETA {fmt_time(eta)}   {note}"
        self._draw(frame, frac, text)
        return (cv2.waitKey(1) & 0xFF) != 27

    def finish(self, frame, text, wait_ms):
        self._draw(frame, 1.0, text)
        cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)


def alert(msg):
    """Show an error where the user can see it when there is no console."""
    log(msg)
    try:
        if sys.platform == "darwin":
            text = msg.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(["osascript", "-e",
                            f'display dialog "{text}" with title "Point Stabilizer" '
                            f'buttons {{"OK"}} default button "OK" with icon caution'],
                           capture_output=True)
        elif os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "Point Stabilizer", 0x10)
    except Exception:
        pass


def reveal(path):
    """Show the result in Finder / Explorer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", path])
        elif os.name == "nt":
            subprocess.run(["explorer", f"/select,{os.path.normpath(path)}"])
    except Exception:
        pass


def ask_video_file():
    """No argument given (e.g. double-clicked app): ask for a file with a system dialog."""
    if sys.platform == "darwin":
        res = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose file with prompt "Choose a video to stabilize" '
             'of type {"public.movie"} default location (path to downloads folder))'],
            capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.strip() or None
        if "-128" in res.stderr:            # user pressed Cancel
            return None
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"), ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


def unique_path(path):
    """name.mp4 -> name 2.mp4 -> name 3.mp4 ... so a double-click never overwrites a result."""
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(path):
        path = f"{base} {n}{ext}"
        n += 1
    return path


AUTO_ZOOM_MAX = 1.6         # never crop in more than this, however violent the shake
AUTO_ZOOM_PERCENTILE = 99   # a few extreme frames may show an edge rather than zoom everything


def auto_zoom_factor(positions, anchor, w, h):
    """Smallest constant zoom (about the anchor) that keeps the picture covering the frame.

    A frame is drawn as dst = z * (src - pos) + anchor, so the source pixel for the
    left edge is pos_x - ax / z and for the right edge pos_x + (w - ax) / z. Both
    must stay inside [0, w], which gives z >= ax / pos_x and z >= (w - ax) / (w - pos_x),
    and the same for y. Returns (zoom, capped).
    """
    ax, ay = float(anchor[0]), float(anchor[1])
    px, py = positions[:, 0].astype(np.float64), positions[:, 1].astype(np.float64)

    def need(num, den):
        out = np.full(den.shape, np.inf)
        np.divide(num, den, out=out, where=den > 0)
        return out

    req = np.maximum.reduce([need(ax, px), need(w - ax, w - px),
                             need(ay, py), need(h - ay, h - py), np.ones_like(px)])
    z = float(np.percentile(np.minimum(req, AUTO_ZOOM_MAX), AUTO_ZOOM_PERCENTILE))
    z = min(z * 1.01, AUTO_ZOOM_MAX)                    # 1% safety margin for interpolation
    return z, bool(np.mean(req > AUTO_ZOOM_MAX) > 1 - AUTO_ZOOM_PERCENTILE / 100.0)


def analyze(cap, first, tracker, total, progress, preview, label):
    """Pass 1: track the point through the whole video, remember where it was on every frame."""
    h, w = first.shape[:2]
    positions, statuses = [tracker.pos.copy()], ["track"]
    frame = first
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        pos, status = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        positions.append(pos.copy())
        statuses.append(status)
        done = len(positions)
        progress.update(done, f"{label} {status}")
        if preview is not None and done % 3 == 0 and not preview.show(frame, done, total, label):
            raise Cancelled()
    return np.array(positions, np.float32), statuses


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app_mode = args.video is None           # started by double click, not from a shell

    def fail(msg, code):
        if app_mode:
            alert(msg)
        else:
            log(msg)
        return code

    src = args.video or ask_video_file()
    if not src:
        if app_mode:
            return 0                        # dialog cancelled
        return fail("No video given. Usage: python stabilize.py video.mp4", 2)
    if not os.path.isfile(src):
        return fail(f"File not found: {os.path.basename(src)}", 2)

    try:
        auto = args.zoom.strip().lower() == "auto"
        zoom = 1.0 if auto else float(args.zoom)
        assert zoom >= 1.0
    except Exception:
        return fail("--zoom must be 'auto' or a number >= 1, e.g. 1.15", 2)

    cap = cv2.VideoCapture(src)
    ok, first = cap.read()
    if not ok:
        return fail(f"Could not open \"{os.path.basename(src)}\" as a video. "
                    f"Please choose a video file (mp4, mov, ...).", 1)
    h, w = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dst = args.output or os.path.splitext(src)[0] + "_stabilized.mp4"
    if os.path.abspath(dst) == os.path.abspath(src):
        return fail("Output path equals input path", 2)
    if app_mode:
        dst = unique_path(dst)
    ffmpeg = find_ffmpeg()

    log(f"Input : {src}")
    log(f"Video : {w}x{h}, {fps:.3f} fps, ~{total} frames")
    log(f"Output: {dst}")
    log(f"ffmpeg: {ffmpeg or 'NOT FOUND - audio will be lost, mp4v encoding'}")

    # --- which point to lock onto
    suggestions = np.zeros((0, 2), np.float32)
    if not args.point or args.point == "auto":
        log("Looking for still details in the first seconds of the video...")
        frames, sscale = sample_frames(src)
        suggestions = suggest_points(frames, sscale)
        log(f"Found {len(suggestions)} still detail(s)" +
            (f", best at ({suggestions[0][0]:.0f}, {suggestions[0][1]:.0f})" if len(suggestions) else ""))

    if args.point == "auto":
        if not len(suggestions):
            return fail("Could not find a steady detail to lock onto. "
                        "Pass --point X,Y or run without --point and click one.", 2)
        click = (float(suggestions[0][0]), float(suggestions[0][1]))
    elif args.point:
        try:
            click = tuple(float(v) for v in args.point.split(","))
            assert len(click) == 2
        except Exception:
            return fail("--point must look like 640,360 (or 'auto')", 2)
    else:
        log("Click the point to stabilize in the window, then press Enter...")
        click = pick_point(first, suggestions)
        if click is None:
            log("Cancelled.")
            return 1
    log(f"Locked point: ({click[0]:.1f}, {click[1]:.1f})")

    border = BORDER_MODES[args.border]
    anchor = np.array(click, np.float32)
    tracker = PointTracker(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), click)
    if np.linalg.norm(tracker.bias) > 0.01:
        log(f"Snapped to a nearby corner ({np.linalg.norm(tracker.bias):.1f} px away)")

    preview = Preview(w, h) if (app_mode or args.preview) else None
    started = time.time()
    positions = statuses = None
    z = zoom
    cleanup = lambda: (cv2.destroyAllWindows(), cv2.waitKey(1)) if preview is not None else None

    try:
        if auto:
            # pass 1: where does the point go? -> how much zoom hides every edge
            log("Pass 1/2: tracking the point through the video...")
            progress = Progress(total)
            positions, statuses = analyze(cap, first, tracker, total, progress, preview, "1/2 tracking")
            progress.finish()
            z, capped = auto_zoom_factor(positions, anchor, w, h)
            log(f"Zoom: {z:.3f}x (auto)" + ("  - capped, the shake is stronger than that" if capped else ""))
            cap.release()
            cap = cv2.VideoCapture(src)
            ok, first = cap.read()
            log("Pass 2/2: rendering...")
        sink = FFmpegSink(ffmpeg, src, dst, w, h, fps, args.crf) if ffmpeg else CvSink(dst, w, h, fps)
    except (KeyboardInterrupt, Cancelled):
        cap.release()
        cleanup()
        log("Cancelled.")
        return 130
    except Exception as e:
        cap.release()
        cleanup()
        return fail(f"Error: {e}", 1)

    label = "2/2 rendering" if auto else ""
    progress = Progress(len(positions) if positions is not None else total)
    if preview is not None:
        preview.start = time.time()
    frame, idx = first, 0
    pos, status = (positions[0], statuses[0]) if positions is not None else (tracker.pos, "track")
    out = first
    done = 0
    max_shift = 0.0
    try:
        while True:
            # shift = anchor - tracked point; the zoom (if any) is about the anchor
            M = np.array([[z, 0, anchor[0] - z * pos[0]],
                          [0, z, anchor[1] - z * pos[1]]], np.float32)
            max_shift = max(max_shift, float(np.linalg.norm(anchor - pos)))
            out = frame if (z == 1.0 and not np.any(M[:, 2])) else cv2.warpAffine(
                frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=border)
            sink.write(out)
            done += 1
            progress.update(done, f"{label} {status}".strip())
            if preview is not None and done % 2 == 0 and not preview.show(out, done, progress.total, label or status):
                raise Cancelled()

            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            idx += 1
            if positions is not None:
                k = min(idx, len(positions) - 1)
                pos, status = positions[k], statuses[k]
            else:
                pos, status = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        progress.update(done, "", force=True)
        progress.finish()
        log("Finishing the encode and adding audio...")
        if preview is not None:
            preview.show(out, progress.total or done, progress.total or done, "adding audio...")
        sink.close()
    except (KeyboardInterrupt, Cancelled):
        progress.finish()
        sink.abort()
        cleanup()
        log("Cancelled.")
        return 130
    except Exception as e:
        progress.finish()
        sink.abort()
        cleanup()
        return fail(f"Error: {e}", 1)
    finally:
        cap.release()

    log("")
    log(f"Frames processed : {done}")
    log(f"Track lost       : {tracker.lost_events} time(s) - found again {tracker.recovered}, "
        f"re-seeded {tracker.reseeds}, position held on {tracker.held_frames} frame(s)")
    log(f"Max frame shift  : {max_shift:.1f} px")
    log(f"Zoom             : {z:.3f}x" + (" (auto)" if auto else ""))
    log(f"Time             : {fmt_time(time.time() - started)}")
    log(f"Saved            : {dst}")
    if not ffmpeg:
        log("Install ffmpeg to keep the original audio: https://ffmpeg.org/download.html")
    if app_mode:
        reveal(dst)
    if preview is not None:
        preview.finish(out, f"Done - saved {os.path.basename(dst)}", 2500 if app_mode else 600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
