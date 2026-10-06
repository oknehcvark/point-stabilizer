"""Engine tests on synthetic shaky scenes built in memory (no video files, no ffmpeg)."""
import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "skills" / "point-stabilizer" / "scripts"))
import stabilize  # noqa: E402

W, H = 640, 360
OX, OY = 200, 150
CLICK = (320.0, 180.0)
N = 60


@pytest.fixture(scope="module")
def scene():
    """A textured world plus a camera path. frame(i) is the world shifted by (tx[i], ty[i])."""
    rng = np.random.default_rng(7)
    canvas = cv2.GaussianBlur(rng.integers(60, 190, (900, 1400, 3), dtype=np.uint8), (0, 0), 2.0)
    for _ in range(260):
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, 1400)), int(rng.integers(0, 900))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + int(rng.integers(8, 60)), y + int(rng.integers(8, 60))), colour, -1)
        else:
            cv2.circle(canvas, (x, y), int(rng.integers(4, 30)), colour, -1)
    cv2.rectangle(canvas, (OX + 322, OY + 181), (OX + 390, OY + 240), (20, 20, 230), -1)   # a sharp corner near the click

    t = np.arange(N) / 30.0
    jitter = np.zeros((N, 2))
    for i in range(1, N):
        jitter[i] = 0.85 * jitter[i - 1] + rng.normal(0, 1.2, 2)
    tx = 14 * np.sin(2 * np.pi * 0.7 * t) + 8 * np.sin(2 * np.pi * 2.3 * t + 1) + jitter[:, 0]
    ty = 10 * np.sin(2 * np.pi * 0.5 * t + 2) + 6 * np.sin(2 * np.pi * 3.1 * t) + jitter[:, 1]
    tx, ty = tx - tx[0], ty - ty[0]

    def frame(i):
        m = np.float32([[1, 0, tx[i] - OX], [0, 1, ty[i] - OY]])
        return cv2.warpAffine(canvas, m, (W, H), flags=cv2.INTER_CUBIC)

    return frame, tx, ty


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_tracker(frame, tx, ty, mutate=None):
    tracker = stabilize.PointTracker(gray(frame(0)), np.array(CLICK, np.float32))
    errors, positions = [0.0], [tracker.pos.copy()]
    for i in range(1, N):
        img = frame(i)
        if mutate is not None:
            img = mutate(i, img)
        pos, _ = tracker.update(gray(img))
        positions.append(pos.copy())
        errors.append(float(np.linalg.norm(pos - (np.array(CLICK) + [tx[i], ty[i]]))))
    return tracker, np.array(errors), np.array(positions, np.float32)


def test_tracker_follows_the_shake_to_sub_pixel(scene):
    frame, tx, ty = scene
    _, errors, _ = run_tracker(frame, tx, ty)
    assert np.median(errors) < 0.3
    assert errors.max() < 1.0


def test_tracker_survives_a_flash_and_a_blur_burst(scene):
    frame, tx, ty = scene

    def damage(i, img):
        if i == 20:
            return np.full_like(img, 250)                       # exposure flash
        if 30 <= i <= 32:                                       # motion-blur burst
            blurred = cv2.GaussianBlur(img, (0, 0), 7)
            return np.clip(blurred.astype(np.int16) + np.random.default_rng(i).normal(0, 8, img.shape), 0, 255).astype(np.uint8)
        return img

    tracker, errors, _ = run_tracker(frame, tx, ty, damage)
    assert tracker.lost_events >= 1                              # the damage was noticed...
    assert errors[40:].max() < 1.0                               # ...and the point is exactly back


def test_auto_zoom_hides_every_edge(scene):
    frame, tx, ty = scene
    _, _, positions = run_tracker(frame, tx, ty)
    anchor = np.array(CLICK, np.float32)
    zoom, capped = stabilize.auto_zoom_factor(positions, anchor, W, H)
    assert 1.0 < zoom < stabilize.AUTO_ZOOM_MAX and not capped
    for i in range(N):
        m = np.array([[zoom, 0, anchor[0] - zoom * positions[i][0]],
                      [0, zoom, anchor[1] - zoom * positions[i][1]]], np.float32)
        out = cv2.warpAffine(frame(i), m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        edges = [out[:2, :], out[-2:, :], out[:, :2], out[:, -2:]]
        assert min(float(e.mean()) for e in edges) > 20, f"black edge on frame {i}"


def test_auto_zoom_is_neutral_without_shake_and_capped_when_extreme():
    still = np.tile(np.array(CLICK, np.float32), (30, 1))
    zoom, capped = stabilize.auto_zoom_factor(still, np.array(CLICK, np.float32), W, H)
    assert 1.0 <= zoom <= 1.02 and not capped
    wild = np.tile(np.array([W - 5.0, H - 5.0], np.float32), (30, 1))     # the point is almost off-screen
    zoom, capped = stabilize.auto_zoom_factor(wild, np.array(CLICK, np.float32), W, H)
    assert zoom == pytest.approx(stabilize.AUTO_ZOOM_MAX) and capped


def test_suggested_points_really_are_still(scene):
    frame, tx, ty = scene
    frames = [gray(frame(i)) for i in range(0, N, 2)]
    suggestions = stabilize.suggest_points(frames, 1.0)
    assert len(suggestions) >= 1
    for point in suggestions:
        tracker = stabilize.PointTracker(gray(frame(0)), point)
        worst = 0.0
        for i in range(1, N):
            pos, _ = tracker.update(gray(frame(i)))
            worst = max(worst, float(np.linalg.norm(pos - (point + [tx[i], ty[i]]))))
        assert worst < 0.5


def test_no_suggestions_when_the_picture_is_blank():
    blank = [np.full((H, W), 120, np.uint8) for _ in range(10)]
    assert len(stabilize.suggest_points(blank, 1.0)) == 0


def test_cli_reports_a_missing_file(capsys):
    assert stabilize.main(["definitely-not-here.mp4", "--point", "1,1"]) == 2
    assert "not found" in capsys.readouterr().out.lower()


def test_cli_rejects_a_bad_zoom(tmp_path, capsys):
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"not a video")
    assert stabilize.main([str(fake), "--zoom", "abc", "--point", "1,1"]) == 2
