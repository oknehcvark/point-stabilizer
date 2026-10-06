---
name: point-stabilizer
description: Stabilize shaky video by locking it onto one still point (motion-track stabilization). Use when the user wants to stabilize, steady or de-shake a video, pin / lock / anchor a point so it stays put, or "track a point and hold it still". The user clicks the point in a window, or the tool picks a still detail by itself. Keeps the audio and zooms in just enough to hide the edges.
---

# Point stabilizer

Locks a video onto one point: it tracks the point through the clip and shifts every
frame so the point stays exactly where it was, then zooms in just enough that no black
edges show. Audio is kept. The engine is `scripts/stabilize.py` in this skill's folder.

## 1. Check the setup (once)

```bash
python3 -c "import cv2, numpy; print('opencv', cv2.__version__)"
ffmpeg -version | head -1
```

- Missing OpenCV or NumPy: `pip3 install opencv-python numpy`
- Missing ffmpeg: `brew install ffmpeg` (macOS) or `winget install ffmpeg` (Windows). Without
  it the tool still works, but the result has no audio.

## 2. Run it

Use the full path to `scripts/stabilize.py` inside this skill's folder.

**The user picks the point (default, the best option).** A window with the first frame
opens on the user's screen; they click a detail (or press Enter to accept the green
suggestion) and the work continues by itself.

```bash
python3 scripts/stabilize.py "/path/to/video.mp4"
```

- Tell the user BEFORE you run it: "A window will open - click the detail that should stay
  still, then press Enter."
- The command waits for the click. Run it in the background or with a long timeout
  (several minutes) and wait for it to finish.
- Needs the agent to run on the user's own computer with a screen. On a remote machine,
  in the cloud or over SSH no window can open - use one of the options below.

**Automatic (no window).** The tool picks a still detail itself:

```bash
python3 scripts/stabilize.py "/path/to/video.mp4" --point auto
```

**Explicit coordinates** (pixels in the first frame):

```bash
python3 scripts/stabilize.py "/path/to/video.mp4" --point 640,360
```

Useful options: `-o out.mp4` (default `<name>_stabilized.mp4` next to the video),
`--zoom auto|N` (`auto` is the default; `1` turns the zoom off), `--border black|replicate|reflect`,
`--crf 18` (quality, lower is better), `--preview` (show progress in a window).

## 3. Report the result

The last lines of the output say what happened. Tell the user:

- where the file is (`Saved: ...`) and the zoom that was applied (`Zoom: 1.13x`);
- whether the track was lost (`Track lost: N time(s)`) - a few recoveries are normal;
- which point was used (`Locked point: (x, y)`).

If `Zoom` hit the cap (`capped`), the shake is stronger than the tool can hide; say so.

## 4. If it does not work

- `--point auto` says it found no still detail: the camera itself moves (pan, dolly, zoom)
  or the clip is edited with cuts and animated zooms. Ask the user to click a point
  themselves or give coordinates; if the whole picture moves on purpose there is nothing
  to lock onto.
- The picture still wobbles: the point was on something that moves (a person, a hand,
  a screen with video). Pick a different one.
- "weak point" in the window: the spot is flat. Pick a corner or an edge with contrast.

## What makes a good point

A **still** object in the real world (corner of a window, edge of a table, a sign on a
wall) that is **sharp and high-contrast** and **visible for the whole clip**. Not a person,
a hand, the sky or a plain wall.

## Limits

Corrects shift only (no rotation or zoom of the camera). One point, chosen on the first
frame. A large object passing over the point can drag the track for a few frames; it is
found again when it becomes visible. See the README for details.
