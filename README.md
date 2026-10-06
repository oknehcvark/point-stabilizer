# Point Stabilizer

**Lock a video onto a point.** Pick a still detail in the first frame (or let the program pick one),
and every frame is shifted so that detail stays exactly where it was. The picture is then zoomed in
just enough to hide the edges. Audio is kept.

![before and after](docs/before-after.gif)

*Left: simulated hand shake. Right: locked onto the cross. [`docs/make_demo.py`](docs/make_demo.py) rebuilds this clip.*

On a test clip with up to ±29 px of simulated shake, the locked detail wandered **20 px on average
before and 0.03 px after** (1280 px wide, 240 frames, no frames lost).

[Русская версия](README.ru.md)

## Use it

```bash
pip install -r requirements.txt      # opencv-python, numpy
python stabilize.py video.mp4
```

A window opens with the first frame. **Click a still detail and press Enter** - or just press Enter to
take the green suggestion. Esc cancels. The result is saved next to the video as
`video_stabilized.mp4`, with the original audio.

You also need the **ffmpeg** binary (`brew install ffmpeg`, `winget install ffmpeg`, `apt install ffmpeg`).
Without it the tool still runs, but the result has no audio and uses a simpler encoder.

| Option | Meaning |
|---|---|
| `--point auto` | no window: the program picks a still detail itself |
| `--point X,Y` | no window: lock onto this pixel of the first frame |
| `--zoom auto` (default) | zoom in just enough to hide every black edge; `--zoom 1.15` sets it by hand, `--zoom 1` turns it off |
| `--border black\|replicate\|reflect` | what to put in the edges when the zoom is off |
| `--crf 18` | video quality (lower is better and bigger) |
| `--preview` | show the frames and a progress bar in a window |
| `-o out.mp4` | output path |

Started without a video argument (for example, as the packaged app), it asks for a file, shows
progress in a window and reveals the result in Finder / Explorer.

## Choosing the point

Lock onto something that is **still in the real world**, **sharp**, **high-contrast** and **visible for the
whole clip**: a window corner, the edge of a table, a sign on a wall. Not a person, a hand, the sky or a
plain wall. The window tells you if you clicked a weak spot, and the green markers are details the program
found that follow the camera exactly.

## How it works

1. **Tracking.** Pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) follows the point
   frame by frame, with a forward-backward check to notice when it slips.
2. **Losing the point.** If the track is lost (blur, a flash, something passes over it) the position is
   held and the point is searched again near where it was: first by appearance (template match against the
   first frame, refined to sub-pixel and verified geometrically); if it stays lost for 12 frames, new
   corners are seeded nearby and the position carries on from their motion, without a jump.
3. **Zoom.** The video is processed in two passes: the first tracks the point through the whole clip,
   then the smallest constant zoom that keeps every frame covered is computed (capped at 1.6x).
4. **Suggestions.** Corners of the first seconds are tracked and fitted to one camera model
   (shift, rotation, zoom, robust to outliers); corners that stay on it are still objects.
5. **Output.** Frames are piped to ffmpeg (H.264) and the original audio is muxed in; the picture is never
   cut because the audio is a hair shorter.

## Limits

- It corrects **shift** only, not rotation or the camera zooming.
- One point, chosen on the **first frame**. If the thing you want appears later, trim the clip.
- A large object that slides over the point can drag the track for a few frames. In a test this was up to
  ~48 px for 4 frames; the point is found again as soon as it is visible.
- No still detail exists in clips where the camera itself moves or that are edited with cuts and animated
  zooms; then `--point auto` says so and you click manually.
- Tested with OpenCV 4.14 and 5.0 on macOS (Apple Silicon). The CI matrix also runs Linux and Windows.

## Use it from Claude Code or Codex

The repository is also an *agent skill*. Install it, and then ask your agent, for example,
"stabilize `reel.mp4`": it opens the window on your screen, you click the point, and it does the rest.

```bash
npx skills add oknehcvark/point-stabilizer
```

The skill is in [`skills/point-stabilizer`](skills/point-stabilizer/SKILL.md); [`AGENTS.md`](AGENTS.md) covers
agents working inside this repository. The agent must run on your own computer to open the window; on a
remote machine it falls back to `--point auto` or coordinates.

## Build the macOS app yourself

```bash
pip install pyinstaller imageio-ffmpeg opencv-python numpy
FFMPEG=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
cp "$FFMPEG" ./ffmpeg
pyinstaller --windowed --name "Point Stabilizer" --add-binary "./ffmpeg:." \
  --exclude-module tkinter skills/point-stabilizer/scripts/stabilize.py
```

The app lands in `dist/` and needs no Python or ffmpeg installed.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

The tests build synthetic shaky scenes in memory (flash, blur burst, jolt) and check sub-pixel accuracy,
loss recovery, that the zoom hides every edge, and that suggested points are really still.

## License

[MIT](LICENSE). Requires OpenCV (Apache-2.0), NumPy (BSD) and the FFmpeg binary (LGPL/GPL depending on the
build; it is not part of this repository). A packaged app that bundles an FFmpeg build with x264 is
distributed under that build's GPL terms.
