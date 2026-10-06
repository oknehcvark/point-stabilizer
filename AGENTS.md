# Point Stabilizer - notes for AI coding agents

This repository is a video stabilizer that locks a clip onto one point. It is meant to be
used both as a command-line tool / desktop app and as an agent skill.

## If the user wants to stabilize a video

Read `skills/point-stabilizer/SKILL.md` and follow it. In short:

```bash
python3 skills/point-stabilizer/scripts/stabilize.py "video.mp4"            # a window opens, the user clicks the point
python3 skills/point-stabilizer/scripts/stabilize.py "video.mp4" --point auto
```

Tell the user before a window opens. It needs a screen on the user's own machine.

## If you are changing the code

- The whole engine is one file: `skills/point-stabilizer/scripts/stabilize.py`. Keep it a single
  file with no dependencies beyond `opencv-python` and `numpy` (and the `ffmpeg` binary).
  The root `stabilize.py` is only a launcher.
- Tests: `pip install -r requirements.txt pytest && pytest -q`. They build synthetic shaky
  scenes in memory, so they need no video files and no ffmpeg.
- Do not commit videos or build output (`dist/`, `build/`).
- When you change tracking, check the accuracy numbers in `tests/test_engine.py`; they are
  deliberately tight.
