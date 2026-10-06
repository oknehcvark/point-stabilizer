#!/usr/bin/env python3
"""Convenience launcher: `python stabilize.py video.mp4` from the repository root.

The real code lives in skills/point-stabilizer/scripts/stabilize.py, so that the
same file is what an AI agent gets when it installs the skill.
"""
import pathlib
import runpy

runpy.run_path(
    str(pathlib.Path(__file__).resolve().parent / "skills" / "point-stabilizer" / "scripts" / "stabilize.py"),
    run_name="__main__",
)
