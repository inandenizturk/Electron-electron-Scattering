---
name: testing-electron-scattering
description: How to set up, run, and visually verify the electron_scattering.py Coulomb-scattering animation (MP4/GIF output) in a headless Devin box.
---

# Testing the electron-electron scattering animation

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy 2.2.1, matplotlib 3.10.0, imageio-ffmpeg 0.6.0
which ffmpeg                                # system ffmpeg is normally present at /usr/bin/ffmpeg
```
No secrets or credentials are needed. **Devin Secrets Needed:** none.

## Running
```bash
rm -f electron_scattering.mp4 electron_scattering.gif
.venv/bin/python electron_scattering.py     # ~40 s wall clock; run it backgrounded and poll get_output
```
Expected stdout: RK4 step count, measured vs Rutherford angle (~24.7° vs ~25.1°),
"Rendering 390 frames at 30 fps...", "Found ffmpeg (system installation) ...; saving as MP4",
then the absolute path and file size (~1.1 MB). The final `xdg-open` attempt is best-effort and
its failure is not a test failure.

## Verifying the video without a browser
`ffprobe` for structure (expect h264 / yuv420p / 1848x1036 / 30 fps / 390 frames / 13.0 s) and
`ffmpeg -v error -i file -f null -` for a clean decode.

The Devin `browser` tool may fail to attach in this environment (all actions returning
"Browser action failed"). Reliable headless-GUI fallbacks on `DISPLAY=:0`:
- Play video: `DISPLAY=:0 ffplay -autoexit -loglevel error -x 1500 -y 840 file.mp4 &`
- Show a PNG: `DISPLAY=:0 display -geometry +30+30 -resize 1500x840 frame.png &`
- Screenshot: `DISPLAY=:0 scrot /tmp/shot.png`
Kill any Chrome windows first (`pkill -f chrome-linux64/chrome`) so the player is on a clean
desktop. Note `ffplay -ss N` seeks to the nearest keyframe, so extract exact frames with
`ffmpeg -ss T -i file.mp4 -frames:v 1 out.png` when you need a specific timestamp.

## Exercising the GIF fallback (no source edits)
Run a wrapper that hides both ffmpeg binaries:
```bash
PATH=/nonexistent .venv/bin/python -c '<meta_path hook raising ImportError for imageio_ffmpeg>; exec(open("electron_scattering.py").read())'
```
Expect "WARNING: ffmpeg was not found ..." and a multi-MB `electron_scattering.gif`
(PillowWriter emits 330 frames — it does not include the duplicated hold frames).

## Things to check in frames
Title "Electron-Electron Scattering", conserved P/E formula row, main panel with two electron
markers + traces + velocity arrows + `t=/r=` readout + "(perpendicularity → 0 means 90°)",
"Momentum" panel (head-to-tail p₁/p₂, dashed P, drift ✓), "Energy" panel (KE₁, KE₂, U(r),
E_total bars + dashed E₀ line), and on the last frame the θ_measured/θ_Rutherford box.
Known cosmetic issue: the KE₁ bar's value label can overlap the dashed total-energy line.
