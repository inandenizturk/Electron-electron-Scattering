# Electron-Electron Scattering

A Python animation of two identical electrons scattering under mutual Coulomb
repulsion. The equations of motion are solved numerically with a fourth-order
Runge-Kutta integrator.

The generated video shows:

- electron trajectories and velocity vectors;
- momentum conservation with a head-to-tail vector diagram;
- kinetic, potential, and total energy throughout the interaction;
- measured and Rutherford-predicted scattering angles.

## Requirements

- Python 3
- NumPy
- Matplotlib
- ffmpeg or `imageio-ffmpeg` for MP4 output

## Run

```bash
python -m pip install -r requirements.txt
python electron_scattering.py
```

The script writes `electron_scattering.mp4`. If ffmpeg is unavailable, it
falls back to `electron_scattering.gif`.
