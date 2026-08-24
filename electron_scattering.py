"""
Electron-Electron Scattering — MP4 animation
============================================

This script numerically solves the scattering of two identical electrons under
mutual Coulomb repulsion with a fourth-order Runge-Kutta integrator and renders
the result as an MP4 video:

  - Electron 2 starts at rest at the origin.
  - Electron 1 approaches from (-X0, B) with speed V0 and impact parameter B.
  - The only force is mutual Coulomb repulsion: F(r) = K / r^2
    (K = k*e^2 in scaled simulation units).
  - The equations of motion, m * r_i'' = F_i, are integrated with RK4.

The video displays momentum and energy conservation in three live panels:
  1) Main panel: trajectories, velocity vectors, and traces.
  2) Momentum: the head-to-tail sum p1 + p2 always reaches the same total
     momentum point.
  3) Energy: KE1, KE2, U(r), and total-energy bars show the KE <-> U exchange
     while total energy remains on the fixed dashed line.

Requirements: numpy and matplotlib.
  - MP4 output requires ffmpeg. The script tries, in order:
      1) ffmpeg installed on the system PATH.
      2) The ffmpeg binary supplied by `pip install imageio-ffmpeg`.
      3) A GIF fallback using Pillow if neither ffmpeg option is available.
  - The simplest MP4 setup is `pip install imageio-ffmpeg`; no brew/apt
    installation is then required.
  - After saving, the script attempts to open the result in the system's
    default video player.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================== PARAMETERS
M   = 1.0     # electron mass (simulation units, not SI)
V0  = 6.0     # initial speed of electron 1
B   = 2.0     # impact parameter
K   = 8.0     # interaction strength, approximately k*e^2 (scaled)
X0  = 34.0    # initial x coordinate of electron 1: (-X0, B)

WORLD_HALF_W = 40.0   # visible half-width of the main panel
WORLD_HALF_H = 24.0   # visible half-height of the main panel

FPS          = 30
DURATION_S   = 11          # duration of the main motion in seconds
N_FRAMES     = FPS * DURATION_S
HOLD_FRAMES  = int(FPS * 2.0)   # hold the final frame for readability

OUT_FILE = "electron_scattering.mp4"

# ---------------------------------------------------------------- dark-theme palette
COL_BG     = "#0a1210"
COL_SURF   = "#111c19"
COL_PANEL  = "#16221e"
COL_INK    = "#e8efe9"
COL_MUTED  = "#8ea099"
COL_LINE   = "#25342d"
COL_E1     = "#57cbe3"   # electron 1 (bright cyan)
COL_E2     = "#ea92b4"   # electron 2 (bright pink)
COL_ENERGY = "#f0a84a"   # potential energy (amber)
COL_GOOD   = "#63dd92"   # conservation indicator (green)

FONT_MONO  = "DejaVu Sans Mono"
FONT_SERIF = "DejaVu Serif"


# ============================================================== PHYSICS ENGINE
def accel(r1, r2, K):
    """Return accelerations caused by mutual Coulomb repulsion."""
    d = r1 - r2
    r2d = d @ d
    r = np.sqrt(r2d)
    f = K / (r2d * r)          # |F| = K / r^2 ,  a = F/m (m=1)
    return f * d, -f * d


def deriv(state, K):
    r1, r2, v1, v2 = state
    a1, a2 = accel(r1, r2, K)
    return np.array([v1, v2, a1, a2])


def rk4_step(state, K, h):
    k1 = deriv(state, K)
    k2 = deriv(state + 0.5 * h * k1, K)
    k3 = deriv(state + 0.5 * h * k2, K)
    k4 = deriv(state + h * k3, K)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def momentum(state):
    _, _, v1, v2 = state
    return v1 + v2


def energy(state, K):
    r1, r2, v1, v2 = state
    r = np.linalg.norm(r1 - r2)
    KE1 = 0.5 * (v1 @ v1)
    KE2 = 0.5 * (v2 @ v2)
    U = K / r
    return KE1, KE2, U, KE1 + KE2 + U


def simulate():
    """Integrate with RK4 until electron 1 passes X0*0.98."""
    state = np.array([[-X0, B], [0.0, 0.0], [V0, 0.0], [0.0, 0.0]])
    t = 0.0
    times = [t]
    R1 = [state[0].copy()]; R2 = [state[1].copy()]
    V1 = [state[2].copy()]; V2 = [state[3].copy()]

    steps, max_steps = 0, 400_000
    while state[0, 0] < X0 * 0.98 and steps < max_steps:
        r = np.linalg.norm(state[0] - state[1])
        h = 0.0006 if r < 3.0 else 0.004      # smaller step near closest approach
        state = rk4_step(state, K, h)
        t += h
        times.append(t)
        R1.append(state[0].copy()); R2.append(state[1].copy())
        V1.append(state[2].copy()); V2.append(state[3].copy())
        steps += 1

    return (np.array(times), np.array(R1), np.array(R2),
            np.array(V1), np.array(V2))


print("Solving the dynamics with RK4...")
times, R1, R2, V1, V2 = simulate()
print(f"  {len(times)} steps, final time = {times[-1]:.2f}")

P0 = V1[0] + V2[0]
E0 = sum(energy(np.array([R1[0], R2[0], V1[0], V2[0]]), K)[:3])

# Measured scattering angle versus Rutherford prediction: b = b0 cot(theta/2)
vrel0 = np.array([V0, 0.0])
vrelF = V1[-1] - V2[-1]
cosang = np.dot(vrel0, vrelF) / (np.linalg.norm(vrel0) * np.linalg.norm(vrelF))
theta_meas = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
mu = M / 2.0
b0 = K / (mu * V0 ** 2)
theta_pred = 2 * np.degrees(np.arctan(b0 / B))
print(f"  measured angle = {theta_meas:.2f} deg")
print(f"  predicted angle (Rutherford) = {theta_pred:.2f} deg")

# --------------------------------------------------------- sample video frames
# Because the time step shrinks near closest approach, uniform index sampling
# naturally produces a physically consistent slow-motion effect there.
idx = np.linspace(0, len(times) - 1, N_FRAMES).astype(int)
Tf, R1f, R2f, V1f, V2f = times[idx], R1[idx], R2[idx], V1[idx], V2[idx]

# Repeat the final frame so the result remains readable.
rep = np.full(HOLD_FRAMES, len(Tf) - 1)
frame_idx = np.concatenate([np.arange(len(Tf)), rep])
TOTAL_FRAMES = len(frame_idx)

Pf = V1f + V2f
KE1f = 0.5 * np.sum(V1f ** 2, axis=1)
KE2f = 0.5 * np.sum(V2f ** 2, axis=1)
Rf = np.linalg.norm(R1f - R2f, axis=1)
Uf = K / Rf
Ef = KE1f + KE2f + Uf
Pdrift = np.linalg.norm(Pf - P0, axis=1)
Edrift = np.abs(Ef - E0)
dotf = np.sum(V1f * V2f, axis=1)   # v1'.v2' -> perpendicularity check


# ============================================================== DRAWING HELPERS
def style_axes(ax):
    ax.set_facecolor(COL_PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=COL_MUTED, labelsize=7)


def draw_arrow(ax, p0, p1, color, lw=2.2, ls="-", z=5):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                         linestyle=ls, shrinkA=0, shrinkB=0,
                         mutation_scale=14),
        zorder=z,
    )


# ============================================================== FIGURE SETUP
fig = plt.figure(figsize=(13.2, 7.4), dpi=140, facecolor=COL_BG)
gs = fig.add_gridspec(
    3, 3, width_ratios=[2.3, 2.3, 1.35], height_ratios=[0.16, 1, 1],
    hspace=0.55, wspace=0.32, left=0.045, right=0.975, top=0.90, bottom=0.07,
)

ax_formula = fig.add_subplot(gs[0, :]); ax_formula.axis("off")
ax_main    = fig.add_subplot(gs[1:, 0:2])
ax_mom     = fig.add_subplot(gs[1, 2])
ax_energy  = fig.add_subplot(gs[2, 2])

fig.suptitle("Electron-Electron Scattering", fontsize=19, fontweight="bold",
             color=COL_INK, family=FONT_SERIF, y=0.975)

ax_formula.text(
    0.5, 0.45,
    r"$\vec{P}=\vec{p}_1+\vec{p}_2=\mathrm{constant}\qquad\qquad$"
    r"$E=\frac{1}{2}mv_1^2+\frac{1}{2}mv_2^2+\frac{ke^2}{r}=\mathrm{constant}$",
    transform=ax_formula.transAxes, ha="center", va="center",
    fontsize=13, color=COL_INK,
)

E_YMAX = E0 * 1.18
MOM_SCALE_REF = max(V0, 6.0)


def draw_main(i):
    ax_main.cla()
    style_axes(ax_main)
    ax_main.set_xlim(-WORLD_HALF_W, WORLD_HALF_W)
    ax_main.set_ylim(-WORLD_HALF_H, WORLD_HALF_H)
    ax_main.set_aspect("equal")
    ax_main.set_xticks(np.arange(-WORLD_HALF_W, WORLD_HALF_W + 1, 8))
    ax_main.set_yticks(np.arange(-WORLD_HALF_H, WORLD_HALF_H + 1, 8))
    ax_main.grid(True, color=COL_LINE, linewidth=0.8, zorder=0)
    ax_main.axhline(0, color=COL_MUTED, linewidth=1, alpha=0.5, zorder=1)

    ax_main.plot(R1f[:i + 1, 0], R1f[:i + 1, 1], color=COL_E1, lw=2, alpha=0.6, zorder=2)
    ax_main.plot(R2f[:i + 1, 0], R2f[:i + 1, 1], color=COL_E2, lw=2, alpha=0.6, zorder=2)

    p1, p2 = R1f[i], R2f[i]
    v1, v2 = V1f[i], V2f[i]
    vscale = 0.9
    if np.linalg.norm(v1) > 0.05:
        draw_arrow(ax_main, p1, p1 + v1 * vscale, COL_E1, lw=1.6)
    if np.linalg.norm(v2) > 0.05:
        draw_arrow(ax_main, p2, p2 + v2 * vscale, COL_E2, lw=1.6)

    for p, c in ((p2, COL_E2), (p1, COL_E1)):
        ax_main.scatter([p[0]], [p[1]], s=210, color=c, zorder=6,
                         edgecolors=COL_SURF, linewidths=1.4)
        ax_main.text(p[0], p[1], "–", ha="center", va="center",
                      color=COL_SURF, fontsize=11, fontweight="bold", zorder=7)

    ax_main.text(
        0.015, 0.965, f"t = {Tf[i]:6.2f}    r = {Rf[i]:6.2f}",
        transform=ax_main.transAxes, ha="left", va="top",
        fontsize=9.5, family=FONT_MONO, color=COL_MUTED,
    )
    ax_main.text(
        0.015, 0.045,
        f"v'₁·v'₂ = {dotf[i]:+7.3f}   (perpendicularity → 0 means 90°)",
        transform=ax_main.transAxes, ha="left", va="bottom",
        fontsize=9, family=FONT_MONO, color=COL_MUTED,
    )

    if i >= len(Tf) - 1:
        box = (f"θ_measured = {theta_meas:5.1f}°   "
               f"θ_Rutherford = {theta_pred:5.1f}°")
        ax_main.text(
            0.985, 0.965, box, transform=ax_main.transAxes,
            ha="right", va="top", fontsize=9.5, family=FONT_MONO,
            color=COL_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=COL_SURF,
                      edgecolor=COL_LINE),
        )


def draw_mom(i):
    ax_mom.cla()
    style_axes(ax_mom)
    ax_mom.set_title("Momentum", loc="left", fontsize=10.5, fontweight="bold",
                      color=COL_INK, family=FONT_SERIF, pad=6)
    lim = MOM_SCALE_REF * 1.15
    ax_mom.set_xlim(-lim * 0.25, lim * 1.15)
    ax_mom.set_ylim(-lim * 0.55, lim * 0.55)
    ax_mom.set_xticks([]); ax_mom.set_yticks([])

    origin = np.array([0.0, 0.0])
    draw_arrow(ax_mom, origin, P0, COL_MUTED, lw=1.6, ls=(0, (4, 3)), z=3)
    ax_mom.text(P0[0] + 0.6, P0[1], "P", fontsize=10, color=COL_MUTED,
                family=FONT_MONO, va="center")

    p2_tip = Pf[i] - V1f[i]     # draw p2 first in the head-to-tail sum
    draw_arrow(ax_mom, origin, p2_tip, COL_E2, lw=2.4, z=5)
    draw_arrow(ax_mom, p2_tip, Pf[i], COL_E1, lw=2.4, z=5)

    ax_mom.text(0.02, 0.02, f"p₁=({V1f[i,0]:.2f}, {V1f[i,1]:.2f})",
                transform=ax_mom.transAxes, fontsize=8, family=FONT_MONO,
                color=COL_E1, va="bottom")
    ax_mom.text(0.02, 0.10, f"p₂=({V2f[i,0]:.2f}, {V2f[i,1]:.2f})",
                transform=ax_mom.transAxes, fontsize=8, family=FONT_MONO,
                color=COL_E2, va="bottom")
    ax_mom.text(0.98, 0.90,
                f"drift: {Pdrift[i]:.1e} ✓", transform=ax_mom.transAxes,
                fontsize=8, family=FONT_MONO, color=COL_GOOD, ha="right")


def draw_energy(i):
    ax_energy.cla()
    style_axes(ax_energy)
    ax_energy.set_title("Energy", loc="left", fontsize=10.5, fontweight="bold",
                         color=COL_INK, family=FONT_SERIF, pad=6)
    labels = ["KE₁", "KE₂", "U(r)", "E$_{total}$"]
    vals = [KE1f[i], KE2f[i], Uf[i], Ef[i]]
    colors = [COL_E1, COL_E2, COL_ENERGY, COL_INK]
    xs = np.arange(4)
    bars = ax_energy.bar(xs, vals, color=colors, width=0.62, zorder=4)
    ax_energy.axhline(E0, color=COL_MUTED, linewidth=1.2,
                       linestyle=(0, (4, 3)), zorder=3)
    ax_energy.set_ylim(0, E_YMAX)
    ax_energy.set_xticks(xs)
    ax_energy.set_xticklabels(labels, fontsize=8.5, family=FONT_MONO,
                               color=COL_MUTED)
    ax_energy.set_yticks([])
    for rect, v in zip(bars, vals):
        ax_energy.text(rect.get_x() + rect.get_width() / 2, v + E_YMAX * 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.6,
                        family=FONT_MONO, color=COL_INK)
    ax_energy.text(0.98, 0.94, f"drift: {Edrift[i]:.1e} ✓",
                    transform=ax_energy.transAxes, fontsize=8,
                    family=FONT_MONO, color=COL_GOOD, ha="right")


def update(frame):
    i = frame_idx[frame]
    draw_main(i)
    draw_mom(i)
    draw_energy(i)
    return []


print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)

# Find ffmpeg on the system, then through imageio-ffmpeg, or fall back to GIF.
ffmpeg_path = shutil.which("ffmpeg")
source = "system installation"

if not ffmpeg_path:
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        source = "imageio-ffmpeg package"
    except Exception:
        ffmpeg_path = None

if ffmpeg_path:
    print(f"  Found ffmpeg ({source}): {ffmpeg_path}; saving as MP4.")
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4200,
                                     extra_args=["-pix_fmt", "yuv420p"])
    out_path = OUT_FILE
else:
    # Pillow provides a dependency-free GIF fallback when ffmpeg is unavailable.
    print("  WARNING: ffmpeg was not found on the system or via imageio-ffmpeg.")
    print("           Saving a GIF instead of an MP4.")
    print("           For MP4 output, run: pip install imageio-ffmpeg")
    print("           Alternatives: 'brew install ffmpeg' or 'sudo apt install ffmpeg'.")
    writer = animation.PillowWriter(fps=FPS)
    out_path = os.path.splitext(OUT_FILE)[0] + ".gif"

try:
    ani.save(out_path, writer=writer)
except Exception as exc:
    print(f"\nERROR: could not save the video/GIF: {exc}", file=sys.stderr)
    print("Check the matplotlib installation and ffmpeg setup, if applicable.",
          file=sys.stderr)
    plt.close(fig)
    sys.exit(1)

plt.close(fig)
abs_path = os.path.abspath(out_path)
print(f"\nFinished: {abs_path}")
print(f"File size: {os.path.getsize(abs_path) / 1e6:.2f} MB")
print(f"Working directory: {os.getcwd()}")

# Open the result in the system's default media player.
try:
    if sys.platform == "darwin":
        subprocess.run(["open", abs_path], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", abs_path], check=False)
    elif sys.platform.startswith("win"):
        os.startfile(abs_path)  # noqa: E501  (defined only on Windows)
    print("Attempted to open the result in the default media player.")
except Exception as exc:
    print(f"Could not open the file automatically; open it manually: {abs_path} ({exc})")