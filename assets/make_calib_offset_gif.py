"""Calibration_Pro and OffsetManager profile animations. Real repo functions on synthetic input.

  calibration-patchtest.gif  Calibration_Pro  'patch-observability' mode (desktop/entry.py) ->
                                              geoview_common.mbes_patch_observability.analyze_patch_observability
                                              (SVD of the declared linear model, free per-patch planes,
                                               returns parameter step, covariance, condition number, SSE)
  offsetmanager-frame.gif    OffsetManager    offsetmanager.axis_convention.detect_convention
                                              offsetmanager.core.transform_frame   (new = FLIP @ R @ (old - origin))
                                              offsetmanager.core.transform_reference_frame
                                              offsetmanager.core.calculate_distance

calibration-patchtest: three synthetic MBES lines (A and B reciprocal on one track, C parallel 60 m east)
over a sloped seabed with a ridge. The transducer has a hidden mounting misalignment (roll, pitch, yaw).
The script ray-traces each beam to the true seabed with the true angles (range + noise), then georeferences
the soundings with the current estimate and builds the 3 x 3 Jacobians by finite differences. Those are
the script's own work (stated in the footer). Each Gauss-Newton pass hands 8 patches to
analyze_patch_observability, which returns the step; the estimate is updated with it. The mismatch between
lines collapses and the last frame lists the repo's estimate, its 1-sigma from the returned covariance,
and the hidden truth.

offsetmanager-frame: a synthetic vessel with ten sensors measured in a deck frame (+X fwd, +Y port, +Z up,
origin at a deck mark). detect_convention reads the axis directions from the geometry, transform_frame
re-expresses all offsets about the MRU with a small yaw and the Y/Z flip (CRP: +Y stbd, +Z down),
detect_convention confirms the new convention, transform_reference_frame moves the zero to GNSS 1, and
calculate_distance gives the GNSS 1 -> MBES lever arm. The sensors do not move; only the frame does.

usage:  python make_calib_offset_gif.py <out_dir> [png_dir]
needs:  numpy, matplotlib, pillow, python-docx and openpyxl (imported by offsetmanager.core)
"""
import sys
sys.dont_write_bytecode = True
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
LAST_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"
sys.path.insert(0, "/home/user/repos/_shared")
sys.path.insert(0, "/home/user/repos/Preprocessing/OffsetManager")

C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
DEPTH = LinearSegmentedColormap.from_list("depth", ["#F6F8FA", "#DDF4FF", "#80CCFF", "#218BFF", "#0A3069"])
LINE_C = {"A": C["acc"], "B": "#BC4C00", "C": "#8250DF"}


# ---- figure helpers (as in make_visuals_ml.py) ----------------------------------------------
def fig_base(title, foot):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=C["bg"])
    prog, what = title
    fig.text(0.015, 0.935, prog, color=C["ink"], fontsize=9, fontweight="bold")
    fig.text(0.015 + 0.0128 * len(prog) + 0.008, 0.935, "·  " + what, color=C["mute"], fontsize=9)
    fig.text(0.015, 0.012, foot, color=C["mute"], fontsize=6, va="bottom", linespacing=1.4)
    status = fig.text(0.985, 0.935, "", ha="right", color=C["acc"], fontsize=7.5)
    return fig, status


def style(ax, grid=False):
    ax.set_facecolor(C["bg"])
    for s in ax.spines.values():
        s.set_color(C["hair"])
    ax.tick_params(colors=C["mute"], labelsize=6, length=2, pad=1.5)
    if grid:
        ax.grid(color=C["hair"], lw=0.5)


def label(ax, text, color=None):
    ax.text(0.0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom",
            color=color or C["mute"], fontsize=6.5)


def grab(fig):
    fig.canvas.draw()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")


def write_gif(frames, name, hold=30):
    pick = frames[::max(1, len(frames) // 8)] + [frames[-1]]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(pick)))
    for i, f in enumerate(pick):
        strip.paste(f, (0, i * f.height))
    q = strip.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    rgb = q.getpalette()[:200 * 3]
    for hx in list(C.values()) + list(LINE_C.values()):        # theme colours kept exact
        rgb += [int(hx[i:i + 2], 16) for i in (1, 3, 5)]
    ref = Image.new("P", (1, 1)); ref.putpalette(rgb + [0] * (768 - len(rgb)))
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / FPS)] * len(pal); dur[-1] = int(1000 / FPS) * hold
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=dur, loop=0,
                optimize=True, disposal=1)
    if LAST_DIR:
        LAST_DIR.mkdir(parents=True, exist_ok=True)
        frames[-1].save(LAST_DIR / (Path(name).stem + "-last.png"))
        frames[len(frames) // 2].save(LAST_DIR / (Path(name).stem + "-mid.png"))
    print(f"{name}: {len(pal)} frames, {(OUT / name).stat().st_size / 1024:.0f} KB")
    plt.close("all")


def sgn(v, nd=2):
    return f"{v:+.{nd}f}"


# =========================================================================================
# 1. Calibration_Pro - patch test solved by analyze_patch_observability
# =========================================================================================
def make_calibration():
    from geoview_common.mbes_patch_observability import analyze_patch_observability, INPUT_SCHEMA

    TRUE = np.array([0.60, 1.50, 2.00])                       # roll, pitch, yaw (deg) hidden misalignment
    NAMES = ("roll", "pitch", "yaw")
    rng = np.random.default_rng(7)

    def depth(x, y):                                          # m, positive down; planar facets
        return 40.0 + 0.04 * y + 0.02 * x - 10.0 * np.clip(1.0 - np.abs(y) / 40.0, 0.0, None)

    def rot(a):                                               # Rz(yaw) Ry(pitch) Rx(roll), as mbes_calibration
        r, p, yw = np.radians(a)
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    theta = np.radians(np.linspace(-60, 60, 81))
    beams = np.stack([np.zeros_like(theta), np.sin(theta), np.cos(theta)], 1)   # fwd, stbd, down
    lines = {"A": (0.0, 0.0), "B": (0.0, 180.0), "C": (60.0, 0.0)}             # track x, heading
    ys = np.arange(-70.0, 70.01, 1.0)

    def world_dirs(a, hdg):
        d = beams @ rot(a).T                                  # (nb, 3) in fwd/stbd/down
        h = np.radians(hdg)
        e = d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
        n = d[:, 0] * np.cos(h) - d[:, 1] * np.sin(h)
        return e, n, d[:, 2]

    # ---- "measured": ray-trace with the true angles to the true seabed, range noise 3 cm ----
    meas = []
    for ln, (xt, hdg) in lines.items():
        e, n, dd = world_dirs(TRUE, hdg)
        for yt in ys:
            r = np.full(len(theta), 45.0)
            for _ in range(40):
                r = depth(xt + r * e, yt + r * n) / dd
            r = r + rng.normal(0, 0.03, len(r))
            meas.append((ln, xt, yt, hdg, r))

    def georef(a):
        out = {k: [] for k in lines}
        for ln, xt, yt, hdg, r in meas:
            e, n, dd = world_dirs(a, hdg)
            out[ln].append(np.stack([xt + r * e, yt + r * n, -r * dd], 1))
        return {k: np.concatenate(v) for k, v in out.items()}

    PATCHES = [(cx, cy) for cy in (-24.0, 24.0) for cx in (-42.0, -12.0, 18.0, 46.0)]
    HALF = 6.0

    def payload(a):
        base = georef(a)
        jac = []
        for k in range(3):
            da = np.array(a, float); da[k] += 1e-3
            jac.append({ln: (v - base[ln]) / 1e-3 for ln, v in georef(da).items()})
        patches, digest = [], hashlib.sha256()
        for pi, (cx, cy) in enumerate(PATCHES):
            pts, jj, tags = [], [], []
            for ln in lines:
                xyz = base[ln]
                m = (np.abs(xyz[:, 0] - cx) <= HALF) & (np.abs(xyz[:, 1] - cy) <= HALF)
                pts.append(xyz[m]); tags += [ln] * int(m.sum())
                jj.append(np.stack([jac[k][ln][m] for k in range(3)], 2))     # (n, xyz, param)
            pts, jj = np.concatenate(pts), np.concatenate(jj)
            A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
            plane = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]              # starting plane (script)
            digest.update(pts.tobytes())
            patches.append({"id": f"patch {pi + 1}", "head_id": 1, "plane": plane.tolist(),
                            "points": [{"id": f"{tags[i]}{i}", "xyz_m": pts[i].tolist(), "sigma_m": 0.05,
                                        "jacobian_xyz": jj[i].tolist()} for i in range(len(pts))]})
        return {"schema": INPUT_SCHEMA, "mode": "free_patch_planes", "data_kind": "synthetic",
                "frame": "local_metric", "length_unit": "m", "head_id": 1,
                "source": {"id": "synthetic three-line patch test", "sha256": digest.hexdigest(),
                           "jacobian_description": "finite differences of a straight-ray georeferencing, "
                                                   "transducer roll/pitch/yaw in degrees"},
                "parameters": [{"name": n, "unit": "degree", "scale": 1.0} for n in NAMES],
                "patches": patches}, base

    # ---- Gauss-Newton passes; every step comes from the repo solver ----
    est = np.zeros(3)
    history, results = [est.copy()], []
    for it in range(4):
        p, _ = payload(est)
        res = analyze_patch_observability(p)
        assert res["observability_status"] == "full_rank_linear_model", res["observability_status"]
        results.append(res)
        est = est + np.array(res["parameter_step"])
        history.append(est.copy())
        n = res["point_count"]
        print(f"pass {it + 1}: step {np.round(res['parameter_step'], 4)}  est {np.round(est, 4)}  "
              f"rms before {np.sqrt(res['weighted_residual_sse_before'] / n) * 0.05:.3f} m"
              f"  after {np.sqrt(res['weighted_residual_sse_after'] / n) * 0.05:.3f} m"
              f"  cond {res['condition_number']:.1f}  points {n}")
    final, fin_pl = payload(est)
    fin = analyze_patch_observability(final)                  # residual at the solution
    cov = np.array(results[-1]["conditional_covariance"])
    sig = np.sqrt(np.diag(cov))
    npts = results[0]["point_count"]

    def rms_of(res):
        return np.sqrt(res["weighted_residual_sse_before"] / res["point_count"]) * 0.05

    rms = [rms_of(r) for r in results] + [rms_of(fin)]

    # ---- figure ----
    fig, status = fig_base(("Calibration_Pro", "MBES patch test"),
                           "geoview_common.mbes_patch_observability.analyze_patch_observability (SVD, free patch planes)"
                           " - every step below\nsynthetic 3-line survey, ray-traced soundings; georeferencing and"
                           " Jacobians by this script, 4 Gauss-Newton passes")
    a0 = fig.add_axes([0.045, 0.17, 0.19, 0.69]); style(a0)
    gx, gy = np.meshgrid(np.linspace(-80, 130, 211), np.linspace(-80, 80, 161))
    a0.imshow(depth(gx, gy), cmap=DEPTH, origin="lower", extent=[-80, 130, -80, 80], aspect="auto",
              vmin=28, vmax=46, interpolation="nearest")
    for ln, (xt, hdg) in lines.items():
        y0, y1 = (-70, 70) if hdg == 0 else (70, -70)
        a0.add_patch(FancyArrowPatch((xt + (3 if ln == "B" else -3 if ln == "A" else 0), y0),
                                     (xt + (3 if ln == "B" else -3 if ln == "A" else 0), y1),
                                     arrowstyle="-|>", mutation_scale=7, color=LINE_C[ln], lw=1.1))
        a0.text(xt + (9 if ln == "B" else -9 if ln == "A" else 6), (75 if hdg == 0 else -77), ln,
                color=LINE_C[ln], fontsize=6.5, ha="center", va="center", fontweight="bold")
    for i, (cx, cy) in enumerate(PATCHES):
        a0.add_patch(Rectangle((cx - HALF, cy - HALF), 2 * HALF, 2 * HALF, fc=(1, 1, 1, 0.45), ec=C["ink"], lw=0.7))
    a0.text(103, 3, "ridge", color=C["ink"], fontsize=5.5, ha="center", va="bottom")
    a0.axhline(0, color=C["mute"], lw=0.4, ls=(0, (2, 2)))
    a0.set_xlim(-80, 130); a0.set_ylim(-80, 80)
    a0.set_xticks([-50, 0, 50, 100]); a0.set_yticks([-50, 0, 50])
    a0.set_xlabel("east (m)", fontsize=6, labelpad=1)
    label(a0, "3 lines, 8 patches")

    def strip(xyz, axis, lo, hi):
        o = 1 - axis
        m = (xyz[:, o] >= lo) & (xyz[:, o] <= hi)
        sel = xyz[m]
        return sel[:, axis], -sel[:, 2] - depth(sel[:, 0], sel[:, 1])

    a1 = fig.add_axes([0.29, 0.17, 0.215, 0.69]); style(a1, grid=True)
    label(a1, "across track, y 22..26 m")
    a1.set_xlim(-70, 130); a1.set_ylim(-1.6, 1.6)
    a1.set_xlabel("east (m)", fontsize=6, labelpad=1)
    a1.set_ylabel("depth error vs true seabed (m)", fontsize=6, labelpad=1)
    a2 = fig.add_axes([0.545, 0.17, 0.2, 0.69]); style(a2, grid=True); a2.set_yticklabels([])
    label(a2, "along track, x 12..24 m")
    a2.set_xlim(-60, 60); a2.set_ylim(-1.6, 1.6)
    a2.set_xlabel("north (m)", fontsize=6, labelpad=1)
    sc1 = {ln: a1.plot([], [], ".", ms=1.6, color=LINE_C[ln], alpha=0.8)[0] for ln in lines}
    sc2 = {ln: a2.plot([], [], ".", ms=1.6, color=LINE_C[ln], alpha=0.8)[0] for ln in lines}
    for k, ln in enumerate(lines):
        a1.text(0.04 + 0.1 * k, 0.93, ln, transform=a1.transAxes, color=LINE_C[ln], fontsize=6.5,
                fontweight="bold")

    a3 = fig.add_axes([0.775, 0.17, 0.21, 0.69]); a3.axis("off")
    label(a3, "repo estimate (deg)")
    a3.text(0.0, 0.94, f"{'':6}{'est':>7}{'true':>7}", transform=a3.transAxes, fontsize=6.5, color=C["mute"])
    rows = [a3.text(0.0, 0.84 - 0.1 * k, "", transform=a3.transAxes, fontsize=6.5, color=C["ink"])
            for k in range(3)]
    info = a3.text(0.0, 0.47, "", transform=a3.transAxes, fontsize=6, color=C["mute"], va="top", linespacing=1.5)
    verdict = a3.text(0.0, -0.02, "", transform=a3.transAxes, fontsize=6.5, color=C["ok"], va="bottom",
                      linespacing=1.4)

    def draw(a, k_pass, note):
        xyz = georef(a)
        for ln in lines:
            sc1[ln].set_data(*strip(xyz[ln], 0, 22, 26))
            sc2[ln].set_data(*strip(xyz[ln], 1, 12, 24))
        for k in range(3):
            rows[k].set_text(f"{NAMES[k]:6}{a[k]:7.3f}{TRUE[k]:7.2f}")
        info.set_text(note)

    frames = []
    draw(history[0], 0, f"pass 0: no correction\n{npts:,} soundings\nin 8 patches\n\nrms misfit to\n"
                        f"patch planes {rms[0]:.3f} m")
    status.set_text("uncorrected soundings")
    frames += [grab(fig)] * 18
    for it in range(4):
        r = results[it]
        stp = r["parameter_step"]
        note = (f"pass {it + 1}/4, step\n" + " ".join(f"{s:+.3f}" for s in stp) +
                f"\ncondition {r['condition_number']:.1f}\n\nrms misfit to\npatch planes\n"
                f"{rms[it]:.3f} -> {rms[it + 1]:.3f} m")
        status.set_text(f"solver pass {it + 1}/4")
        nsub = 24 if it == 0 else 5
        for j in range(nsub):
            f = (j + 1) / nsub
            draw(history[it] * (1 - f) + history[it + 1] * f, it + 1, note)
            frames.append(grab(fig))
        frames += [frames[-1]] * (8 if it == 0 else 3)
    for k in range(3):
        rows[k].set_text(f"{NAMES[k]:6}{est[k]:7.3f}{TRUE[k]:7.2f}")
        rows[k].set_color(C["ok"])
    info.set_text("1-sigma (covariance)\n" + " ".join(f"{s:.3f}" for s in sig) +
                  f"\ncondition {results[-1]['condition_number']:.1f}"
                  f"\nrms misfit to planes\n{rms[0]:.3f} -> {rms[-1]:.3f} m")
    verdict.set_text(f"rank {fin['rank']}/3, {fin['review_status']}")
    status.set_text("solved after 4 passes")
    frames += [grab(fig)] * 6
    write_gif(frames, "calibration-patchtest.gif", hold=48)
    return est, sig, rms


# =========================================================================================
# 2. OffsetManager - frame change and reference point move
# =========================================================================================
def make_offsetmanager():
    from offsetmanager.models import SensorOffset
    from offsetmanager.axis_convention import detect_convention
    from offsetmanager.core import transform_frame, transform_reference_frame, calculate_distance

    # deck frame: +X fwd, +Y port, +Z up, origin at a deck mark amidships (all synthetic)
    S = [("GNSS 1 PORT", "GPS", -2.10, 1.20, 9.40), ("GNSS 2 STBD", "GPS", -2.10, -1.25, 9.40),
         ("MRU", "MRU", 1.60, 0.10, -1.85), ("Gyro", "Gyro", 1.35, 0.35, -1.80),
         ("MBES", "MBES Transducer", 4.20, -0.45, -3.60), ("SBP", "SBP Transducer", 2.80, 0.55, -3.55),
         ("USBL", "USBL", 0.90, -0.30, -3.95), ("Davit PORT", "Other", -11.40, 3.60, 1.10),
         ("Davit STBD", "Other", -11.40, -3.60, 1.10), ("Towpoint", "Towpoint", -15.20, 0.00, 0.55)]
    vcs = [SensorOffset(id=i + 1, sensor_name=n, sensor_type=t, x_offset=x, y_offset=y, z_offset=z)
           for i, (n, t, x, y, z) in enumerate(S)]
    mru = vcs[2]
    spec = dict(origin_x=mru.x_offset, origin_y=mru.y_offset, origin_z=mru.z_offset,
                roll_deg=0.0, pitch_deg=0.0, yaw_deg=-0.60, flip_x=1, flip_y=-1, flip_z=-1)
    v0 = detect_convention(vcs)
    crp = transform_frame(vcs, spec)
    v1 = detect_convention(crp)
    gref = transform_reference_frame(crp, "GNSS 1 PORT")
    lever = calculate_distance(gref[0], gref[4])
    print("deck  :", v0.z_axis, v0.y_axis, v0.outliers)
    print("CRP   :", v1.z_axis, v1.y_axis, v1.outliers)
    print("lever GNSS1->MBES", lever)
    for a, b in zip(crp, gref):
        print(f"  {a.sensor_name:12} CRP {a.x_offset:8.3f} {a.y_offset:8.3f} {a.z_offset:8.3f}"
              f"   GNSS1 {b.x_offset:8.3f} {b.y_offset:8.3f} {b.z_offset:8.3f}")

    fig, status = fig_base(("OffsetManager", "one survey, three frames"),
                           "detect_convention · transform_frame (FLIP @ R @ (old - origin))"
                           " · transform_reference_frame · calculate_distance\n"
                           "synthetic vessel and sensors; drawing and frame glide by this script,"
                           " every table value from the repo functions")
    # plan view (x fwd to the right, port up) and profile (z up)
    ap = fig.add_axes([0.04, 0.53, 0.43, 0.33]); style(ap); ap.set_xticks([]); ap.set_yticks([])
    az = fig.add_axes([0.04, 0.17, 0.43, 0.33]); style(az); az.set_yticks([])
    hull = [(-17, -4.2), (6, -4.2), (11, -2.5), (14, 0), (11, 2.5), (6, 4.2), (-17, 4.2)]
    ap.add_patch(Polygon(hull, closed=True, fc="#F6F8FA", ec=C["mute"], lw=0.8))
    az.add_patch(Polygon([(-17, 0), (-17, 3.0), (14, 3.0), (11.5, -2.2), (8, -4.0), (-15, -4.0)],
                         closed=True, fc="#F6F8FA", ec=C["mute"], lw=0.8))
    az.add_patch(Polygon([(-5, 3.0), (-5, 6.0), (2, 6.0), (2, 3.0)], closed=True, fc="#F6F8FA",
                         ec=C["mute"], lw=0.6))
    az.plot([-2.1, -2.1], [6.0, 9.4], color=C["mute"], lw=0.8)
    az.axhline(-0.4, color="#80CCFF", lw=0.8)
    az.text(13.4, -0.15, "WL", color="#218BFF", fontsize=5.5)
    for ax, (lo, hi) in ((ap, (-6.0, 6.0)), (az, (-5.0, 10.6))):
        ax.set_xlim(-19, 16); ax.set_ylim(lo, hi)
    az.set_xticks([-15, -10, -5, 0, 5, 10]); az.tick_params(labelsize=5.5)
    az.set_xlabel("deck frame X (m)", fontsize=6, labelpad=1)
    label(ap, "plan (port up)"); label(az, "profile")
    for o in vcs:
        col = C["acc"] if o.sensor_type in ("GPS",) else C["bad"] if "Transducer" in o.sensor_type \
            or o.sensor_type == "USBL" else C["ink"]
        ap.plot(o.x_offset, o.y_offset, "o", ms=2.6, color=col)
        az.plot(o.x_offset, o.z_offset, "o", ms=2.6, color=col)
    for nm, dx, dy in (("GNSS 1", -1.0, 0.8), ("MBES", 0.4, -1.4), ("MRU", -0.4, -2.4), ("Davit", 0.4, 0.5)):
        o = next(s for s in vcs if s.sensor_name.startswith(nm))
        ap.text(o.x_offset + dx, o.y_offset + dy, nm, fontsize=5.5, color=C["mute"])
    for nm, dx, dz in (("GNSS", 0.4, -0.2), ("MBES", 0.4, -0.9), ("MRU", -2.8, 0.5)):
        o = next(s for s in vcs if s.sensor_name.startswith(nm))
        az.text(o.x_offset + dx, o.z_offset + dz, nm, fontsize=5.5, color=C["mute"])

    # frame triad (drawn in deck coordinates)
    tri = {}
    for ax, k in ((ap, "p"), (az, "z")):
        tri[k] = [ax.annotate("", xy=(0, 0), xytext=(0, 0),
                              arrowprops=dict(arrowstyle="-|>", color=C["acc"], lw=1.2, mutation_scale=6))
                  for _ in range(2)]
        tri[k + "t"] = [ax.text(0, 0, "", color=C["acc"], fontsize=5.5, fontweight="bold") for _ in range(2)]
        tri[k + "o"] = ax.plot([], [], "s", ms=3.5, color=C["acc"])[0]

    def set_triad(origin, yaw, ys, zs):
        ox, oy, oz = origin
        L = 4.0
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        xa = np.array([c, -s]) * L                            # X axis in the deck plan
        ya = np.array([s, c]) * L * 0.6 * ys                  # Y axis (port +1 / stbd -1)
        for arr, txt, vec, nm in ((tri["p"][0], tri["pt"][0], xa, "X"), (tri["p"][1], tri["pt"][1], ya, "Y")):
            arr.xy = (ox + vec[0], oy + vec[1]); arr.set_position((ox, oy))
            txt.set_position((ox + vec[0] * 1.12 + 0.2, oy + vec[1] * 1.12)); txt.set_text(nm)
        tri["po"].set_data([ox], [oy])
        for arr, txt, vec, nm in ((tri["z"][0], tri["zt"][0], (L, 0), "X"), (tri["z"][1], tri["zt"][1], (0, 2.6 * zs), "Z")):
            arr.xy = (ox + vec[0], oz + vec[1]); arr.set_position((ox, oz))
            txt.set_position((ox + vec[0] * 1.1 + 0.25, oz + vec[1] * 1.1)); txt.set_text(nm)
        tri["zo"].set_data([ox], [oz])

    # table
    at = fig.add_axes([0.51, 0.17, 0.475, 0.69]); at.axis("off")
    ttl = at.text(0.0, 1.0, "", transform=at.transAxes, fontsize=6.5, color=C["mute"], va="bottom")
    at.text(0.0, 0.93, f"{'sensor':12}{'X':>8}{'Y':>8}{'Z':>8}", transform=at.transAxes, fontsize=6.3,
            color=C["mute"])
    cells = [[at.text(0.0 if j == 0 else 0.315 + 0.155 * (j - 1), 0.85 - 0.068 * i, "", transform=at.transAxes,
                      fontsize=6.3, color=C["ink"], ha="left" if j == 0 else "right")
              for j in range(4)] for i in range(len(vcs))]
    for i in range(len(vcs)):
        for j in range(1, 4):
            cells[i][j].set_position((0.33 + 0.155 * j, 0.85 - 0.068 * i))
    verd = at.text(0.0, 0.11, "", transform=at.transAxes, fontsize=6.2, color=C["ok"], va="top", linespacing=1.45)

    def fill(rows, changed=None):
        for i, o in enumerate(rows):
            vals = (o.sensor_name, o.x_offset, o.y_offset, o.z_offset)
            for j in range(4):
                cells[i][j].set_text(vals[0][:12] if j == 0 else f"{vals[j] + 0.0:8.3f}".replace("-0.000", " 0.000"))
                hot = changed is not None and j > 0 and changed[i][j - 1]
                cells[i][j].set_color(C["warn"] if hot else C["ink"])

    def verdict_text(v):
        z = {"up": "+Z=Up", "down": "+Z=Down"}.get(v.z_axis, v.z_axis)
        y = {"port": "+Y=Port", "stbd": "+Y=Stbd"}.get(v.y_axis, v.y_axis)
        return f"detect_convention: {z}, {y}, outliers {len(v.outliers)}"

    frames = []
    deck0 = (0.0, 0.0, 0.0)
    set_triad(deck0, 0.0, 1, 1)
    ttl.set_text("deck frame, origin = deck mark")
    fill(vcs)
    status.set_text("as measured")
    verd.set_text("")
    frames += [grab(fig)] * 8
    verd.set_text(verdict_text(v0) + "\nhull sensors below zero, GNSS above,\nPORT-named sensors at +Y")
    status.set_text("detect_convention")
    frames += [grab(fig)] * 12

    # glide origin to the MRU and yaw -0.6 deg; table recomputed by transform_frame on the partial spec
    target = (mru.x_offset, mru.y_offset, mru.z_offset)
    ttl.set_text("transform_frame: origin MRU, yaw -0.60 deg")
    verd.set_text("")
    n = 14
    for j in range(n):
        f = (j + 1) / n
        o = tuple(deck0[k] * (1 - f) + target[k] * f for k in range(3))
        part = dict(origin_x=o[0], origin_y=o[1], origin_z=o[2], roll_deg=0.0, pitch_deg=0.0,
                    yaw_deg=-0.60 * f, flip_x=1, flip_y=1, flip_z=1)
        set_triad(o, -0.60 * f * 6, 1, 1)                      # yaw drawn x6 so it is visible
        fill(transform_frame(vcs, part))
        status.set_text(f"move origin  {f * 100:3.0f} %")
        frames.append(grab(fig))
    frames += [frames[-1]] * 4
    pre = transform_frame(vcs, dict(spec, flip_y=1, flip_z=1))
    for step_name, ys, zs in (("flip Y (port -> stbd)", -1, 1), ("flip Z (up -> down)", -1, -1)):
        sp = dict(spec, flip_y=ys, flip_z=zs)
        rows = transform_frame(vcs, sp)
        set_triad(target, -3.6, ys, zs)
        ch = [[abs(a.x_offset - b.x_offset) > 1e-9, abs(a.y_offset - b.y_offset) > 1e-9,
               abs(a.z_offset - b.z_offset) > 1e-9] for a, b in zip(pre, rows)]
        fill(rows, ch)
        status.set_text(step_name)
        frames += [grab(fig)] * 9
        pre = rows
    fill(crp)
    ttl.set_text("CRP frame, origin = MRU (+X fwd, +Y stbd, +Z down)")
    verd.set_text(verdict_text(v1) + "\nsame geometry, opposite signs: Y and Z flipped")
    status.set_text("detect_convention again")
    frames += [grab(fig)] * 14

    # move the zero to GNSS 1
    g = vcs[0]
    for j in range(8):
        f = (j + 1) / 8
        o = tuple(target[k] * (1 - f) + (g.x_offset, g.y_offset, g.z_offset)[k] * f for k in range(3))
        set_triad(o, -3.6, -1, -1)
        status.set_text("transform_reference_frame")
        frames.append(grab(fig))
    ch = [[True, True, True] for _ in gref]
    fill(gref, ch)
    ttl.set_text("same CRP axes, zero moved to GNSS 1 PORT")
    verd.set_text(f"calculate_distance GNSS 1 -> MBES\nhorizontal {lever.horizontal_distance:.3f} m, "
                  f"vertical {lever.vertical_distance:.3f} m,\n3D lever arm {lever.total_distance:.3f} m")
    verd.set_color(C["acc"])
    status.set_text(f"lever arm GNSS 1 -> MBES {lever.total_distance:.3f} m")
    frames += [grab(fig)] * 8
    fill(gref)
    frames += [grab(fig)] * 4
    write_gif(frames, "offsetmanager-frame.gif", hold=48)
    return v0, v1, lever


if WHICH in ("both", "calib"):
    make_calibration()
if WHICH in ("both", "offset"):
    make_offsetmanager()
