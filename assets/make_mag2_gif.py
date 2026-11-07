"""Two profile animations for the magnetometer chain, same style as make_visuals.py (helpers copied below).

  magproc-leveling.gif  MagProc  magproc/core/corrections/leveling.py::tieline_leveling (Oasis XLEVEL style:
                                   _find_intersections -> mis-tie per crossing -> _interpolate_corrections along line)
                                   (micro_leveling is run for the printed numbers only, not drawn: on this
                                    synthetic set it raised the error against the truth, see the report)
                                 magproc/core/gridding/minimum_curvature.py::grid_minimum_curvature
                                 magproc/core/data_model.py::MagLine.from_arrays, MagGrid
  magprep-clean.gif     MagPrep  backend/parsers.py -> _shared/mag_parsers (MagPrep's parser package):
                                   parse_nmea_gps (bit-flip repair, field validation)
                                   export_clean_gps (neighbour outlier rejection, field-length check, time/position
                                     continuity filter)
                                   parse_explorer (Explorer records, Cycling off/on segments, garbage lines)
                                   auto_split (segment-based line split), merge_gps_to_mag (time match),
                                   export_explorer_lines_zip (per-line .MAG + .GPS files)

Inputs are synthetic and generated below. MagProc: 21 east-west survey lines 20 m apart and 4 north-south tie lines
over an invented residual field (smooth geology + two dipoles); each survey line carries an invented level error
(offset + drift + a 380-460 m wiggle). MagPrep: a 1 Hz GPGGA stream and a 2 Hz Explorer stream for three lines with
turns, with invented bit-flip corruption, one truncated sentence, five position jumps and one garbage line.
No survey data, line, vessel or project names are used. Positions in the images are local metres.

What the script does itself (stated in the footers): drawing the transition between stages, the RMS against the
synthetic truth (MagProc), and marking the characters it corrupted itself (MagPrep, red in the text panel).

usage:  python make_mag2_gif.py <out_dir> [magproc_repo] [shared_repo] [proc,prep]
needs:  numpy, scipy, matplotlib, pillow, pyyaml (MagProc import chain)
"""
import sys, types
sys.dont_write_bytecode = True                     # import from the repos without leaving __pycache__ there
import io, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
MAGPROC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/Processing/MagProc")
SHARED = Path(sys.argv[3] if len(sys.argv) > 3 else "/home/user/repos/_shared")
LAST_DIR = OUT

# ---- style: copied from make_visuals.py ---------------------------------------------------
C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}")


def pkg(name, path=None):
    m = types.ModuleType(name); m.__path__ = [str(path)] if path else []; sys.modules[name] = m


def fig_base(title, foot):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=C["bg"])
    prog, what = title
    fig.text(0.015, 0.935, prog, color=C["ink"], fontsize=9, fontweight="bold")
    fig.text(0.015 + 0.0128 * len(prog) + 0.008, 0.935, "·  " + what, color=C["mute"], fontsize=9)
    foot = [foot] if isinstance(foot, str) else foot          # one or two footer lines
    for i, f in enumerate(foot):
        fig.text(0.015, 0.025 + 0.042 * (len(foot) - 1 - i), f, color=C["mute"], fontsize=6)
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
    return ax.text(0.0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom",
                   color=color or C["mute"], fontsize=6.5)


def grab(fig):
    fig.canvas.draw()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")


def write_gif(frames, name, hold=30, mid=None):
    pick = frames[::max(1, len(frames) // 8)] + [frames[-1]]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(pick)))
    for i, f in enumerate(pick):
        strip.paste(f, (0, i * f.height))
    q = strip.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    rgb = q.getpalette()[:200 * 3]
    for hx in C.values():
        rgb += [int(hx[i:i + 2], 16) for i in (1, 3, 5)]
    ref = Image.new("P", (1, 1)); ref.putpalette(rgb + [0] * (768 - len(rgb)))
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / FPS)] * len(pal); dur[-1] = int(1000 / FPS) * hold
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=dur, loop=0,
                optimize=True, disposal=1)
    frames[-1].save(LAST_DIR / (Path(name).stem + "-last.png"))
    frames[mid if mid is not None else len(frames) // 2].save(LAST_DIR / (Path(name).stem + "-mid.png"))
    print(f"{name}: {len(pal)} frames, {(OUT / name).stat().st_size / 1024:,.0f} KB")
    plt.close("all")


MAGCM = LinearSegmentedColormap.from_list("mag", ["#57606A", "#AFB8C1", "#F6F8FA", "#FFFFFF",
                                                  "#F5E0A3", "#D4A72C", "#7D4E00"])   # same as maguxo-euler
MAGCM.set_bad(C["bg"])


# =========================================================================================
# 1. MagProc - tie-line levelling, micro-levelling, minimum-curvature grid
# =========================================================================================
def magproc():
    sys.path[:0] = [str(MAGPROC)]
    pkg("magproc", MAGPROC / "magproc")
    for sub in ("corrections", "filters"):                          # skip those package __init__s
        pkg(f"magproc.core.{sub}", MAGPROC / f"magproc/core/{sub}")
    from magproc.core.data_model import MagLine
    from magproc.core.corrections import leveling as LV
    from magproc.core.gridding.minimum_curvature import grid_minimum_curvature

    rng = np.random.default_rng(11)
    DX, LX = 2.0, 1000.0                               # along-line sampling and line length (m)
    YS = np.arange(0.0, 401.0, 20.0)                   # 21 survey lines, 20 m apart
    XT = np.array([50.0, 350.0, 650.0, 950.0])         # 4 tie lines
    xs = np.arange(0.0, LX + 0.1, DX)

    def truth(x, y):                                   # residual field (nT): geology + two dipoles
        g = 6 * np.sin(2 * np.pi * x / 1400 + 0.4) * np.cos(2 * np.pi * y / 900) + 0.004 * x - 2

        def dip(x0, y0, h, a):
            r2 = (x - x0) ** 2 + (y - y0) ** 2 + h * h
            return a * h ** 3 * (2 * h * h - ((x - x0) ** 2 + (y - y0) ** 2)) / r2 ** 2.5 \
                - 0.35 * a * h ** 3 * (y - y0) * h / r2 ** 2.5 * 3
        return g + dip(560, 230, 26, 34) + dip(240, 110, 18, 9)

    lines, tru = {}, {}
    for i, y in enumerate(YS):
        off = rng.uniform(-4.5, 4.5); drift = rng.uniform(-2.5, 2.5)
        lam = rng.uniform(380, 460); ph = rng.uniform(0, 2 * np.pi)
        err = off + drift * (xs / LX - 0.5) + 0.5 * np.sin(2 * np.pi * xs / lam + ph)
        z0 = truth(xs, np.full_like(xs, y))
        f = z0 + err + rng.normal(0, 0.04, xs.size)
        lines[i + 1] = MagLine.from_arrays(xs / DX * 500.0, f, x=xs, y=np.full_like(xs, y), line_id=i + 1)
        tru[i + 1] = z0
    yt = np.arange(-10.0, 410.1, DX)
    for j, x0 in enumerate(XT):
        z0 = truth(np.full_like(yt, x0), yt)
        lines[100 + j] = MagLine.from_arrays(yt / DX * 500.0, z0 + rng.normal(0, 0.04, yt.size),
                                             x=np.full_like(yt, x0), y=yt, line_id=100 + j, line_type="tie")
    sids = [k for k in lines if lines[k].line_type == "survey"]

    survey = {k: lines[k] for k in sids}; ties = {k: lines[k] for k in lines if k >= 100}
    xp0 = LV._find_intersections(survey, ties, 50.0)                 # MagProc, before
    lev1, xp_used = LV.tieline_leveling(lines, tolerance_m=50.0)      # MagProc
    xp1 = LV._find_intersections({k: lev1[k] for k in sids}, ties, 50.0)
    # micro_leveling is run for the report only, not drawn: on this synthetic set it raised the error against
    # the truth (see the returned 'micro' numbers), so the animation stops at tie-line levelling + gridding.
    micro = {}
    for cs in (10.0, 20.0):
        lev2 = LV.micro_leveling(lev1, cell_size=cs, azimuth_deg=90.0, filter_width_deg=30.0)   # MagProc
        micro[cs] = np.array([lev2[k].field for k in sids])

    def img(d):                                       # one row per survey line, for display
        return np.array([d[k].field for k in sids])
    I0, I1 = img(lines), img(lev1)
    IT = np.array([tru[k] for k in sids])
    rms = lambda I: float(np.sqrt(np.mean((I - IT) ** 2)))           # against the synthetic truth (this script)
    r0, r1 = rms(I0), rms(I1)
    micro = {cs: (rms(I), float(I.max())) for cs, I in micro.items()}

    xa = np.concatenate([lev1[k].x for k in lines]); ya = np.concatenate([lev1[k].y for k in lines])
    za = np.concatenate([lev1[k].field for k in lines])
    G = grid_minimum_curvature(xa, ya, za, cell_size=5.0, blanking_distance=30.0)          # MagProc
    Gd = G.data

    foot = ["MagProc _find_intersections · tieline_leveling (XLEVEL style) · grid_minimum_curvature (5 m cells)",
            "synthetic lines and level errors  ·  stage blending and RMS against the synthetic truth drawn here"]
    fig, status = fig_base(("MagProc", "tie-line levelling and gridding"), foot)
    a1 = fig.add_axes([0.065, 0.175, 0.50, 0.68]); style(a1)
    a2 = fig.add_axes([0.655, 0.585, 0.33, 0.27]); style(a2, grid=True)
    a3 = fig.add_axes([0.655, 0.175, 0.33, 0.27]); style(a3, grid=True)
    lim = float(np.percentile(np.abs(IT), 99.5)) * 0.9
    ext = [-DX / 2, LX + DX / 2, YS[0] - 10, YS[-1] + 10]
    shown = np.full_like(I0, np.nan)
    im = a1.imshow(shown, origin="lower", aspect="auto", cmap=MAGCM, vmin=-lim, vmax=lim, extent=ext,
                   interpolation="nearest")
    a1.set_xlim(-10, LX + 10); a1.set_ylim(-14, 414)
    a1.xaxis.set_major_formatter(COMMA)
    a1.set_xlabel("east (m)", fontsize=6, labelpad=1); a1.set_ylabel("north (m)", fontsize=6, labelpad=1)
    lab1 = label(a1, "residual field (nT) · 21 survey lines, 20 m apart")
    tie_ln = [a1.plot([], [], color=C["ink"], lw=0.8)[0] for _ in XT]
    xsc = a1.scatter([], [], s=9, facecolors="none", edgecolors=C["bad"], lw=0.6, zorder=5)
    XC = 500.0                                         # cross-section between ties 2 and 3
    ic = int(round(XC / DX))
    cut = a1.axvline(XC, color=C["acc"], lw=0.6, ls=(0, (2, 2)), alpha=0)
    a2.set_xlim(-10, 410); a2.set_ylim(IT[:, ic].min() - 7, IT[:, ic].max() + 7)
    a2.set_xlabel("north (m)", fontsize=6, labelpad=1)
    lab2 = label(a2, f"section at east {XC:,.0f} m (nT) · truth grey")
    a2.plot(YS, IT[:, ic], color=C["mute"], lw=1.6, alpha=0.45)
    sec, = a2.plot([], [], color=C["acc"], lw=0.8, marker="o", ms=1.8)
    nx_ = len(xp0)
    a3.set_xlim(-1, nx_); a3.set_ylim(-7.5, 7.5)
    a3.set_xticks([])
    a3.set_xlabel(f"{nx_} line / tie crossings", fontsize=6, labelpad=1)
    lab3 = label(a3, "mis-tie at crossings (nT)")
    a3.axhline(0, color=C["mute"], lw=0.5)
    stem = a3.vlines([], [], [], color=C["bad"], lw=1.0)
    frames = []

    def put_mistie(xps, color, n=None):
        v = np.array([p.mismatch for p in xps])[:n]
        stem.set_segments([[(i, 0), (i, m)] for i, m in enumerate(v)])
        stem.set_color(color)
        return float(np.sqrt(np.mean(v ** 2))) if len(v) else 0.0

    # --- stage 0: lines come in, ties, crossings --------------------------------------------
    for k in list(range(1, len(sids), 3)) + [len(sids)]:
        shown[:k] = I0[:k]; im.set_data(shown)
        sec.set_data(YS[:k], I0[:k, ic])
        status.set_text(f"{k:d} / {len(sids)} survey lines in"); status.set_color(C["acc"])
        frames.append(grab(fig))
    shown[:] = I0; im.set_data(shown); sec.set_data(YS, I0[:, ic]); cut.set_alpha(0.8)
    for j, x0 in enumerate(XT):
        tie_ln[j].set_data([x0, x0], [yt[0], yt[-1]])
        status.set_text(f"tie line {j + 1} of {len(XT)}")
        frames.append(grab(fig))
    for n in np.linspace(nx_ / 6, nx_, 6).astype(int):
        xsc.set_offsets(np.c_[[p.x for p in xp0[:n]], [p.y for p in xp0[:n]]])
        rm = put_mistie(xp0, C["bad"], n)
        status.set_text(f"{n} crossings · RMS mis-tie {rm:.2f} nT"); status.set_color(C["bad"])
        frames.append(grab(fig))
    rm0 = put_mistie(xp0, C["bad"])
    lab3.set_text(f"mis-tie at crossings (nT) · RMS {rm0:.2f}")
    lab1.set_text(f"raw lines · RMS error vs truth {r0:.2f} nT")
    frames += [grab(fig)] * 5

    # --- stage 1: tie-line levelling ---------------------------------------------------------
    mt0 = np.array([p.mismatch for p in xp0]); mt1 = np.array([p.mismatch for p in xp1])
    for w in np.linspace(0.1, 1.0, 10):
        I = I0 + (I1 - I0) * w
        im.set_data(I); sec.set_data(YS, I[:, ic])
        v = mt0 + (mt1 - mt0) * w
        stem.set_segments([[(i, 0), (i, m)] for i, m in enumerate(v)])
        status.set_text("tieline_leveling: mis-tie spread along lines"); status.set_color(C["warn"])
        frames.append(grab(fig))
    a3.vlines(np.arange(nx_), 0, mt0, color=C["mute"], lw=1.0, alpha=0.3, zorder=1)   # before, for reference
    rm1 = put_mistie(xp1, C["ok"])
    lab3.set_text(f"mis-tie (nT) · RMS {rm0:.2f} -> {rm1:.2f}")
    lab1.set_text(f"after tie-line levelling · RMS error vs truth {r1:.2f} nT")
    status.set_text(f"crossings closed · error {r0:.2f} -> {r1:.2f} nT"); status.set_color(C["ok"])
    frames += [grab(fig)] * 6
    n_mid = len(frames) - 3

    # --- stage 3: minimum-curvature grid -----------------------------------------------------
    gext = [G.x_origin, G.x_max, G.y_origin, G.y_max]
    for t_ in tie_ln:
        t_.set_alpha(0.35)
    xsc.set_offsets(np.empty((0, 2)))
    im2 = a1.imshow(Gd, origin="upper", aspect="auto", cmap=MAGCM, vmin=-lim, vmax=lim, extent=gext,
                    interpolation="bilinear", alpha=0.0, zorder=2)
    for t_ in tie_ln:
        t_.set_zorder(3)
    cut.set_zorder(3)
    for a_ in np.linspace(0.25, 1.0, 4):
        im2.set_alpha(a_)
        status.set_text("grid_minimum_curvature: 5 m cells"); status.set_color(C["acc"])
        frames.append(grab(fig))
    a1.contour(np.linspace(G.x_origin + 2.5, G.x_max - 2.5, G.nx), np.linspace(G.y_max - 2.5, G.y_origin + 2.5, G.ny),
               Gd, levels=[-12, -6, 6, 12, 18], colors=C["ink"], linewidths=0.35, alpha=0.6, zorder=3)
    lab1.set_text(f"levelled lines gridded · {G.nx:,} x {G.ny:,} cells · contours 6 nT")
    status.set_text(f"mis-tie {rm0:.2f} -> {rm1:.2f} nT · error {r0:.2f} -> {r1:.2f}"); status.set_color(C["ink"])
    frames.append(grab(fig))
    if "--peek" in sys.argv:
        frames[22].save(OUT / "_peek_proc.png")
    write_gif(frames, "magproc-leveling.gif", hold=50, mid=n_mid)
    return dict(n_cross=nx_, mistie=(rm0, rm1), err=(r0, r1), peak=(float(IT.max()), float(I1.max())),
                micro_rms_peak=micro, grid=(G.nx, G.ny))

# =========================================================================================
# 2. MagPrep - raw GPGGA + Explorer streams: repair, clean, merge, split into lines
# =========================================================================================
FLIP = {"0": " ", "1": "!", "2": '"', "3": "#", "5": "%", "7": "'", "8": "(", "9": ")"}   # inverse of _BIT4_FLIP


def nmea_ck(body):
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def magprep():
    sys.path[:0] = [str(SHARED)]
    from mag_parsers import (parse_nmea_gps, export_clean_gps, parse_explorer, auto_split, merge_gps_to_mag,
                             export_explorer_lines_zip)

    rng = np.random.default_rng(7)
    LAT0, LON0 = 34 + 24.30 / 60, 128 + 30.10 / 60       # open water; the image shows local metres only
    MLAT = 111_320.0; MLON = MLAT * np.cos(np.radians(LAT0))
    V, LL, R = 2.2, 900.0, 50.0                          # speed (m/s), line length (m), turn radius (m)
    T0 = 2 * 3600 + 10 * 60                              # 02:10:00 UTC

    # --- vessel path: three east-west lines joined by half-circle turns -------------------------
    segs = []                                            # (kind, duration s, fn(tau) -> e, n)
    for i in range(3):
        n0 = i * 2 * R; d = 1 if i % 2 == 0 else -1
        e0 = 0.0 if d > 0 else LL
        segs.append(("line", LL / V, (lambda e0, d, n0: lambda s: (e0 + d * V * s, n0))(e0, d, n0)))
        if i < 2:
            ec = LL if d > 0 else 0.0
            segs.append(("turn", np.pi * R / V, (lambda ec, d, n0: lambda s: (
                ec + d * R * np.sin(V * s / R), n0 + R - R * np.cos(V * s / R)))(ec, d, n0)))
    bounds = np.cumsum([0] + [s[1] for s in segs])
    TT = bounds[-1]

    def pos(t):
        k = min(int(np.searchsorted(bounds, t, side="right")) - 1, len(segs) - 1)
        return segs[k][2](t - bounds[k]), segs[k][0]

    def field(e, n, t):                                  # invented residual + target + slow drift
        def dip(e0, n0, h, a):
            r2 = (e - e0) ** 2 + (n - n0) ** 2 + h * h
            return a * h ** 3 * (2 * h * h - (r2 - h * h)) / r2 ** 2.5
        return (49_512.0 + 0.004 * e - 0.004 * n + 0.8 * np.sin(t / 400.0)
                + dip(300, 8, 12, 6) + dip(640, 196, 14, 7))

    hms = lambda t: f"{int(t // 3600):02d}{int(t % 3600 // 60):02d}{t % 60:05.2f}"

    # --- GPGGA 1 Hz with planted damage ---------------------------------------------------------
    tg = np.arange(0, int(TT))
    jumps = {150: (0, 210), 520: (160, 0), 780: (0, -190), 1010: (-170, 120), 1240: (0, 230)}   # metres
    plan = {60: ("flip", (11, 19)), 205: ("flip", (21, 33)), 330: ("star", 28), 455: ("amp", 21),
            610: ("flip", (34,)), 700: ("trunc", 27), 860: ("star", 41), 935: ("flip", (13, 24, 36)),
            1100: ("alpha", 35), 1180: ("flip", (22,)), 1300: ("flip", (16, 31))}
    sentences, marks, gps_en = [], [], []
    for t in tg:
        (e, n), _ = pos(float(t))
        je, jn = jumps.get(int(t), (0, 0))
        e_, n_ = e + je, n + jn
        lat = LAT0 + n_ / MLAT; lon = LON0 + e_ / MLON
        lat_s = f"{int(lat):02d}{(lat - int(lat)) * 60:08.5f}"; lon_s = f"{int(lon):03d}{(lon - int(lon)) * 60:08.5f}"
        body = (f"GPGGA,{hms(T0 + t)},{lat_s},N,{lon_s},E,2,{rng.integers(12, 15)},0.8,"
                f"{12.3 + rng.normal(0, 0.05):.1f},M,25.1,M,1.0,0123")
        s = "$" + body + "*" + nmea_ck(body)
        bad = []
        kind = plan.get(int(t), (None, None))
        if kind[0] == "flip":
            ch = list(s)
            for p_ in kind[1]:
                while ch[p_] not in FLIP:
                    p_ += 1
                ch[p_] = FLIP[ch[p_]]; bad.append(p_)
            s = "".join(ch)
        elif kind[0] in ("star", "amp"):
            p_ = kind[1]
            want = "," if kind[0] == "star" else "."
            while s[p_] != want:
                p_ += 1
            s = s[:p_] + ("*" if kind[0] == "star" else "&") + s[p_ + 1:]; bad.append(p_)
        elif kind[0] == "trunc":
            s = s[:kind[1]]; bad = list(range(len(s) - 1, len(s)))
        elif kind[0] == "alpha":
            p_ = kind[1]
            s = s[:p_] + "A" + s[p_ + 1:]; bad.append(p_)
        sentences.append(s); marks.append(bad); gps_en.append((e_, n_))
    gps_en = np.array(gps_en)
    gtxt = "\n".join(sentences) + "\n"

    # --- Explorer 2 Hz, logging only on lines --------------------------------------------------
    mtxt, mrec_t, mrec_en = ["Cycle at 2Hz"], [], []
    garbage_at = None
    for k, (kind, dur, fn) in enumerate(segs):
        if kind != "line":
            continue
        t0 = bounds[k]
        if k > 0:
            mtxt.append("Cycle at 2Hz")
        for j, tt in enumerate(np.arange(np.ceil(t0 * 2) / 2, t0 + dur, 0.5)):
            (e, n), _ = pos(tt)
            ts = T0 + tt
            f_ = field(e, n, tt) + rng.normal(0, 0.02)
            mtxt.append(f"*26.134/{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:04.1f} "
                        f"F:{f_:010.3f} S:{rng.integers(130, 141)} D:+0{rng.normal(41, 0.3):.1f}m L0 0500ms Q:99")
            mrec_t.append(tt); mrec_en.append((e, n))
            if k == 2 and j == 300:
                mtxt.append("¿æ~F:04Ø§1." + "ÿ" * 3); garbage_at = tt
        mtxt.append("Cycling off")
    mtxt = "\n".join(mtxt) + "\n"

    # --- the program ----------------------------------------------------------------------------
    g = parse_nmea_gps(gtxt)                                            # repair + validate
    clean = export_clean_gps(g["valid"])                                # outlier / field / continuity filter
    gc = parse_nmea_gps(clean.decode("utf-8"))["valid"]
    m = parse_explorer(mtxt)                                            # records, segments, garbage
    lines = auto_split(m["data"], file_format="explorer", segments=m["segments"])
    mg = merge_gps_to_mag(m["data"], gc, tolerance_ms=500)
    zb = export_explorer_lines_zip(lines, gps_records=gc)
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        outs = [(nm, len(zf.read(nm).decode().strip().splitlines())) for nm in sorted(zf.namelist())]

    rej_ln = {r["lineNum"] for r in g["rejected"]}
    rep_ln = {r["lineNum"] for r in g["valid"] if r.get("isRepaired")}
    kept_t = {round(r["timeMs"] / 1000.0) for r in gc}
    valid_t = {round(r["timeMs"] / 1000.0) for r in g["valid"]}
    removed_t = sorted(valid_t - kept_t)
    st = lambda i: ("rejected" if i + 1 in rej_ln else "repaired" if i + 1 in rep_ln else None)

    # --- figure ---------------------------------------------------------------------------------
    foot = ["MagPrep mag_parsers: parse_nmea_gps · export_clean_gps · parse_explorer · auto_split",
            "merge_gps_to_mag · export_explorer_lines_zip  ·  synthetic streams, planted damage  ·  local metres"]
    fig, status = fig_base(("MagPrep", "raw streams to line files"), foot)
    at1 = fig.add_axes([0.015, 0.44, 0.465, 0.42]); at1.axis("off")
    at2 = fig.add_axes([0.015, 0.14, 0.465, 0.22]); at2.axis("off")
    am = fig.add_axes([0.555, 0.475, 0.43, 0.385]); style(am, grid=True)
    ap = fig.add_axes([0.555, 0.15, 0.43, 0.18]); style(ap, grid=True)
    lt1 = label(at1, "GPGGA 1 Hz · latest, then repaired / rejected")
    lt2 = label(at2, "Explorer MAG 2 Hz · latest, then events")
    NCH = 47
    FS_ = 5.4

    def rows(ax, n):
        out = []
        for r in range(n):
            y = 0.90 - r * (0.84 / max(n - 1, 1))
            out.append((ax.text(0.0, y, "", fontsize=FS_, color=C["ink"], transform=ax.transAxes, va="center"),
                        ax.text(0.0, y, "", fontsize=FS_, color=C["bad"], transform=ax.transAxes, va="center",
                                fontweight="bold"),
                        ax.text(1.0, y, "", fontsize=FS_, transform=ax.transAxes, va="center", ha="right")))
        return out
    R1 = rows(at1, 6); R2 = rows(at2, 3)

    def put(row, s, bad=(), tag="", tcol=C["mute"], col=C["ink"]):
        s = s[:NCH] + ("…" if len(s) > NCH else "")
        base = "".join(" " if i in bad else c for i, c in enumerate(s))
        over = "".join(c if i in bad else " " for i, c in enumerate(s))
        row[0].set_text(base); row[0].set_color(col); row[1].set_text(over)
        row[2].set_text(tag); row[2].set_color(tcol)

    am.set_xlim(-120, 1020); am.set_ylim(-260, 470); am.set_aspect("auto")
    am.xaxis.set_major_formatter(COMMA)
    am.set_xlabel("east (m)", fontsize=6, labelpad=1)
    lmap = label(am, "GPS fixes · repaired ringed")
    gdots = am.scatter([], [], s=1.2, color=C["mute"], lw=0, zorder=2)
    grep_ = am.scatter([], [], s=14, facecolors="none", edgecolors=C["warn"], lw=0.7, zorder=4)
    gout = am.scatter([], [], s=22, facecolors="none", edgecolors=C["bad"], lw=0.8, zorder=5)
    boat, = am.plot([], [], "o", ms=3.5, color=C["acc"], zorder=6)
    mt_ = np.array(mrec_t); mf = np.array([r["field"] for r in m["data"]])
    rec_t = np.array([(r["timeMs"] / 1000.0) - T0 for r in m["data"]])
    ap.set_xlim(0, TT / 60); ap.set_ylim(mf.min() - 4, mf.max() + 4)
    ap.yaxis.set_major_formatter(COMMA); ap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ap.set_xlabel("time (min)", fontsize=6, labelpad=1)
    lprof = label(ap, "total field (nT) · parse_explorer")
    pl, = ap.plot([], [], color=C["ink"], lw=0.6)

    gl = [i for i in range(len(sentences))]
    ev_g = [i for i in gl if st(i)]
    mlines = mtxt.strip().splitlines()
    mev = [i for i, s in enumerate(mlines) if not s.startswith("*")]
    rec_line_idx = [i for i, s in enumerate(mlines) if s.startswith("*")]
    frames = []
    T_STEPS = np.linspace(40, TT, 40)
    for tcur in T_STEPS:
        ig = int(min(tcur, len(sentences) - 1))
        shown = [i for i in range(ig + 1) if st(i) != "rejected" and i not in ()]
        gdots.set_offsets(gps_en[shown])
        rp = [i for i in range(ig + 1) if st(i) == "repaired"]
        grep_.set_offsets(gps_en[rp] if rp else np.empty((0, 2)))
        (be, bn), _ = pos(tcur - 0.01); boat.set_data([be], [bn])
        put(R1[0], sentences[ig])
        evs = [i for i in ev_g if i <= ig][-5:][::-1]
        for r in range(1, 6):
            if r - 1 < len(evs):
                i = evs[r - 1]; s_ = st(i)
                put(R1[r], sentences[i], marks[i], s_, C["ok"] if s_ == "repaired" else C["bad"])
            else:
                put(R1[r], "")
        sel = rec_t <= tcur
        tm = np.where(np.r_[False, np.diff(rec_t[sel]) > 5], np.nan, mf[sel])
        pl.set_data(rec_t[sel] / 60, tm)
        n_rec = int(sel.sum())
        last_m = rec_line_idx[n_rec - 1] if n_rec else 0
        put(R2[0], mlines[last_m] if n_rec else "")
        me = [i for i in mev if i < last_m][-2:][::-1]
        for r in range(1, 3):
            if r - 1 < len(me):
                s_ = mlines[me[r - 1]]
                gar = any(ord(c) > 126 for c in s_)
                put(R2[r], "".join("?" if ord(c) > 126 else c for c in s_), (), "garbage" if gar else "control",
                    C["bad"] if gar else C["warn"], C["bad"] if gar else C["warn"])
            else:
                put(R2[r], "")
        nrep = len([i for i in rp]); nrej = len([i for i in range(ig + 1) if st(i) == "rejected"])
        mm, ss = divmod(int(tcur), 60)
        status.set_text(f"{mm:d}:{ss:02d}  repaired {nrep} · rejected {nrej}")
        status.set_color(C["warn"] if nrep or nrej else C["acc"])
        frames.append(grab(fig))
    boat.set_data([], [])
    lt1.set_text(f"GPGGA · {g['stats']['repairedCount']} repaired, {g['stats']['rejectedCount']} rejected"
                 f" of {g['stats']['gpggaFound']:,}")
    frames += [grab(fig)] * 4

    # --- clean -----------------------------------------------------------------------------------
    rem_i = [int(t) - T0 for t in removed_t]
    gout.set_offsets(gps_en[rem_i] if rem_i else np.empty((0, 2)))
    lmap.set_text(f"export_clean_gps: {len(rem_i)} fixes off the track removed"); lmap.set_color(C["bad"])
    status.set_text(f"clean GPS {len(gc):,} of {len(g['valid']):,} fixes"); status.set_color(C["bad"])
    frames += [grab(fig)] * 8
    keep_i = [int(round(r["timeMs"] / 1000.0)) - T0 for r in gc]
    gdots.set_offsets(gps_en[keep_i])
    gout.set_offsets(np.empty((0, 2)))
    lmap.set_color(C["mute"])
    frames += [grab(fig)] * 4

    # --- merge + split ---------------------------------------------------------------------------
    LC = [C["acc"], C["ok"], C["warn"]]
    for r in R1 + R2:
        put(r, "")
    lt1.set_text("export_explorer_lines_zip · files written"); lt2.set_text("")
    grep_.set_offsets(np.empty((0, 2)))
    pl.set_data([], [])
    lat_ = lambda r: float(r["latitude"]); lon_ = lambda r: float(r["longitude"])
    for li, (ln_id, recs) in enumerate(sorted(lines.items())):
        pts = np.array([((lon_(r) - LON0) * MLON, (lat_(r) - LAT0) * MLAT) for r in recs if r.get("_hasGPS")])
        tt_ = np.array([(r["timeMs"] / 1000.0) - T0 for r in recs]); ff = np.array([r["field"] for r in recs])
        sc = am.scatter([], [], s=1.6, color=LC[li % 3], lw=0, zorder=3)
        pli, = ap.plot([], [], color=LC[li % 3], lw=0.7)
        for w in np.linspace(0.25, 1.0, 4):
            k = max(1, int(len(pts) * w)); kk = max(1, int(len(tt_) * w))
            sc.set_offsets(pts[:k]); pli.set_data(tt_[:kk] / 60, ff[:kk])
            status.set_text(f"auto_split: line {ln_id} · {len(recs):,} records"); status.set_color(LC[li % 3])
            frames.append(grab(fig))
        mag_n = [c for nm, c in outs if nm.endswith(f"{ln_id:03d}.MAG")][0]
        gps_n = [c for nm, c in outs if nm.endswith(f"{ln_id:03d}.GPS")]
        put(R1[li], f"MG_Line{ln_id:03d}.MAG  {mag_n:,} rec", tcol=LC[li % 3], col=LC[li % 3])
        if gps_n:
            put(R1[3 + li], f"MG_Line{ln_id:03d}.GPS  {gps_n[0]:,} fixes", col=C["ink"])
        frames += [grab(fig)] * 2
    lmap.set_text("merge_gps_to_mag: MAG records at GPS time"); lmap.set_color(C["mute"])
    lprof.set_text("total field (nT) per line file")
    gar = len([r for r in m["rejected"] if r["reason"] == "garbage"])
    put(R2[0], f"{len(m['data']):,} MAG records · {gar} garbage line dropped", col=C["ink"])
    put(R2[1], f"{mg['stats']['matched']:,} matched to GPS within 0.5 s", col=C["ink"])
    status.set_text(f"{len(lines)} lines · {mg['stats']['match_pct']:.1f}% matched"); status.set_color(C["ink"])
    frames.append(grab(fig))
    if "--peek" in sys.argv:
        frames[47].save(OUT / "_peek_prep.png")
    write_gif(frames, "magprep-clean.gif", hold=50, mid=24)
    return dict(gps=g["stats"], clean=len(gc), removed=len(rem_i), removed_t=removed_t[:10], mag=len(m["data"]),
                segments=len(m["segments"]), rejected_mag=m["rejected"][:2], lines={k: len(v) for k, v in lines.items()},
                merge=mg["stats"], outs=outs)


if __name__ == "__main__":
    which = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"proc", "prep"}
    if "proc" in which:
        print(magproc())
    if "prep" in which:
        print(magprep())
