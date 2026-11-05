"""Four profile animations, each driven by a real GeoView program function on synthetic data.

  sonarqc-waterfall.gif     SonarQC   sonarqc/analysis/{altitude,trackline,noise}.py
                                      analyze_altitude, analyze_trackline, analyze_noise
  maguxo-euler.gif          MagUXO    core/euler.py::euler_deconvolution
  mbesqc-density.gif        MBESQC    mbes_qc/surface_builder.py::_reduce_to_cells/_merge_cells/_grid_from_cells
                                      mbes_qc/grid_checks.py::check_grid
  seismic-velocity.gif      SeismicProcessor  processing/velocity_ai.py::compute_semblance, pick_semblance_peaks
                                      processing/nmo_stack.py::nmo_correction

Everything plotted (verdicts, depths, percentages, picks) comes back from those functions;
the data they are fed is synthetic and generated below. QC limits are set here, per GIF.

usage:  python make_visuals.py <out_dir> [repos_root] [geoview_shared] [last_png_dir] [sonar,mag,mbes,seis]
needs:  numpy, scipy, matplotlib, pillow, defusedxml (MBESQC import chain)
"""
import ast, sys, types, importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
R = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos")
SHARED = Path(sys.argv[3] if len(sys.argv) > 3 else R / "GeoViewShared")

C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})


# ---- loading real code without the heavy package imports ---------------------------------
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def pkg(name):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m


def exec_defs(path, names, ns):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body, []), str(path), "exec"), ns)
    return ns


# ---- figure helpers -----------------------------------------------------------------------
def fig_base(title, foot):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=C["bg"])
    prog, what = title
    fig.text(0.015, 0.935, prog, color=C["ink"], fontsize=9, fontweight="bold")
    fig.text(0.015 + 0.0128 * len(prog) + 0.008, 0.935, "·  " + what, color=C["mute"], fontsize=9)
    fig.text(0.015, 0.025, foot, color=C["mute"], fontsize=6)
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
    # one shared adaptive palette, built from a strip of sampled frames, so unchanged pixels stay
    # unchanged between frames (small files) and colours seen only mid-animation survive
    pick = frames[::max(1, len(frames) // 8)] + [frames[-1]]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(pick)))
    for i, f in enumerate(pick):
        strip.paste(f, (0, i * f.height))
    q = strip.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    rgb = q.getpalette()[:200 * 3]
    for hx in C.values():                                      # theme colours kept exact
        rgb += [int(hx[i:i + 2], 16) for i in (1, 3, 5)]
    ref = Image.new("P", (1, 1)); ref.putpalette(rgb + [0] * (768 - len(rgb)))
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / FPS)] * len(pal); dur[-1] = int(1000 / FPS) * hold
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=dur, loop=0,
                optimize=True, disposal=1)
    if LAST_DIR:
        frames[-1].save(LAST_DIR / (Path(name).stem + "-last.png"))
    print(f"{name}: {len(pal)} frames, {(OUT / name).stat().st_size / 1024:.0f} KB")
    plt.close("all")


LAST_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else None   # optional: where to drop last-frame PNGs
if LAST_DIR:
    LAST_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================================
# 1. SonarQC - waterfall + altitude / track / noise QC
# =========================================================================================
def sonarqc():
    sys.path[:0] = [str(R / "sonarqc"), str(SHARED)]
    from sonarqc.analysis.altitude import analyze_altitude
    from sonarqc.analysis.trackline import analyze_trackline
    from sonarqc.analysis.noise import analyze_noise
    from sonarqc.models.line import LineData
    from sonarqc.models.ping import PingData

    rng = np.random.default_rng(11)
    npg, ns, rmax = 240, 320, 150.0            # pings, samples per side, range per side (m)
    dr, dy = rmax / ns, 1.0                    # sample size, ping spacing (m)
    p = np.arange(npg)
    alt = 21 + 2.0 * np.sin(p / 23) + rng.normal(0, 0.25, npg)
    alt += 11.5 * np.exp(-((p - 122) / 11) ** 2)          # fish climbs: above window
    alt -= 8.5 * np.exp(-((p - 199) / 6) ** 2)            # fish dips: below window
    dev = 2.2 * np.sin(p / 31) + 13.5 * np.exp(-((p - 160) / 13) ** 2)  # cross-track drift (m)

    # seabed backscatter in seabed coordinates (along, across)
    def smooth_noise(shape, k):
        z = rng.normal(0, 1, shape)
        f = np.fft.rfft2(z); ky = np.fft.fftfreq(shape[0])[:, None]; kx = np.fft.rfftfreq(shape[1])[None]
        f *= np.exp(-(kx ** 2 + ky ** 2) * k)
        z = np.fft.irfft2(f, s=shape); return (z - z.mean()) / z.std()
    patch = smooth_noise((npg, 2 * ns), 4000)
    grain = rng.gamma(9.0, 1 / 9.0, (npg, 2 * ns))
    r = (np.arange(ns) + 0.5) * dr
    port = np.zeros((npg, ns)); stbd = np.zeros((npg, ns))
    targets = [(+1, 72, 58.0, 2.6, 5), (-1, 168, 92.0, 2.0, 7), (+1, 214, 34.0, 1.4, 3)]  # side, ping, range, h, half-len
    for i in range(npg):
        a = alt[i]
        g = np.sqrt(np.clip(r ** 2 - a ** 2, 0, None))
        graze = np.where(r > a, a / np.maximum(r, a), 0)
        for side, arr in ((-1, port), (+1, stbd)):
            col = np.clip((ns + side * g / dr).astype(int), 0, 2 * ns - 1)
            across = side * g
            ripple = 0.14 * np.sin(2 * np.pi * (across * 0.94 + i * dy * 0.34) / 6.5)
            gravel = 0.45 * np.clip((patch[i, col] - 0.7) * 3, 0, 1)
            bs = (0.55 + ripple + gravel + 0.12 * patch[i, col]) * grain[i, col]
            bs *= (0.35 + 0.65 * graze ** 0.35) * np.exp(-r / 260.0) / (1 + np.exp((r - 128) / 5.0))
            for ts, tp, tg, th, tl in targets:
                if ts == side and abs(i - tp) <= tl:
                    hi = (g > tg - 2.5) & (g < tg + 0.5)
                    bs[hi] = 2.6 * grain[i, col][hi]
                    sh = (g >= tg + 0.5) & (g < tg + 0.5 + th * tg / max(a - th, 1))
                    bs[sh] *= 0.06
            bs[r < a] = 0.0                               # water column
            bs += np.abs(rng.normal(0, 0.012, ns))        # receiver noise floor
            arr[i] = bs

    t0 = datetime(2026, 5, 14, 9, 30)
    lat0, lon0 = 35.10, 129.05
    mlat, mlon = 111320.0, 111320.0 * np.cos(np.radians(lat0))
    pings = [PingData(timestamp=t0 + timedelta(seconds=0.5 * k), ping_number=1000 + k,
                      latitude=lat0 + k * dy / mlat, longitude=lon0 + dev[k] / mlon,
                      altitude=float(alt[k]), heading=0.0, channel="port", frequency="HF")
             for k in range(npg)]
    line = LineData(line_name="SSS-L07", frequency="HF", pings=pings)
    planned = [(lat0, lon0), (lat0 + npg * dy / mlat, lon0)]
    AMIN, AMAX, TRK = 10.0, 20.0, 10.0
    altr = analyze_altitude(line, range_per_side=rmax, min_pct=AMIN, max_pct=AMAX)
    trk = analyze_trackline(line, max_allowed_deviation_m=TRK, planned_coords=planned, subsample_step=1)
    alt_bad = (altr.altitudes < altr.target_min) | (altr.altitudes > altr.target_max)
    tdev = trk.planned_deviations_m
    trk_bad = tdev > TRK

    fig, status = fig_base(("SonarQC", "side-scan waterfall + live QC"),
                           "analyze_altitude · analyze_trackline · analyze_noise  (sonarqc.analysis)"
                           "   ·   synthetic data · limits set by me")
    cmap = LinearSegmentedColormap.from_list("sss", ["#FFFFFF", "#F1E6CC", "#C9A35A", "#7A5418", "#2B1D07"])
    cmap.set_bad(C["bg"])
    a1 = fig.add_axes([0.065, 0.135, 0.50, 0.73]); style(a1)
    a2 = fig.add_axes([0.605, 0.135, 0.175, 0.73]); style(a2, grid=True)
    a3 = fig.add_axes([0.815, 0.135, 0.17, 0.73]); style(a3, grid=True)
    img = np.full((npg, 2 * ns), np.nan)
    full = np.hstack([port[:, ::-1], stbd])
    vmax = np.percentile(full, 99.7)
    im = a1.imshow(img, origin="lower", aspect="auto", cmap=cmap, norm=matplotlib.colors.PowerNorm(0.8, 0, vmax),
                   extent=[-rmax, rmax, 0, npg], interpolation="nearest")
    a1.set_xticks([-150, -75, 0, 75, 150]); a1.set_xticklabels(["150", "75", "0", "75", "150"])
    a1.set_xlabel("port      range (m)      stbd", fontsize=6, labelpad=1)
    a1.set_ylabel("ping", fontsize=6, labelpad=1)
    label(a1, "waterfall (HF)")
    for a in (a2, a3):
        a.set_ylim(0, npg); a.set_yticklabels([])
    a2.set_xlim(0, 40); a2.axvspan(altr.target_min, altr.target_max, color=C["ok"], alpha=0.08, lw=0)
    for v in (altr.target_min, altr.target_max):
        a2.axvline(v, color=C["ok"], lw=0.6, ls="--")
    a2.set_xlabel("altitude (m)", fontsize=6, labelpad=1)
    lab2 = a2.text(0.0, 1.015, "", transform=a2.transAxes, fontsize=6.5, color=C["mute"])
    a3.set_xlim(0, 20); a3.axvline(TRK, color=C["bad"], lw=0.6, ls="--")
    a3.set_xlabel("off-line (m)", fontsize=6, labelpad=1)
    lab3 = a3.text(0.0, 1.015, "", transform=a3.transAxes, fontsize=6.5, color=C["mute"])
    l2, = a2.plot([], [], color=C["ink"], lw=0.8)
    s2 = a2.scatter([], [], s=5, color=C["bad"], zorder=5, lw=0)
    l3, = a3.plot([], [], color=C["acc"], lw=0.8)
    s3 = a3.scatter([], [], s=5, color=C["bad"], zorder=5, lw=0)
    bands = []
    frames = []
    step = 4
    ks = list(range(step, npg + 1, step))
    for k in ks:
        img[:k] = full[:k]; im.set_data(img)
        y = p[:k] + 0.5
        l2.set_data(altr.altitudes[:k], y)
        b = alt_bad[:k]; s2.set_offsets(np.c_[altr.altitudes[:k][b], y[b]] if b.any() else np.empty((0, 2)))
        l3.set_data(tdev[:k], y)
        b3 = trk_bad[:k]; s3.set_offsets(np.c_[tdev[:k][b3], y[b3]] if b3.any() else np.empty((0, 2)))
        for bb in bands:
            bb.remove()
        bands = []
        flag = alt_bad[:k] | trk_bad[:k]
        if flag.any():                                           # red tick on the waterfall edge
            idx = np.flatnonzero(flag)
            bands.append(a1.scatter(np.full(idx.size, rmax - 2.5), idx + 0.5, marker="s", s=3.5,
                                    color=C["bad"], lw=0, zorder=6, clip_on=False))
        nin = int((~alt_bad[:k]).sum())
        lab2.set_text(f"in window {100 * nin / k:5.1f}%")
        lab2.set_color(C["bad"] if alt_bad[:k].any() else C["mute"])
        lab3.set_text(f"max {tdev[:k].max():4.1f} m")
        lab3.set_color(C["bad"] if trk_bad[:k].any() else C["mute"])
        if k >= 64:
            nz = analyze_noise(port[:k], stbd[:k], "SSS-L07", ping_rate_hz=2.0)
            snr = " / ".join(f"{n.overall_snr_db:.0f}" for n in nz)
            status.set_text(f"ping {k:3d}/{npg}   SNR P/S {snr} dB")
            status.set_color(C["acc"] if all(n.passed for n in nz) else C["bad"])
        else:
            status.set_text(f"ping {k:3d}/{npg}")
        frames.append(grab(fig))
    # verdict frame from the real result objects
    nz = analyze_noise(port, stbd, "SSS-L07", ping_rate_hz=2.0)
    alt_ok = altr.compliance_rate >= altr.pass_threshold_pct
    lab2.set_text(f"{altr.compliance_rate:.1f}% {'PASS' if alt_ok else 'FAIL'}")
    lab2.set_color(C["ok"] if alt_ok else C["bad"])
    lab3.set_text(f"max {trk.planned_max_deviation_m:.1f} m {'PASS' if trk.passed else 'FAIL'}")
    lab3.set_color(C["ok"] if trk.passed else C["bad"])
    status.set_text(f"{npg} pings   noise {'PASS' if all(n.passed for n in nz) else 'FAIL'}"
                    f" (SNR {min(n.overall_snr_db for n in nz):.0f} dB)")
    status.set_color(C["ok"] if all(n.passed for n in nz) else C["bad"])
    frames.append(grab(fig))
    write_gif(frames, "sonarqc-waterfall.gif", hold=40)


# =========================================================================================
# 2. MagUXO - Euler deconvolution on a survey grid
# =========================================================================================
def maguxo():
    pkg("core")
    load("core.features", R / "maguxo/core/features.py")
    syn = load("core.synthetic", R / "maguxo/core/synthetic.py")
    eu = load("core.euler", R / "maguxo/core/euler.py")
    euler_deconvolution = eu.euler_deconvolution

    rng = np.random.default_rng(5)
    x = syn.make_profile_axis()                       # 0..100 m, 256 samples (the program's line axis)
    ys = np.arange(0, 41, 1.0)                        # 41 lines, 1 m apart
    hs = syn.SENSOR_HEIGHT_M
    F = syn.earth_field_unit_vector()
    T = [(27.0, 12.0, 1.0, 2.0), (63.0, 28.0, 2.0, 6.0), (82.0, 9.0, 3.0, 10.0)]  # x, y, burial (m), moment
    X, Y = np.meshgrid(x, ys)
    G = np.zeros_like(X)
    for sx, sy, d, m in T:                           # point-dipole TMI, as in core/synthetic.py but 3-D
        rv = np.stack([X - sx, Y - sy, np.full_like(X, -(hs + d))], -1)
        dist = np.maximum(np.linalg.norm(rv, axis=-1), 0.1); rh = rv / dist[..., None]; mv = m * F
        b = 1e-7 * (3 * (rh @ mv)[..., None] * rh - mv) / dist[..., None] ** 3
        G += (b @ F) * 1e9
    G += 0.6 * np.sin(X / 37 + Y / 23) + rng.normal(0, 0.05, X.shape)   # geology trend + sensor noise

    WIN, FE, AMP = 21, 0.02, 2.0                      # window (program default), my acceptance filter
    half = WIN // 2; dx = x[1] - x[0]
    per_line = []
    for j in range(len(ys)):
        acc = []
        for s in euler_deconvolution(G[j], x, window=WIN, step=2):
            c = s.window_center
            if (s.fit_error < FE and abs(s.x0 - x[c]) < half * dx and s.depth < 12
                    and np.ptp(G[j, c - half:c + half + 1]) > AMP):
                acc.append((s.x0, ys[j], s.depth))
        per_line.append(np.array(acc).reshape(-1, 3))
    allsol = np.vstack(per_line)

    # cluster accepted solutions (greedy, 6 m radius) - display only
    clusters, left = [], allsol.copy()
    while len(left):
        dmat = np.hypot(left[:, None, 0] - left[None, :, 0], left[:, None, 1] - left[None, :, 1])
        i = int(np.argmax((dmat < 6).sum(1)))
        m = dmat[i] < 6
        if m.sum() >= 8:
            clusters.append((np.median(left[m, 0]), np.median(left[m, 1]), np.median(left[m, 2]), int(m.sum())))
        left = left[~m]

    fig, status = fig_base(("MagUXO", "Euler deconvolution finds buried dipoles"),
                           f"core/euler.py::euler_deconvolution (SI 3, window {WIN})   ·   synthetic data"
                           f" · acceptance filter set by me")
    mcm = LinearSegmentedColormap.from_list("mag", ["#57606A", "#AFB8C1", "#F6F8FA", "#FFFFFF",
                                                    "#F5E0A3", "#D4A72C", "#7D4E00"])
    mcm.set_bad(C["bg"])
    dcm = LinearSegmentedColormap.from_list("dep", ["#54AEFF", "#0969DA", "#0A3069"])
    a1 = fig.add_axes([0.06, 0.135, 0.56, 0.73]); style(a1)
    a2 = fig.add_axes([0.69, 0.135, 0.295, 0.73]); style(a2, grid=True)
    lim = np.percentile(np.abs(G), 99.5)
    img = np.full_like(G, np.nan)
    im = a1.imshow(img, origin="lower", aspect="auto", cmap=mcm, vmin=-lim, vmax=lim,
                   extent=[x[0], x[-1], ys[0] - 0.5, ys[-1] + 0.5], interpolation="bilinear")
    a1.set_xlabel("along line (m)", fontsize=6, labelpad=1); a1.set_ylabel("line (m)", fontsize=6, labelpad=1)
    label(a1, "total-field anomaly, 41 lines x 1 m")
    a1.set_xlim(x[0], x[-1]); a1.set_ylim(ys[0] - 0.5, ys[-1] + 0.5)
    cur = a1.axhline(-5, color=C["acc"], lw=0.9)
    sc1 = a1.scatter([], [], s=4, c=[], cmap=dcm, vmin=2, vmax=9, lw=0, zorder=5)
    a2.set_xlim(0, 100); a2.set_ylim(10, 0)
    a2.set_xlabel("x0 (m)", fontsize=6, labelpad=1); a2.set_ylabel("depth below sensor (m)", fontsize=6, labelpad=1)
    label(a2, "Euler solutions, section view")
    sc2 = a2.scatter([], [], s=4, c=[], cmap=dcm, vmin=2, vmax=9, lw=0, alpha=0.6)
    frames = []
    for j in range(len(ys)):
        img[:j + 1] = G[:j + 1]; im.set_data(img)
        cur.set_ydata([ys[j]] * 2)
        sol = np.vstack(per_line[:j + 1])
        for s_ in (sc1, sc2):
            if len(sol):
                s_.set_array(sol[:, 2])
        if len(sol):
            sc1.set_offsets(sol[:, :2]); sc2.set_offsets(sol[:, [0, 2]])
        status.set_text(f"line {j + 1:2d}/{len(ys)}   {len(sol):4d} solutions accepted")
        frames.append(grab(fig))
    cur.set_visible(False)
    status.set_text(f"{len(allsol)} solutions  ->  {len(clusters)} sources")
    for k, (cx, cy, cz, n) in enumerate(sorted(clusters)):
        a1.scatter([cx], [cy], s=150, facecolors="none", edgecolors=C["ink"], lw=0.9, zorder=6)
        frames += [grab(fig)] * 3
    frames.append(grab(fig))
    write_gif(frames, "maguxo-euler.gif", hold=40)
    return clusters, T


# =========================================================================================
# 3. MBESQC - lines fill a sounding-density grid; check_grid judges it
# =========================================================================================
def mbesqc():
    sys.path[:0] = [str(R / "mbesqc"), str(SHARED)]
    from mbesqc.mbes_qc.surface_builder import _reduce_to_cells, _merge_cells, _grid_from_cells
    from mbesqc.mbes_qc.grid_checks import check_grid

    rng = np.random.default_rng(2)
    XW, YH, CELL = 400.0, 250.0, 2.5
    MINC, REQ = 5, 95.0                                   # density criterion: >=5 soundings in 95% of cells
    depth = lambda e, n: 30 - 22 * np.exp(-((e - 255) / 70) ** 2 - ((n - 150) / 80) ** 2) + 0.004 * e
    lines = [25, 75, 125, 175, 225]
    nbeam, swath, dping = 100, 60.0, 1.4
    th = np.radians(np.linspace(-swath, swath, nbeam))
    nx, ny = int(XW / CELL), int(YH / CELL)
    foot = np.ones((ny, nx), dtype=bool)

    def ping_block(ln, e_from, e_to, direction):
        es = np.arange(e_from, e_to, dping) if direction > 0 else np.arange(e_from, e_to, -dping)
        yv = lines[ln] + 0.8 * np.sin(es / 40 + ln)
        z0 = depth(es, yv)
        across = z0[:, None] * np.tan(th)[None] * (1 + rng.normal(0, 0.01, (len(es), nbeam)))
        e = np.repeat(es[:, None], nbeam, 1) + rng.normal(0, 0.15, across.shape)
        n = yv[:, None] - direction * across
        d = depth(e, n) + rng.normal(0, 0.08, e.shape)
        keep = rng.random(e.shape) > 0.04                  # a few rejected beams
        inside = (e >= 0) & (e < XW) & (n >= 0) & (n < YH) & keep
        return es, yv, e[inside], n[inside], d[inside]

    fig, status = fig_base(("MBESQC", "sounding density, line by line"),
                           "surface_builder._reduce_to_cells/_merge_cells/_grid_from_cells · check_grid"
                           "  ·  synthetic data · criteria set by me")
    a1 = fig.add_axes([0.055, 0.135, 0.585, 0.73]); style(a1)
    a2 = fig.add_axes([0.70, 0.135, 0.285, 0.73]); a2.axis("off")
    ee, nn = np.meshgrid(np.linspace(0, XW, 120), np.linspace(0, YH, 80))
    a1.contour(ee, nn, depth(ee, nn), levels=np.arange(10, 32, 4), colors=C["ink"], linewidths=0.4, alpha=0.35, zorder=2)
    a1.set_xlim(0, XW); a1.set_ylim(0, YH); a1.set_aspect("auto")
    a1.set_xlabel("easting (m)", fontsize=6, labelpad=1); a1.set_ylabel("northing (m)", fontsize=6, labelpad=1)
    label(a1, f"{CELL} m cells · contours = depth (m)")
    dens_cm = LinearSegmentedColormap.from_list("den", ["#DDF4FF", "#54AEFF", "#0969DA", "#0A3069"])
    rgba = np.zeros((ny, nx, 4))
    im = a1.imshow(rgba, origin="lower", extent=[0, XW, 0, YH], aspect="auto", interpolation="nearest", zorder=1)
    for y in lines:
        a1.axhline(y, color=C["mute"], lw=0.4, ls=(0, (2, 3)), zorder=0)
    boat, = a1.plot([], [], marker="o", ms=3.5, color=C["ink"], zorder=4, lw=0)
    sw, = a1.plot([], [], color=C["ink"], lw=1.0, alpha=0.6, zorder=3)

    # right panel: bars driven by check_grid's result dict
    def bar(y, txt):
        a2.text(0, y + 0.075, txt, fontsize=6.5, color=C["ink"], transform=a2.transAxes)
        a2.add_patch(plt.Rectangle((0, y), 1, 0.045, transform=a2.transAxes, color=C["hair"], lw=0))
        b = plt.Rectangle((0, y), 0, 0.045, transform=a2.transAxes, color=C["acc"], lw=0); a2.add_patch(b)
        v = a2.text(1, y + 0.075, "", fontsize=6.5, ha="right", color=C["ink"], transform=a2.transAxes)
        return b, v
    b_occ, t_occ = bar(0.80, "node occupancy")
    b_den, t_den = bar(0.58, f"cells >= {MINC} soundings")
    a2.plot([REQ / 100] * 2, [0.565, 0.64], color=C["ink"], lw=0.8, transform=a2.transAxes)
    a2.text(REQ / 100, 0.525, f"req {REQ:.0f}%", ha="center", fontsize=5.5, color=C["mute"], transform=a2.transAxes)
    t_hol = a2.text(0, 0.40, "", fontsize=6.5, color=C["ink"], transform=a2.transAxes)
    t_pts = a2.text(0, 0.31, "", fontsize=6.5, color=C["mute"], transform=a2.transAxes)
    keys = [("#0969DA", f">= {MINC}"), ("#9A6700", f"1-{MINC - 1}"), ("#CF222E", "holiday")]
    for i, (col, tx) in enumerate(keys):
        a2.add_patch(plt.Rectangle((i * 0.34, 0.17), 0.06, 0.04, transform=a2.transAxes, color=col, lw=0))
        a2.text(i * 0.34 + 0.08, 0.17, tx, fontsize=6, color=C["mute"], transform=a2.transAxes)
    verdict = a2.text(0, 0.02, "", fontsize=8.5, fontweight="bold", transform=a2.transAxes)

    acc, npts = None, 0
    frames = []
    nseg = 10

    def render(final=False):
        g = _grid_from_cells(acc, CELL, num_points=npts, num_lines=5)
        dens = np.zeros((ny, nx), dtype=np.int64); dtm = np.full((ny, nx), np.nan)
        c0, r0 = int(round(g.x_min / CELL)), int(round(g.y_min / CELL))
        dens[r0:r0 + g.ny, c0:c0 + g.nx] = g.density.astype(np.int64)
        dtm[r0:r0 + g.ny, c0:c0 + g.nx] = g.dtm
        res = check_grid(np.ma.masked_invalid(dtm), density_count=dens, footprint=foot,
                         minimum_count=MINC, required_percent=REQ)
        rgba[:] = 0
        good = dens >= MINC; low = (dens > 0) & ~good
        rgba[good] = dens_cm(np.clip(dens[good] / 40, 0, 1))
        rgba[low] = matplotlib.colors.to_rgba(C["warn"])
        if final:
            rgba[foot & (dens == 0)] = matplotlib.colors.to_rgba(C["bad"])
        im.set_data(rgba)
        occ = res["occupancy"]["occupied_percent"]; den = res["density"]["within_percent"]
        b_occ.set_width(occ / 100); t_occ.set_text(f"{occ:5.1f}%")
        b_den.set_width(den / 100); t_den.set_text(f"{den:5.1f}%")
        b_den.set_color(C["acc"] if den >= REQ else C["warn"])
        t_pts.set_text(f"{npts / 1e3:,.0f} k soundings")
        return res, occ, den

    for ln in range(len(lines)):
        dirn = 1 if ln % 2 == 0 else -1
        edges = np.linspace(0, XW, nseg + 1) if dirn > 0 else np.linspace(XW, 0, nseg + 1)
        for s in range(nseg):
            es, yv, e, n, d = ping_block(ln, edges[s], edges[s + 1], dirn)
            cells, nv = _reduce_to_cells(e, n, d, None, None, CELL)
            npts += nv
            acc = cells if acc is None else _merge_cells(acc, cells)
            res, occ, den = render()
            zb = depth(es[-1], yv[-1]); hw = zb * np.tan(np.radians(swath))
            boat.set_data([es[-1]], [yv[-1]]); sw.set_data([es[-1]] * 2, [yv[-1] - hw, yv[-1] + hw])
            status.set_text(f"line {ln + 1}/{len(lines)}   coverage {occ:5.1f}%")
            frames.append(grab(fig))
    boat.set_data([], []); sw.set_data([], [])
    res, occ, den = render(final=True)
    comp = res["nodata_components"]
    t_hol.set_text(f"holidays  {comp['interior_components'] + comp['exterior_components']} areas,"
                   f" {comp['interior_cells'] + comp['exterior_cells']} cells")
    t_hol.set_color(C["bad"])
    met = res["density"]["threshold_met"]
    verdict.set_text(f"density {'PASS' if met else 'FAIL'}  (req {REQ:.0f}%)")
    verdict.set_color(C["ok"] if met else C["bad"])
    status.set_text(f"5 lines   coverage {occ:5.1f}%")
    frames.append(grab(fig))
    write_gif(frames, "mbesqc-density.gif", hold=45)
    return res


# =========================================================================================
# 4. SeismicProcessor - semblance velocity analysis + NMO
# =========================================================================================
def seismic():
    from scipy.signal import medfilt2d
    from scipy.ndimage import maximum_filter
    import logging
    sp = R / "seismicprocessor/seismic_processor/processing"
    ns_ = {"np": np, "medfilt2d": medfilt2d, "maximum_filter": maximum_filter,
           "Tuple": tuple, "List": list, "Optional": object, "logger": logging.getLogger("sp")}
    exec_defs(sp / "velocity_ai.py", {"compute_semblance", "pick_semblance_peaks", "_viterbi_pick"}, ns_)
    exec_defs(sp / "nmo_stack.py", {"nmo_correction"}, ns_)
    compute_semblance, pick_semblance_peaks, nmo_correction = (ns_["compute_semblance"],
                                                               ns_["pick_semblance_peaks"], ns_["nmo_correction"])

    rng = np.random.default_rng(4)
    dt, nsamp, ntr = 0.004, 501, 24
    off = np.linspace(100, 2400, ntr); t = np.arange(nsamp) * dt
    events = [(0.36, 1500, 1.0), (0.72, 1640, -0.75), (1.06, 1830, 0.8), (1.42, 2080, -0.65), (1.78, 2330, 0.6)]
    gather = np.zeros((ntr, nsamp))
    for t0, v, a in events:
        tx = np.sqrt(t0 ** 2 + (off / v) ** 2)
        for i in range(ntr):
            arg = (np.pi * 22 * (t - tx[i])) ** 2
            gather[i] += a * (1 - 2 * arg) * np.exp(-arg) * (1 - 0.25 * off[i] / off[-1])
    gather += 0.06 * rng.standard_normal(gather.shape)
    vels = np.arange(1300, 3001, 25.0)
    semb = compute_semblance(gather.astype(np.float32), off, dt, vels, half_window=5)
    picks = pick_semblance_peaks(semb, vels, t, min_semblance=0.15, n_picks=8, method="peaks")
    tp = np.array([p[0] for p in picks]); vp = np.array([p[1] for p in picks])

    fig, status = fig_base(("SeismicProcessor", "semblance velocity analysis"),
                           "velocity_ai.compute_semblance · pick_semblance_peaks · nmo_stack.nmo_correction"
                           "   ·   synthetic CMP gather")
    axs = [fig.add_axes([0.06 + i * 0.322, 0.135, 0.28, 0.73]) for i in range(3)]
    for a, ttl in zip(axs, ("CMP gather", "semblance", "NMO-corrected")):
        style(a); label(a, ttl); a.set_ylim(t[-1], 0)
    axs[0].set_ylabel("time (s)", fontsize=6, labelpad=1)
    for a in axs[1:]:
        a.set_yticklabels([])

    def wiggle(ax, data):
        for art in list(ax.lines) + list(ax.collections):
            art.remove()
        sc = 0.95 * (off[1] - off[0]) / max(np.abs(gather).max(), 1e-9)
        for i in range(ntr):
            xx = off[i] + data[i] * sc
            ax.plot(xx, t, color=C["ink"], lw=0.3)
            ax.fill_betweenx(t, off[i], xx, where=data[i] > 0, color=C["ink"], lw=0)
        ax.set_xlim(off[0] - 120, off[-1] + 120)
    for a in (axs[0], axs[2]):
        a.set_xticks([500, 1500, 2400]); a.set_xlabel("offset (m)", fontsize=6, labelpad=1)
    wiggle(axs[0], gather); wiggle(axs[2], gather)
    scm = LinearSegmentedColormap.from_list("semb", ["#FFFFFF", "#FFF1C2", "#E3A008", "#9A3412", "#3B0A0A"])
    shown = np.full_like(semb, np.nan)
    im = axs[1].imshow(shown, aspect="auto", cmap=scm, vmin=0, vmax=float(semb.max()),
                       extent=[vels[0], vels[-1], t[-1], t[0]], interpolation="bilinear")
    axs[1].set_xticks([1500, 2000, 2500, 3000]); axs[1].set_xlabel("velocity (m/s)", fontsize=6, labelpad=1)
    scan = axs[1].axvline(vels[0], color=C["acc"], lw=0.8)
    pl, = axs[1].plot([], [], color=C["acc"], lw=1.0)
    ps = axs[1].scatter([], [], s=16, color=C["acc"], edgecolors="white", lw=0.6, zorder=5)
    frames = []
    for k in np.linspace(1, len(vels), 22).astype(int):
        shown[:, :k] = semb[:, :k]; im.set_data(shown); scan.set_xdata([vels[k - 1]] * 2)
        status.set_text(f"scanning v = {vels[k - 1]:.0f} m/s")
        frames.append(grab(fig))
    scan.set_visible(False)
    for k in range(1, len(tp) + 1):
        ps.set_offsets(np.c_[vp[:k], tp[:k]]); pl.set_data(vp[:k], tp[:k])
        status.set_text(f"pick {k}/{len(tp)}   t {tp[k - 1]:.2f} s   v {vp[k - 1]:.0f} m/s")
        frames += [grab(fig)] * 2
    for s in np.linspace(0, 1, 16) ** 1.5:
        veff = vp / np.sqrt(max(s, 1e-4))
        wiggle(axs[2], nmo_correction(gather.astype(np.float32), off, dt, tp, veff, stretch_mute=1.5))
        status.set_text(f"NMO  {100 * s:3.0f}% of picked v(t)")
        frames.append(grab(fig))
    status.set_text(f"{len(tp)} picks   events flattened")
    frames.append(grab(fig))
    write_gif(frames, "seismic-velocity.gif", hold=40)
    return picks, events


if __name__ == "__main__":
    which = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"sonar", "mag", "mbes", "seis"}
    if "sonar" in which: sonarqc()
    if "mag" in which: print(maguxo())
    if "mbes" in which: r = mbesqc(); print(r["occupancy"]["occupied_percent"], r["density"])
    if "seis" in which: print(seismic())
