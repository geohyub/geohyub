"""Two more profile animations, same style as make_visuals.py (palette, figure size, helpers copied below).

  magqc-diurnal.gif    MagQC    magqc/core.py::calc_fourth_diff, detect_spikes, detect_anomalies
                                magqc/base_diurnal.py::chord_deviation
                                magqc/pipeline.py::analyze_line (4th-difference noise, noise, integrity,
                                  timestamp, processed residual) -> magqc/mag_verdict.py::line_verdict
                       MagProc  magproc/core/corrections/igrf.py::compute_igrf_field_array
                                magproc/core/corrections/diurnal.py::interpolate_diurnal
                                (MagQC does not correct data itself; IGRF and diurnal removal are MagProc's
                                 job, so the residual shown is made by MagProc's functions)
  tidebedpy-tide.gif   TideBedPy  tidebedpy/lat_khoa.py::coef_from_constants, synth (utide harmonic synthesis,
                                  constituent subsets M2 -> +S2 -> +K1 -> +O1)
                                tidebedpy/data_io/tide_series.py::interpolate_akima
                                tidebedpy/core/tide_correction.py::TideCorrectionEngine.process_nav_point
                                  (IDW over gauges, co-tidal range ratio and MHWI time shift -> Tc per fix)
                                The reduction of soundings to datum (depth - Tc) is a subtraction done here:
                                TideBedPy writes Tc, the processing software applies it.

Everything plotted (FD, flags, verdict, diurnal, IGRF, tide curves, Tc) comes back from those functions; the data
they are fed is synthetic and generated below: a towed-magnetometer line with two buried targets, three spikes and
a noise burst; two tide gauges with invented M2/S2/K1/O1 constants and a co-tidal grid stand-in (linear in
longitude) that answers get_cotidal_values like the KHOA NetCDF grid does. No survey data, station or project
names are used. QC limits: MagQC in-house defaults (fd_threshold 0.5 nT raw, pp_limit 1.0 nT).

usage:  python make_magqc_tide_gif.py <out_dir> [magqc_repo] [magproc_repo] [tidebedpy_repo] [mag,tide]
needs:  numpy, scipy, matplotlib, pillow, pandas, utide, pyyaml (MagProc import chain)
"""
import sys, types
sys.dont_write_bytecode = True                     # import from the repos without leaving __pycache__ there
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
MAGQC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/QC/MagQC")
MAGPROC = Path(sys.argv[3] if len(sys.argv) > 3 else "/home/user/repos/Processing/MagProc")
TIDEBED = Path(sys.argv[4] if len(sys.argv) > 4 else "/home/user/repos/Preprocessing/TideBedPy")
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


# =========================================================================================
# 1. MagQC - towed magnetometer line QC'd as it is logged
# =========================================================================================
def magqc():
    sys.path[:0] = [str(MAGQC)]
    import magqc.core as core
    from magqc.base_diurnal import chord_deviation
    from magqc import pipeline
    from magqc import mag_verdict as MV
    pkg("magproc", MAGPROC / "magproc"); pkg("magproc.core", MAGPROC / "magproc/core")
    pkg("magproc.core.corrections", MAGPROC / "magproc/core/corrections")   # skip the package __init__s
    from magproc.core.corrections.igrf import compute_igrf_field_array
    from magproc.core.corrections.diurnal import interpolate_diurnal

    rng = np.random.default_rng(5)
    FS, N, V = 10.0, 18000, 2.3                    # 10 Hz, 30 min, 2.3 m/s (about 4.1 km)
    k = np.arange(N); ts = k / FS; x = ts * V      # seconds / metres along the line
    T0 = datetime(2026, 5, 14, 2, 10, 0)
    t0_ms = (T0 - datetime(1970, 1, 1)).total_seconds() * 1000
    lat0, lon0 = 34.4, 128.5                       # open water, not shown
    lats = np.full(N, lat0); lons = lon0 + x / (111320 * np.cos(np.radians(lat0)))
    igrf = compute_igrf_field_array(lats, lons, 0.0, "2026-05-14")

    def dipole(xc, h, A, phi=np.radians(50)):      # 2-D buried-dipole total-field profile
        d = x - xc
        return A * h * h * ((h * h - d * d) * np.cos(phi) + 2 * h * d * np.sin(phi)) / (d * d + h * h) ** 2
    Lm = x[-1]
    geol = 2.0 * np.sin(2 * np.pi * x / 2600) + 0.002 * x
    targets = dipole(0.36 * Lm, 6, 25) + dipole(0.72 * Lm, 5, 8)   # two buried objects, 6 m and 5 m below sensor

    def diurnal_true(t):                           # slow daily trend + two slow pulsations
        return 3.4 * np.sin(2 * np.pi * t / 5400) + 0.8 * np.sin(2 * np.pi * t / 300 + 1) + 0.4 * np.sin(2 * np.pi * t / 600)
    bt = np.arange(-240.0, ts[-1] + 300.0, 1.0)    # base station, 1 Hz, 4-5 min either side
    base = 49_962.0 + diurnal_true(bt) + rng.normal(0, 0.03, bt.size)
    noise = rng.normal(0, 0.010, N)
    b0 = int(0.61 * N); burst = slice(b0, b0 + 310)   # 31 s of cable-strum noise
    noise[burst] += rng.normal(0, 0.05, burst.stop - burst.start)
    raw = igrf + geol + targets + diurnal_true(ts) + noise
    for f, a in ((0.27, 4.5), (0.81, -3.2), (0.93, 2.6)):
        raw[int(f * N)] += a                       # three single-sample spikes
    epoch = t0_ms + ts * 1000

    diu = interpolate_diurnal(t0_ms + bt * 1000, base, epoch)          # MagProc
    resid = raw - igrf - diu
    cd = chord_deviation(bt, base, chord_minutes=4.0)                 # MagQC, no in-house limit -> not judged
    QP = {"fd_threshold": core.FD_THRESHOLD_DEFAULT, "pp_limit": core.PP_LIMIT_DEFAULT,
          "mad_mult": core.MAD_MULT_DEFAULT}
    THR = QP["fd_threshold"]

    foot = ["MagQC calc_fourth_diff · detect_spikes · chord_deviation · analyze_line · line_verdict"
            "  ·  in-house 0.5 nT",
            "IGRF and diurnal removal by MagProc (compute_igrf_field_array, interpolate_diurnal)"
            "  ·  synthetic line"]
    fig, status = fig_base(("MagQC", "towed magnetometer line QC"), foot)
    X0, XW = 0.085, 0.475
    a1 = fig.add_axes([X0, 0.655, XW, 0.20]); style(a1, grid=True)
    a2 = fig.add_axes([X0, 0.405, XW, 0.17]); style(a2, grid=True)
    a3 = fig.add_axes([X0, 0.175, XW, 0.15]); style(a3, grid=True)
    a4 = fig.add_axes([0.635, 0.50, 0.35, 0.355]); style(a4, grid=True)
    ach = fig.add_axes([0.635, 0.12, 0.35, 0.265]); ach.axis("off")
    tmin = ts / 60
    for a in (a1, a2, a3, a4):
        a.set_xlim(0, tmin[-1])
    for a in (a1, a2):
        a.set_xticklabels([])
    a3.set_xlabel("time on line (min)", fontsize=6, labelpad=1)
    a4.set_xlabel("time on line (min)", fontsize=6, labelpad=1)
    pad = lambda lo, hi, f=0.08: (lo - f * (hi - lo), hi + f * (hi - lo))
    a1.set_ylim(*pad(raw.min(), raw.max())); a1.yaxis.set_major_formatter(COMMA)
    a1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    label(a1, "raw total field (nT) · IGRF-14 dashed")
    a1.plot(tmin, igrf + (raw.mean() - igrf.mean()), color=C["mute"], lw=0.6, ls=(0, (3, 3)))
    a2.set_ylim(*pad(diu.min(), diu.max(), 0.15)); a2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    lab2 = label(a2, "base-station diurnal (nT) · 4-min chords")
    a3.set_ylim(*pad(resid.min(), resid.max())); a3.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    label(a3, "residual = raw - IGRF - diurnal (nT)")
    a4.set_yscale("log"); a4.set_ylim(0.002, 60)
    a4.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    a4.axhline(THR, color=C["bad"], lw=0.7, ls="--")
    a4.text(tmin[-1], THR * 1.25, f"{THR:g} nT", color=C["bad"], fontsize=6, ha="right", va="bottom")
    lab4 = label(a4, "|4th difference| (nT, raw convention)")

    l1, = a1.plot([], [], color=C["acc"], lw=0.7)
    sp1 = a1.scatter([], [], s=16, facecolors="none", edgecolors=C["bad"], lw=0.9, zorder=5)
    l2, = a2.plot([], [], color=C["ink"], lw=0.7)
    chord_lines = []
    l3, = a3.plot([], [], color=C["ink"], lw=0.7)
    l4, = a4.plot([], [], color=C["mute"], lw=0.3)
    ex4 = a4.scatter([], [], s=4, color=C["bad"], lw=0, zorder=5)
    boat_lines = [a.axvline(-1, color=C["acc"], lw=0.5, alpha=0.6) for a in (a1, a2, a3, a4)]
    btm = bt / 60

    frames = []
    steps = np.linspace(160, N, 46).astype(int)
    for kk in steps:
        f_ = raw[:kk]
        fd = np.abs(np.array(core.calc_fourth_diff(list(f_))))          # MagQC, on what is logged so far
        spk = core.detect_spikes(list(f_), QP["mad_mult"])
        si = [s["index"] for s in spk["spikes"]]
        l1.set_data(tmin[:kk], f_)
        sp1.set_offsets(np.c_[tmin[si], f_[si]] if si else np.empty((0, 2)))
        l2.set_data(tmin[:kk], diu[:kk])
        l3.set_data(tmin[:kk], resid[:kk])
        tf = tmin[2:kk - 2]
        l4.set_data(tf, np.clip(fd, 0.0025, None))
        ex = fd > THR
        ex4.set_offsets(np.c_[tf[ex], np.minimum(fd[ex], 55)] if ex.any() else np.empty((0, 2)))
        for c_ in cd["chords"]:                    # chord appears once the line has passed its end
            if c_["max_dev_nt"] is None or c_["index"] in [cl[0] for cl in chord_lines]:
                continue
            if c_["end_s"] / 60 <= tmin[kk - 1] + 1e-9 and c_["start_s"] >= 0:
                m = (bt >= c_["start_s"]) & (bt < c_["end_s"])
                e0, e1 = np.flatnonzero(m)[[0, -1]]
                b0 = np.interp(bt[[e0, e1]], bt, base - base.mean())
                ln, = a2.plot(btm[[e0, e1]], b0, color=C["warn"], lw=0.7, ls="--")
                chord_lines.append((c_["index"], ln))
        for b_ in boat_lines:
            b_.set_xdata([tmin[kk - 1]] * 2)
        mm, ss = divmod(int(ts[kk - 1]), 60)
        nex = int(ex.sum())
        status.set_text(f"{mm:d}:{ss:02d}   FD > {THR:g} nT: {nex:,}   spikes {len(si)}")
        status.set_color(C["bad"] if nex or si else C["acc"])
        frames.append(grab(fig))
    for b_ in boat_lines:
        b_.set_visible(False)

    # --- the program's QC of the full line ------------------------------------------------
    recs = [dict(field=float(v), epochMs=float(e)) for v, e in zip(raw, epoch)]
    line = pipeline.analyze_line(recs, qc_params=QP)
    lv = MV.line_verdict(line)
    an = core.detect_anomalies(list(raw))["anomalies"]       # the spans the contract FD exempts as signal
    spans = []                                     # merge spans closer than 30 s, for drawing
    for a_ in sorted(an, key=lambda a: a["start_index"]):
        s0, s1 = tmin[a_["start_index"]], tmin[min(a_["end_index"], N - 1)]
        if spans and s0 - spans[-1][1] < 0.5:
            spans[-1][1] = s1
        else:
            spans.append([s0, s1])
    for s0, s1 in spans:
        c_ = (s0 + s1) / 2; hw = max((s1 - s0) / 2, 0.25)
        for ax_ in (a3, a4):
            ax_.axvspan(c_ - hw, c_ + hw, color=C["ok"], alpha=0.18, lw=0, zorder=0)
        a3.text(c_ + hw + 0.3, a3.get_ylim()[1] * 0.78, "anomaly", fontsize=5.6, color=C["ok"], va="center")
    nfd = line["fourth_diff_noise"]
    lab4.set_text(f"|4th diff| · {nfd['exceedance_count']:,} noise samples > {THR:g} nT")
    lab4.set_color(C["bad"] if nfd["exceedance_count"] else C["mute"])
    lab2.set_text(f"base diurnal (nT) · max chord deviation {cd['max_dev_nt']:.1f} nT")
    status.set_text(f"{len(an)} anomaly spans exempt · {line['spikes']['spike_count']} spikes despiked")
    status.set_color(C["ink"])
    frames += [grab(fig)] * 6

    WORD = {MV.PASS: ("PASS", C["ok"]), MV.CONDITIONAL: ("CONDITIONAL", C["warn"]),
            MV.FAIL: ("FAIL", C["bad"]), MV.NA: ("NOT MEASURED", C["mute"])}
    SC = {"PASS": C["ok"], "WARNING": C["warn"], "FAIL": C["bad"], "N/A": C["mute"]}
    rows = [("4th difference", lv["dimensions"]["fourth_diff"],
             f"max {nfd['max_abs']:.2f} nT"),
            ("noise P-P", lv["dimensions"]["noise"], f"{line['noise']['pp']:.2f} / {line['noise']['pp_limit']:g} nT"),
            ("integrity", lv["dimensions"]["integrity"], f"{line['integrity']['validPct']:.0f}% valid"),
            ("timestamps", lv["dimensions"]["timestamp"],
             f"{line['timestamp'].get('regularity_pct', 0):.0f}% regular")]
    for i, (nm, st, det) in enumerate(rows):
        y = 0.92 - i * 0.2
        ach.text(0.0, y, nm, fontsize=6.3, color=C["ink"], transform=ach.transAxes, va="center")
        ach.text(0.40, y, st, fontsize=6.3, color=SC[st], fontweight="bold", transform=ach.transAxes, va="center")
        ach.text(0.62, y, det, fontsize=5.8, color=C["mute"], transform=ach.transAxes, va="center")
        frames += [grab(fig)] * 3
    word, col = WORD[lv["verdict"]]
    ach.text(0.0, 0.08, f"line verdict  {word}", fontsize=7.5, fontweight="bold", color=col,
             transform=ach.transAxes, va="center")
    status.set_text(f"{lv['pass']} pass · {lv['warn']} warn · {lv['fail']} fail -> {word}")
    status.set_color(col)
    frames.append(grab(fig))
    write_gif(frames, "magqc-diurnal.gif", hold=50, mid=30)
    return dict(verdict=word, dims=lv["dimensions"], fd_noise=nfd, noise=line["noise"], anomalies=len(an),
                spikes=line["spikes"]["spike_count"], chord_max=cd["max_dev_nt"], igrf=(igrf.min(), igrf.max()))


# =========================================================================================
# 2. TideBedPy - harmonic tide built constituent by constituent; soundings reduced to datum
# =========================================================================================
def tidebed():
    import pandas as pd
    sys.path[:0] = [str(TIDEBED)]
    from tidebedpy import lat_khoa as LK
    from tidebedpy.data_io.station import RefStation
    from tidebedpy.data_io.tide_series import TideSeries, TideRecord, interpolate_akima
    from tidebedpy.data_io.navigation import NavPoint
    from tidebedpy.core.tide_correction import TideCorrectionEngine
    from tidebedpy.config import TideBedConfig

    LAT = 34.0
    KST = timedelta(hours=9)
    CON = ["M2", "S2", "K1", "O1"]
    PER = {"M2": 12.4206, "S2": 12.0, "K1": 23.9345, "O1": 25.8193}
    # invented constants (amplitude cm, Greenwich phase deg) for two gauges
    GA = dict(M2=(118.0, 205.0), S2=(52.0, 248.0), K1=(31.0, 162.0), O1=(23.0, 131.0))
    GB = dict(M2=(96.0, 221.0), S2=(41.0, 266.0), K1=(28.0, 170.0), O1=(21.0, 138.0))

    def coef_for(g):
        z0 = sum(a for a, _ in g.values())          # MSL above DL = M2 + S2 + K1 + O1 (approximate DL)
        return LK.coef_from_constants([dict(name=n, amp_cm=g[n][0], pha_gmt=g[n][1]) for n in CON], LAT, z0), z0
    cA, zA = coef_for(GA); cB, zB = coef_for(GB)

    # --- part 1: 15-day prediction at gauge A, constituent by constituent --------------------
    D0 = datetime(2026, 5, 1)                       # KST
    t15 = np.arange(np.datetime64(D0 - KST), np.datetime64(D0 + timedelta(days=15) - KST), np.timedelta64(20, "m"))
    days = (t15 - t15[0]) / np.timedelta64(1, "D")
    parts = [LK.synth(cA, t15, constit=CON[:i + 1]) for i in range(4)]      # mean + first i+1 constituents
    single = [LK.synth(cA, t15, constit=[n]) - zA for n in CON]
    full = parts[-1]

    foot = ["TideBedPy lat_khoa.coef_from_constants · synth (utide) · interpolate_akima"
            " · TideCorrectionEngine.process_nav_point",
            "synthetic gauges, constants and co-tidal grid  ·  depth - Tc done here (TideBedPy writes Tc)"]
    fig, status = fig_base(("TideBedPy", "harmonic tide and tide reduction"), foot)
    a1 = fig.add_axes([0.075, 0.555, 0.56, 0.30]); style(a1, grid=True)
    a2 = fig.add_axes([0.715, 0.555, 0.27, 0.30]); style(a2, grid=True)
    a3 = fig.add_axes([0.075, 0.175, 0.91, 0.27]); style(a3, grid=True)
    a1.set_xlim(0, 15); a1.set_ylim(-30, 2 * zA + 30)
    a1.set_xlabel("day", fontsize=6, labelpad=1)
    lab1 = label(a1, "gauge A prediction above DL (cm)")
    a1.axhline(zA, color=C["mute"], lw=0.5, ls=(0, (3, 3)))
    a1.axhline(0, color=C["ink"], lw=0.5)
    lsum, = a1.plot([], [], color=C["acc"], lw=0.8)
    COLS = [C["acc"], C["warn"], C["ok"], C["bad"]]
    xs = np.arange(4)
    a2.set_xlim(-0.6, 3.6); a2.set_ylim(0, 135); a2.set_xticks(xs); a2.set_xticklabels(CON)
    lab2 = label(a2, "amplitude (cm) · period (h)")
    bars = a2.bar(xs, [GA[n][0] for n in CON], color=C["hair"], width=0.6)
    btxt = [a2.text(i, GA[n][0] + 4, f"{PER[n]:.2f}", ha="center", fontsize=5.6, color=C["mute"]) for i, n in enumerate(CON)]
    a3.set_xlim(0, 15); a3.set_ylim(-140, 140); a3.set_xlabel("day", fontsize=6, labelpad=1)
    label(a3, "constituents (cm)")
    lsing = [a3.plot([], [], color=COLS[i], lw=0.6)[0] for i in range(4)]
    frames = []
    prev = np.full_like(full, zA)
    for i, n in enumerate(CON):
        bars[i].set_color(COLS[i])
        for j, w in enumerate(np.linspace(0.2, 1.0, 6)):
            lsing[i].set_data(days, single[i] * w)
            lsum.set_data(days, prev + (parts[i] - prev) * w)
            lab1.set_text("gauge A prediction above DL (cm) = MSL + " + " + ".join(CON[:i + 1]))
            rng_ = parts[i].max() - parts[i].min()
            status.set_text(f"+ {n}   range {rng_:,.0f} cm")
            frames.append(grab(fig))
        prev = parts[i]
    frames += [grab(fig)] * 6
    n_part1 = len(frames)
    if "--part1" in sys.argv:
        frames[n_part1 - 8].save(OUT / "tidebedpy-tide-part1.png")

    # --- part 2: a survey line run twice; Tc from the engine; soundings reduced to DL ----------
    def series(coef, name):                         # 10-min predicted series (KST), as a TOPS file would hold
        tk = pd.date_range("2026-05-08 00:00", "2026-05-09 00:00", freq="10min")
        h = LK.synth(coef, (tk - KST).to_numpy())
        s = TideSeries(station_name=name, records=[TideRecord(time=t.to_pydatetime(), level=float(v))
                                                    for t, v in zip(tk, h)])
        interpolate_akima(s)
        return s
    # two gauges 14 km west and east of the line; co-tidal values vary linearly between them
    LON_A, LON_B, LATG = 127.90, 128.20, 34.0
    stA = RefStation(name="Gauge A", longitude=LON_A, latitude=LATG, spr_range=2 * (GA["M2"][0] + GA["S2"][0]),
                     msl=zA, mhwi=6.9, tide_pred=series(cA, "Gauge A"))
    stB = RefStation(name="Gauge B", longitude=LON_B, latitude=LATG, spr_range=2 * (GB["M2"][0] + GB["S2"][0]),
                     msl=zB, mhwi=7.4, tide_pred=series(cB, "Gauge B"))

    class CoTidalStandIn:                           # answers like data_io.cotidal.CoTidalGrid.get_cotidal_values
        def get_cotidal_values(self, x, y):
            f = (x - LON_A) / (LON_B - LON_A)
            return (stA.spr_range + f * (stB.spr_range - stA.spr_range), zA + f * (zB - zA),
                    stA.mhwi + f * (stB.mhwi - stA.mhwi))
    cfg = TideBedConfig(tide_series_type="예측", is_kst=True, utc_offset=9.0, rank_limit=10)
    eng = TideCorrectionEngine(cfg, [stA, stB], CoTidalStandIn())

    rng = np.random.default_rng(2)
    L, NP = 1800.0, 90                              # line length (m), fixes per pass
    lon_s, lon_e, latl = 128.02, 128.02 + L / (111320 * np.cos(np.radians(LATG))), 34.012
    dist = np.linspace(0, L, NP)
    seabed = 38 + 6 * np.sin(dist / 420) + 2.5 * np.exp(-((dist - 1150) / 90) ** 2) + 0.004 * dist   # below DL (m)
    passes = [(datetime(2026, 5, 8, 9, 40), +1), (datetime(2026, 5, 8, 13, 10), -1)]   # KST, ~2 m/s
    dur = timedelta(seconds=L / 2.0)
    out = []
    for t_start, dirn in passes:
        dd = dist if dirn > 0 else dist[::-1]
        tt = [t_start + dur * (i / (NP - 1)) for i in range(NP)]
        tc, wts = [], []
        for d_, t_ in zip(dd, tt):
            nav = NavPoint(x=lon_s + (lon_e - lon_s) * d_ / L, y=latl, t=t_)
            tc.append(eng.process_nav_point(nav))  # cm above DL
            w_ = {c.station_name: c.weight for c in eng.last_corrections}
            wts.append((w_.get("Gauge A", 0.0), w_.get("Gauge B", 0.0)))
        tc = np.array(tc)
        surge = 0.06 + 0.02 * np.sin(np.arange(NP) / 25)                # weather residual the prediction misses (m)
        sb = np.interp(dd, dist, seabed)
        obs = sb + tc / 100 + surge + rng.normal(0, 0.05, NP)          # echo depth below the water surface
        out.append(dict(d=dd, t=tt, tc=tc, obs=obs, red=obs - tc / 100, w=wts))

    # re-lay the figure for part 2
    for a in (a1, a2, a3):
        a.cla(); style(a, grid=True)
    hours = lambda ts_: np.array([(t - datetime(2026, 5, 8)).total_seconds() / 3600 for t in ts_])
    tday = pd.date_range("2026-05-08 06:00", "2026-05-08 17:00", freq="5min")
    hA = LK.synth(cA, (tday - KST).to_numpy()); hB = LK.synth(cB, (tday - KST).to_numpy())
    th = (tday - pd.Timestamp("2026-05-08")).total_seconds().to_numpy() / 3600
    a1.set_xlim(6, 17); a1.set_ylim(-20, 2 * zA + 20)
    a1.set_xticks([6, 9, 12, 15]); a1.set_xticklabels(["06:00", "09:00", "12:00", "15:00"])
    label(a1, "predicted tide above DL (cm) · gauge A, gauge B · Tc at line")
    a1.plot(th, hA, color=C["mute"], lw=0.6); a1.plot(th, hB, color=C["mute"], lw=0.6, ls=(0, (3, 2)))
    a1.text(6.2, hA[0] + 8, "A", fontsize=6, color=C["mute"]); a1.text(6.2, hB[0] - 26, "B", fontsize=6, color=C["mute"])
    for t_start, _ in passes:
        h0 = (t_start - datetime(2026, 5, 8)).total_seconds() / 3600
        a1.axvspan(h0, h0 + dur.total_seconds() / 3600, color=C["acc"], alpha=0.07, lw=0)
    tcl = [a1.plot([], [], color=c_, lw=1.4)[0] for c_ in (C["acc"], C["warn"])]
    # map: gauges, line, vessel
    km = lambda lon: (lon - LON_A) * 111.32 * np.cos(np.radians(LATG))
    a2.set_xlim(-2, km(LON_B) + 2); a2.set_ylim(-4, 4); a2.set_aspect("auto")
    a2.set_yticks([]); a2.set_xlabel("east (km)", fontsize=6, labelpad=1)
    lab2 = label(a2, "IDW weights")
    a2.scatter([km(LON_A), km(LON_B)], [0, 0], marker="^", s=26, color=C["ink"], zorder=4)
    a2.text(km(LON_A), -1.6, "A", ha="center", fontsize=6, color=C["ink"])
    a2.text(km(LON_B), -1.6, "B", ha="center", fontsize=6, color=C["ink"])
    ly = (latl - LATG) * 111.32
    a2.plot([km(lon_s), km(lon_e)], [ly, ly], color=C["mute"], lw=0.8)
    boat, = a2.plot([], [], "o", ms=3.5, color=C["acc"], zorder=5)
    wl = [a2.plot([], [], color=C["acc"], lw=0.6, alpha=0.6)[0] for _ in range(2)]
    a3.set_xlim(-20, L + 20); a3.set_ylim(seabed.max() + 4.5, seabed.min() - 2.5)
    a3.set_xticks([0, 400, 800, 1200, 1600])
    a3.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    a3.set_xlabel("distance along line (m)", fontsize=6, labelpad=1)
    lab3 = label(a3, "depth (m) · observed below water surface (hollow) · reduced to DL (filled)")
    PC = [C["acc"], C["warn"]]
    obs_sc = [a3.scatter([], [], s=7, facecolors="none", edgecolors=PC[i], lw=0.5, alpha=0.8) for i in range(2)]
    red_sc = [a3.scatter([], [], s=9, color=PC[i], lw=0) for i in range(2)]
    STEP = 6
    for pi, P in enumerate(out):
        h = hours(P["t"])
        for kk in list(range(STEP, NP, STEP)) + [NP]:
            tcl[pi].set_data(h[:kk], P["tc"][:kk])
            obs_sc[pi].set_offsets(np.c_[P["d"][:kk], P["obs"][:kk]])
            red_sc[pi].set_offsets(np.c_[P["d"][:kk], P["red"][:kk]])
            j = kk - 1
            bx = km(lon_s + (lon_e - lon_s) * P["d"][j] / L)
            boat.set_data([bx], [ly])
            wA, wB = P["w"][j]                     # the engine's own IDW weights for this fix
            wl[0].set_data([km(LON_A), bx], [0, ly]); wl[1].set_data([km(LON_B), bx], [0, ly])
            wl[0].set_linewidth(0.3 + 2.2 * wA); wl[1].set_linewidth(0.3 + 2.2 * wB)
            lab2.set_text(f"IDW weights  A {wA:.2f} · B {wB:.2f}")
            status.set_text(f"pass {pi + 1}  {P['t'][j]:%H:%M}   Tc {P['tc'][j]:,.0f} cm")
            status.set_color(PC[pi])
            frames.append(grab(fig))
        frames += [grab(fig)] * 4
    # agreement of the two passes at the same distance, before and after reduction
    o2 = np.interp(dist, out[1]["d"][::-1], out[1]["obs"][::-1]); r2 = np.interp(dist, out[1]["d"][::-1], out[1]["red"][::-1])
    raw_gap = float(np.median(np.abs(out[0]["obs"] - o2))); red_gap = float(np.median(np.abs(out[0]["red"] - r2)))
    lab3.set_text(f"depth (m) · pass 1 vs 2, median gap: observed {raw_gap:.2f} m -> reduced to DL {red_gap:.2f} m")
    lab3.set_color(C["ink"])
    status.set_text(f"two passes agree to {red_gap * 100:,.0f} cm after Tc"); status.set_color(C["ok"])
    frames.append(grab(fig))
    write_gif(frames, "tidebedpy-tide.gif", hold=50, mid=n_part1 + 30)
    return dict(tcA=out[0]["tc"][[0, -1]], tcB=out[1]["tc"][[0, -1]], raw_gap=raw_gap, red_gap=red_gap,
                range15=float(full.max() - full.min()))


if __name__ == "__main__":
    which = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"mag", "tide"}
    if "mag" in which:
        print(magqc())
    if "tide" in which:
        print(tidebed())
