"""Two more profile animations, same style as make_visuals.py (palette, figure size, helpers copied below).

  sgycompare-headers.gif  SGYCompare  sgycompare/core.py::compare_segy (read_segy_info per file -> per-byte
                                        trace-header statistics, binary header diff, build_compare_story verdict)
                                      sgycompare/core.py::extract_coordinates (scalar applied, byte 71)
                                      sgycompare/core.py::detect_header_anomalies (sequence gaps)
                                      sgycompare/core.py::get_trace_statistics (per-trace amplitude stats)
                          Trace samples for the two sections and the single-trace overlay are read here with
                          segyio through core.open_segy (the desktop trace panel does the same in a Qt worker);
                          the position gap between the two files (matched by byte 1) is a subtraction done here.
  p190-receivers.gif      P190_NavConverter  p190converter/engine/geometry/gps_interpolation.py::compute_vessel_cog
                                      engine/geometry/interpolation.py::compute_heading, interpolate_receivers
                                        (method "linear" = straight chord, method "feathering" = t^alpha correction),
                                        compute_feathering_angle
                                      engine/crs/transformer.py::CRSTransformer.utm_to_latlon
                                      engine/writer/s_record.py::format_s_record, r_record.py::format_r_records
                          Per-shot loop mirrors engine/pipeline.py::run_style_a (source = front GPS, cable heading
                          front -> tail, vessel COG window 5); the tow-frame view (along / cross track) is a rotation
                          done here.

The input is synthetic and generated below. SGYCompare: a 400-trace UHR-style line (file A, IBM float, coordinate
scalar -100) and a re-export of it (file B: IEEE float, scalar -10, three traces dropped with the original numbering
kept, gain 0.5 and a 3 ms static). P190: an 80-shot line with a 96-channel streamer whose tail buoy drifts to
starboard as an invented cross-current grows. No survey data, line, vessel or project names are used.

usage:  python make_seis2_gif.py <out_dir> [sgycompare_repo] [p190_repo] [sgy,p190]
needs:  numpy, matplotlib, pillow, segyio, pyproj
"""
import sys
sys.dont_write_bytecode = True                     # import from the repos without leaving __pycache__ there
import math
import tempfile
import textwrap
from copy import copy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
SGYC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/Preprocessing/SGYCompare")
P190 = Path(sys.argv[3] if len(sys.argv) > 3 else "/home/user/repos/Preprocessing/P190_NavConverter")
LAST_DIR = OUT

# ---- style: copied from make_visuals.py ---------------------------------------------------
C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}")


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
# 1. SGYCompare - an original line and its re-export, compared byte field by byte field
# =========================================================================================
def sgycompare():
    sys.path[:0] = [str(SGYC)]
    import segyio
    from sgycompare import core

    rng = np.random.default_rng(11)
    NTR, NS, DT_US = 400, 600, 500                  # 400 traces, 600 samples at 0.5 ms (0.3 s)
    t = np.arange(NS) * DT_US / 1e6
    x = np.arange(NTR)

    def ricker(f, tt):
        a = (np.pi * f * tt) ** 2
        return (1 - 2 * a) * np.exp(-a)
    wav = ricker(140.0, (np.arange(61) - 30) * DT_US / 1e6)
    refl = np.zeros((NTR, NS))
    seabed = 0.060 + 0.012 * np.sin(x / 70.0) + 0.00004 * x
    layers = [(seabed, 1.0), (seabed + 0.030 + 0.00006 * x, -0.45), (seabed + 0.075 - 0.00004 * x, 0.35),
              (seabed + 0.120 + 0.010 * np.sin(x / 40.0), 0.28)]
    for tt0, amp in layers:
        k = np.clip(np.round(tt0 / (DT_US / 1e6)).astype(int), 0, NS - 1)
        refl[x, k] += amp
    for i in range(NTR):                            # one buried point diffractor
        tt = np.sqrt(0.105 ** 2 + ((i - 250) * 0.0012) ** 2)
        k = int(round(tt / (DT_US / 1e6)))
        if k < NS:
            refl[i, k] += 0.25
    traces_a = np.array([np.convolve(r, wav, mode="same") for r in refl]) + rng.normal(0, 0.02, (NTR, NS))
    traces_a = traces_a.astype(np.float32)

    E0, N0, AZ, DX = 451_200.00, 3_801_300.00, np.radians(62.0), 3.125     # synthetic UTM, not a real line
    ex = E0 + x * DX * np.sin(AZ) + 2.0 * np.sin(x / 55.0) * np.cos(AZ)
    ny = N0 + x * DX * np.cos(AZ) - 2.0 * np.sin(x / 55.0) * np.sin(AZ)

    DROP = {180, 181, 182}                          # file B: three traces lost in the re-export
    keep = [i for i in range(NTR) if i not in DROP]
    SHIFT = int(3000 / DT_US)                       # 3 ms static
    traces_b = np.zeros_like(traces_a)
    traces_b[:, SHIFT:] = 0.5 * traces_a[:, :-SHIFT]

    tmp = Path(tempfile.mkdtemp(prefix="sgyc_", dir=str(OUT)))
    pa, pb = tmp / "line_A_original.sgy", tmp / "line_B_reexport.sgy"

    def write(path, idx, data, fmt, scal):
        spec = segyio.spec(); spec.format = fmt; spec.samples = list(range(NS)); spec.tracecount = len(idx)
        with segyio.create(str(path), spec) as f:
            f.text[0] = segyio.tools.wrap("C 1 SYNTHETIC DEMO LINE - NOT SURVEY DATA")
            f.bin.update({segyio.BinField.Samples: NS, segyio.BinField.Interval: DT_US,
                          segyio.BinField.Format: fmt, segyio.BinField.SortingCode: 1})
            mult = -scal                                    # scalar -100 -> store x100
            for j, i in enumerate(idx):
                f.header[j] = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.TRACE_SEQUENCE_FILE: j + 1,
                    segyio.TraceField.FieldRecord: 1001 + i,
                    segyio.TraceField.TraceNumber: 1,
                    segyio.TraceField.EnergySourcePoint: 1001 + i,
                    segyio.TraceField.CDP: 2001 + i,
                    segyio.TraceField.CDP_TRACE: 1,
                    segyio.TraceField.TraceIdentificationCode: 1,
                    segyio.TraceField.ElevationScalar: 1,
                    segyio.TraceField.SourceGroupScalar: scal,
                    segyio.TraceField.SourceX: int(round(ex[i] * mult)),
                    segyio.TraceField.SourceY: int(round(ny[i] * mult)),
                    segyio.TraceField.GroupX: int(round(ex[i] * mult)),
                    segyio.TraceField.GroupY: int(round(ny[i] * mult)),
                    segyio.TraceField.CoordinateUnits: 1,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: NS,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: DT_US,
                }
                f.trace[j] = data[i]
    write(pa, list(range(NTR)), traces_a, 1, -100)
    write(pb, keep, traces_b, 5, -10)

    # ---- the program's own comparison -------------------------------------------------------
    res = core.compare_segy(str(pa), str(pb))
    story = res.story
    coord_a = core.extract_coordinates(str(pa)); coord_b = core.extract_coordinates(str(pb))
    anom_b = core.detect_header_anomalies(str(pb))
    stat_a = core.get_trace_statistics(str(pa), max_traces=NTR)
    stat_b = core.get_trace_statistics(str(pb), max_traces=NTR)
    with core.open_segy(str(pa), "r", ignore_geometry=True) as f:
        sec_a = np.stack([np.asarray(tr, dtype=float) for tr in f.trace])
        seq_a = np.array([h[segyio.TraceField.TRACE_SEQUENCE_LINE] for h in f.header])
    with core.open_segy(str(pb), "r", ignore_geometry=True) as f:
        sec_b = np.stack([np.asarray(tr, dtype=float) for tr in f.trace])
        seq_b = np.array([h[segyio.TraceField.TRACE_SEQUENCE_LINE] for h in f.header])
    for p_ in (pa, pb):
        p_.unlink()                                   # my own temporary synthetic files
    tmp.rmdir()

    rows = {}
    for e in res.matched_fields:
        rows[e["byte_pos"]] = (e["field"], False, e.get("impact", ""), e.get("delta_summary", ""))
    for e in res.trace_header_diffs:
        rows[e["byte_pos"]] = (e["field"], True, e.get("impact", ""), e.get("delta_summary", ""))
    ladder = [(f"{bp:3d}", *rows[bp]) for bp in sorted(rows)]
    bins = {d["field"]: d for d in res.binary_diffs}
    for k in core.BINARY_HEADER_FIELDS:
        if k in bins:
            d = bins[k]
            ladder.append(("bin", k, True, d.get("impact", "High"), f"{d['file_a']} -> {d['file_b']}"))
        else:
            ladder.append(("bin", k, False, "", ""))
    ca, cb = np.array(coord_a["coordinates"]), np.array(coord_b["coordinates"])
    pos_a = {int(s): ca[i] for i, s in enumerate(seq_a)}
    gaps = np.array([np.hypot(*(cb[i] - pos_a[int(s)])) for i, s in enumerate(seq_b)])
    gap_items = [a for a in anom_b["anomalies"] if a["type"] == "sequence_gap"]
    n_tr = len(res.trace_header_diffs) + len(res.matched_fields)

    # ---- figure -----------------------------------------------------------------------------
    fig, status = fig_base(("SGYCompare", "original vs re-export, byte field by field"),
                           ["compare_segy (read_segy_info, build_compare_story) · extract_coordinates · "
                            "detect_header_anomalies · get_trace_statistics",
                            "synthetic SEG-Y pair written here; sections read here with segyio; position gap = "
                            "B - A by byte 1, done here"])
    aA = fig.add_axes([0.045, 0.43, 0.155, 0.43]); style(aA)
    aB = fig.add_axes([0.215, 0.43, 0.155, 0.43]); style(aB)
    aT = fig.add_axes([0.045, 0.15, 0.325, 0.19]); style(aT, grid=True)
    aL = fig.add_axes([0.395, 0.15, 0.305, 0.71]); aL.axis("off"); aL.set_xlim(0, 1); aL.set_ylim(0, 1)
    aM = fig.add_axes([0.745, 0.50, 0.24, 0.36]); style(aM)
    aV = fig.add_axes([0.725, 0.11, 0.26, 0.31]); aV.axis("off"); aV.set_xlim(0, 1); aV.set_ylim(0, 1)

    clip = float(np.percentile(np.abs(sec_a), 99))
    tmax = t[-1] * 1000
    gkw = dict(aspect="auto", cmap="Greys", vmin=-clip, vmax=clip, interpolation="bilinear")
    blank = np.full((NS, NTR), np.nan)
    imA = aA.imshow(blank.copy(), extent=(0.5, NTR + 0.5, tmax, 0), **gkw)
    imB = aB.imshow(blank[:, :len(keep)].copy(), extent=(0.5, len(keep) + 0.5, tmax, 0), **gkw)
    for ax_, nm in ((aA, "A  original"), (aB, "B  re-export")):
        label(ax_, nm); ax_.set_xticks([1, 200, 400]); ax_.tick_params(labelsize=5.5)
    aA.set_ylabel("ms", fontsize=6, labelpad=1); aB.set_yticklabels([])
    TR = 120                                         # the trace overlaid below
    aT.set_xlim(40, 140); yl = 1.1 * float(np.abs(sec_a[TR]).max()); aT.set_ylim(-yl, yl)
    aT.set_yticks([]); aT.tick_params(labelsize=5.5)
    lT = label(aT, f"trace {TR + 1} (ms)  A black · B red")
    lnA, = aT.plot([], [], color=C["ink"], lw=0.7); lnB, = aT.plot([], [], color=C["bad"], lw=0.7)

    label(aL, "header bytes (compare_segy)")
    NR = len(ladder); dy = 0.98 / NR
    ltxt, lmark = [], []
    for i, (bp, name, diff, imp, delta) in enumerate(ladder):
        y = 0.985 - (i + 0.5) * dy
        ltxt.append(aL.text(0.0, y, f"{bp} {name[:22]}", fontsize=5.3, color=C["hair"], va="center"))
        lmark.append(aL.text(1.0, y, "", fontsize=5.3, color=C["hair"], va="center", ha="right"))
    IMPC = {"High": C["bad"], "Medium": C["warn"], "Low": C["warn"]}

    aM.ticklabel_format(useOffset=False, style="plain")
    aM.set_xlim(ca[:, 0].min() - 60, ca[:, 0].max() + 60); aM.set_ylim(ca[:, 1].min() - 60, ca[:, 1].max() + 60)
    aM.set_xticks([]); aM.set_yticks([])
    lM = label(aM, "positions, scalar applied")
    mA, = aM.plot([], [], color=C["mute"], lw=2.6, alpha=0.35, solid_capstyle="round")
    mB, = aM.plot([], [], ls="", marker=".", ms=1.4, color=C["acc"])
    mG = aM.scatter([], [], s=30, facecolors="none", edgecolors=C["bad"], lw=0.9, zorder=5)
    mGt = aM.text(0, 0, "", fontsize=5.5, color=C["bad"], ha="left", va="top")

    vt = [aV.text(0.0, 0.93 - i * 0.155, "", fontsize=6.0, color=C["ink"], va="top") for i in range(6)]
    frames = []

    # phase 1: both files read (sections fill in)
    for k in np.linspace(10, NTR, 18).astype(int):
        da = blank.copy(); da[:, :k] = sec_a[:k].T; imA.set_data(da)
        kb = min(len(keep), k); db = blank[:, :len(keep)].copy(); db[:, :kb] = sec_b[:kb].T; imB.set_data(db)
        status.set_text(f"read_segy_info  A {min(k, NTR):,} / B {kb:,} traces"); status.set_color(C["acc"])
        frames.append(grab(fig))
    lnA.set_data(t * 1000, sec_a[TR]); lnB.set_data(t * 1000, sec_b[list(seq_b).index(TR + 1)])
    frames += [grab(fig)] * 4

    # phase 2: field by field
    nd = 0
    for i, (bp, name, diff, imp, delta) in enumerate(ladder):
        ltxt[i].set_color(C["ink"] if diff else C["mute"])
        if diff:
            nd += 1
            lmark[i].set_text(f"differs {imp.lower()}" if imp else "differs")
            lmark[i].set_color(IMPC.get(imp, C["bad"]))
        else:
            lmark[i].set_text("same"); lmark[i].set_color(C["ok"])
        status.set_text(f"byte {bp.strip()} {name}: {'differs' if diff else 'same'}")
        status.set_color(C["bad"] if diff else C["ok"])
        if diff and delta:
            vt[0].set_text(name); vt[0].set_color(C["bad"])
            dl = "\n".join(textwrap.wrap(delta.replace(" -> ", " ->| ").replace("; ", " "), 31,
                                         break_on_hyphens=False)[:5]).replace("|", "")
            vt[1].set_text(dl); vt[1].set_color(C["mute"]); vt[1].set_fontsize(5.4)
        frames.append(grab(fig))
        if diff:
            frames.append(grab(fig))
    vt[0].set_text(""); vt[1].set_text("")

    # phase 3: is the geometry moved, or only rescaled?
    for k in np.linspace(20, NTR, 12).astype(int):
        mA.set_data(ca[:k, 0], ca[:k, 1])
        kb = int(np.searchsorted(seq_b, k))
        mB.set_data(cb[:kb, 0], cb[:kb, 1])
        status.set_text("extract_coordinates  A and B (byte 71 scalar applied)"); status.set_color(C["acc"])
        frames.append(grab(fig))
    if gap_items:
        gi = gap_items[0]; j = gi.get("trace_index", None)
        if j is None:
            j = int(np.argmax(np.diff(seq_b) > 1)) + 1
        gx, gy = cb[j - 1]
        mG.set_offsets([[gx, gy]])
        mGt.set_position((gx + 25, gy - 20))
        mGt.set_text(f"gap {seq_b[j - 1]} -> {seq_b[j]}")
    lM.set_text(f"scaled: B - A max {gaps.max():.2f} m")
    lM.set_color(C["ok"])
    status.set_text(f"raw SourceX x10 apart, same place: max {gaps.max():.2f} m"); status.set_color(C["ok"])
    frames += [grab(fig)] * 10
    rms_ratio = stat_b["rms"] / stat_a["rms"]
    lT.set_text(f"RMS B/A {rms_ratio:.2f} (get_trace_statistics)")
    lT.set_color(C["ink"])
    frames += [grab(fig)] * 6

    # phase 4: the program's verdict
    SCOL = {"ALIGNED": C["ok"], "CHECK": C["warn"], "REVIEW": C["bad"]}
    sev = anom_b["severity"]
    lines_v = [
        (f"{story['status']}", SCOL.get(story["status"], C["ink"]), 9),
        (f"{len(res.trace_header_diffs)} / {n_tr} header fields differ", C["ink"], 6.0),
        (f"binary: format {res.file_a.data_format} -> {res.file_b.data_format}", C["ink"], 6.0),
        (f"traces {res.file_a.trace_count:,} -> {res.file_b.trace_count:,}", C["ink"], 6.0),
        (f"B: {anom_b['anomaly_count']} anomalies ({sev})", C["bad"] if sev != "OK" else C["ok"], 6.0),
        (f"top: {story['top_findings'][0].get('field')}" if story["top_findings"] else "", C["mute"], 6.0),
    ]
    for i, (txt, col, fs) in enumerate(lines_v):
        vt[i].set_text(txt); vt[i].set_color(col); vt[i].set_fontsize(fs)
        if i == 0:
            vt[i].set_fontweight("bold")
        frames += [grab(fig)] * 2
    status.set_text(f"build_compare_story -> {story['status']}"); status.set_color(SCOL.get(story["status"], C["ink"]))
    frames.append(grab(fig))
    write_gif(frames, "sgycompare-headers.gif", hold=50, mid=18 + 4 + 2 * nd + (NR - nd) - 4)
    return dict(status=story["status"], diffs=len(res.trace_header_diffs), fields=n_tr,
                binary=len(res.binary_diffs), gap_max=float(gaps.max()), rms_ratio=rms_ratio,
                anomalies=anom_b["anomaly_count"], severity=sev, headline=story["headline"],
                insight=story["insight_lines"])


# =========================================================================================
# 2. P190_NavConverter - streamer receivers placed shot by shot, then S / R records
# =========================================================================================
def p190():
    sys.path[:0] = [str(P190)]
    from p190_navconverter.p190converter.engine.geometry.interpolation import (
        compute_heading, interpolate_receivers, compute_feathering_angle)
    from p190_navconverter.p190converter.engine.geometry.gps_interpolation import compute_vessel_cog
    from p190_navconverter.p190converter.engine.crs.transformer import CRSTransformer
    from p190_navconverter.p190converter.engine.writer.s_record import format_s_record
    from p190_navconverter.p190converter.engine.writer.r_record import format_r_records
    from p190_navconverter.p190converter.models.shot_gather import ShotGather
    from p190_navconverter.p190converter.models.survey_config import MarineGeometry, CRSConfig

    rng = np.random.default_rng(3)
    NSHOT, SP = 80, 12.5                            # 80 shots, 12.5 m shot interval
    s = np.arange(NSHOT)
    HDG = 75.0 + 1.2 * np.sin(s / 13.0)             # vessel heading weaves about 75 deg
    E0, N0 = 448_300.0, 3_802_100.0                 # synthetic UTM 52N, open water
    hr = np.radians(HDG)
    fe = E0 + np.cumsum(SP * np.sin(hr)) + rng.normal(0, 0.15, NSHOT)    # front GPS = source
    fn = N0 + np.cumsum(SP * np.cos(hr)) + rng.normal(0, 0.15, NSHOT)
    geom = MarineGeometry(rx1_dx=0.0, rx1_dy=-50.0, n_channels=96, rx_interval=6.25,
                          cable_depth=3.0, interp_method="feathering", feathering_alpha=2.0)
    LCH = 50.0 + geom.total_spread + 20.0           # front GPS -> tail buoy chord
    theta = 3.0 + 7.5 / (1 + np.exp(-(s - 42) / 8.0))     # invented cross-current: drift grows 3 -> 10 deg
    # tail buoy: the chord swung to starboard by theta from straight astern (synthetic input)
    aft = hr + np.pi - np.radians(theta)            # starboard drift rotates the aft bearing CCW
    te = fe + LCH * np.sin(aft) + rng.normal(0, 0.3, NSHOT)
    tn = fn + LCH * np.cos(aft) + rng.normal(0, 0.3, NSHOT)

    cog = compute_vessel_cog(fe, fn, window=5)
    lin = copy(geom); lin.interp_method = "linear"
    tr = CRSTransformer(CRSConfig.from_zone(52))
    shots, rl, rf, fang = [], [], [], []
    for i in range(NSHOT):
        hd = compute_heading(fe[i], fn[i], te[i], tn[i])
        rf_i = interpolate_receivers(fe[i], fn[i], hd, geom, head_x=fe[i], head_y=fn[i],
                                     tail_x=te[i], tail_y=tn[i], vessel_heading_deg=float(cog[i]))
        rl_i = interpolate_receivers(fe[i], fn[i], hd, lin)
        lat, lon = tr.utm_to_latlon(fe[i], fn[i])
        sec = 9 * 3600 + 12 * 60 + int(i * SP / 2.3)     # about 2.3 m/s
        sh = ShotGather(ffid=2001 + i, source_x=fe[i], source_y=fn[i], receivers=rf_i, day=158,
                        hour=sec // 3600, minute=(sec // 60) % 60, second=sec % 60, source_depth=2.0,
                        heading=hd, line_name="LINE-A", source_lat=lat, source_lon=lon)
        shots.append(sh); rf.append(rf_i); rl.append(rl_i)
        fang.append(compute_feathering_angle(fe[i], fn[i], te[i], tn[i], float(cog[i])))
    fang = np.array(fang)
    srec = [format_s_record(sh) for sh in shots]
    rrec = [format_r_records(sh) for sh in shots]

    def tow_frame(i, rx):                           # along (m aft of source) / cross (m, + starboard)
        h = math.radians(cog[i])
        dE = np.array([r.x for r in rx]) - fe[i]; dN = np.array([r.y for r in rx]) - fn[i]
        along = -(dE * math.sin(h) + dN * math.cos(h))
        cross = dE * math.cos(h) - dN * math.sin(h)
        return along, cross
    corr = []
    for i in range(NSHOT):
        _, cf = tow_frame(i, rf[i]); _, cl = tow_frame(i, rl[i])
        corr.append(np.max(np.abs(cf - cl)))
    corr = np.array(corr)

    fig, status = fig_base(("P190_NavConverter", "receivers per shot -> P1/90"),
                           ["compute_vessel_cog · compute_heading · interpolate_receivers (linear / feathering) · "
                            "compute_feathering_angle",
                            "format_s_record · format_r_records  ·  synthetic line, buoy GPS and cross-current; "
                            "tow-frame rotation done here"])
    aP = fig.add_axes([0.045, 0.45, 0.27, 0.41]); style(aP)
    aF = fig.add_axes([0.36, 0.45, 0.33, 0.41]); style(aF, grid=True)
    aG = fig.add_axes([0.74, 0.45, 0.245, 0.41]); style(aG, grid=True)
    aR = fig.add_axes([0.045, 0.115, 0.94, 0.26]); aR.set_facecolor(C["bg"])
    for s_ in aR.spines.values():
        s_.set_color(C["hair"])
    aR.set_xticks([]); aR.set_yticks([]); aR.set_xlim(0, 1); aR.set_ylim(0, 1)

    allE = np.r_[fe, te]; allN = np.r_[fn, tn]
    pad = 40
    cx, cy = (allE.min() + allE.max()) / 2, (allN.min() + allN.max()) / 2
    hw = max((allE.max() - allE.min()) / 2, (allN.max() - allN.min()) / 2 * 1.405) + pad
    aP.set_xlim(cx - hw, cx + hw); aP.set_ylim(cy - hw / 1.405, cy + hw / 1.405)
    aP.set_xticks([]); aP.set_yticks([])
    label(aP, "plan · source track + streamer")
    trk, = aP.plot([], [], color=C["ink"], lw=0.9)
    old = [aP.plot([], [], color=C["hair"], lw=0.7)[0] for _ in range(8)]
    cab, = aP.plot([], [], color=C["acc"], lw=1.2)
    boat, = aP.plot([], [], marker="o", ms=3.2, color=C["ink"], lw=0)
    tailm, = aP.plot([], [], marker="s", ms=2.6, color=C["warn"], lw=0)

    aF.set_xlim(-10, LCH + 15); aF.set_ylim(-12, 125)
    aF.set_xlabel("m aft of source (tow frame)", fontsize=6, labelpad=1)
    aF.set_ylabel("m to starboard", fontsize=6, labelpad=1)
    lF = label(aF, "")
    aF.axhline(0, color=C["mute"], lw=0.6, ls=(0, (3, 3)))
    fl_lin, = aF.plot([], [], ls="", marker="o", ms=1.6, mfc="none", mec=C["mute"], mew=0.5)
    fl_fea, = aF.plot([], [], ls="", marker="o", ms=1.8, color=C["acc"])
    fl_tail, = aF.plot([], [], marker="s", ms=3.0, color=C["warn"], lw=0)
    fl_grp = aF.scatter([], [], s=16, facecolors="none", edgecolors=C["bad"], lw=0.8, zorder=6)
    aF.text(0.03, 0.93, "o linear chord   • feathering t^2", transform=aF.transAxes, ha="left", va="top",
            fontsize=5.5, color=C["mute"])

    aG.set_xlim(2001, 2001 + NSHOT - 1); aG.set_ylim(0, 12)
    aG.set_xlabel("shot (FFID)", fontsize=6, labelpad=1)
    aG.xaxis.set_major_formatter(COMMA)
    lG = label(aG, "feathering (deg)")
    gl, = aG.plot([], [], color=C["acc"], lw=1.0)
    gd, = aG.plot([], [], marker="o", ms=2.5, color=C["acc"], lw=0)

    hdr = aR.text(0.008, 0.86, "", fontsize=6.3, color=C["mute"], transform=aR.transAxes)
    rtx = [aR.text(0.008, 0.66 - j * 0.19, "", fontsize=6.6, color=C["ink"], family="DejaVu Sans Mono",
                   transform=aR.transAxes) for j in range(4)]

    frames = []
    ffid = np.arange(2001, 2001 + NSHOT)
    for i in range(NSHOT):
        trk.set_data(fe[:i + 1], fn[:i + 1]); boat.set_data([fe[i]], [fn[i]])
        rx = rf[i]
        cab.set_data([fe[i]] + [r.x for r in rx], [fn[i]] + [r.y for r in rx])
        tailm.set_data([te[i]], [tn[i]])
        for j, o in enumerate(old):
            k = i - (j + 1) * 10
            if k >= 0:
                o.set_data([fe[k]] + [r.x for r in rf[k]], [fn[k]] + [r.y for r in rf[k]])
        al, cl = tow_frame(i, rl[i]); af, cf = tow_frame(i, rf[i])
        fl_lin.set_data(al, cl); fl_fea.set_data(af, cf)
        h = math.radians(cog[i]); dE = te[i] - fe[i]; dN = tn[i] - fn[i]
        fl_tail.set_data([-(dE * math.sin(h) + dN * math.cos(h))], [dE * math.cos(h) - dN * math.sin(h)])
        fl_grp.set_offsets(np.c_[af[:3], cf[:3]])
        lF.set_text(f"96 ch · max shift from chord {corr[i]:4.1f} m")
        gl.set_data(ffid[:i + 1], fang[:i + 1]); gd.set_data([ffid[i]], [fang[i]])
        lG.set_text(f"feathering (deg) {fang[i]:+5.1f}")
        hdr.set_text(f"format_s_record / format_r_records  ->  FFID {ffid[i]:,}: 1 S + {len(rrec[i])} R lines "
                     "(80 columns, 3 channels per R line)")
        rtx[0].set_text(srec[i]); rtx[0].set_color(C["ink"])
        rtx[1].set_text(rrec[i][0]); rtx[1].set_color(C["bad"])
        rtx[2].set_text(rrec[i][1]); rtx[2].set_color(C["mute"])
        rtx[3].set_text(f"... {len(rrec[i]) - 2} more R lines to channel 96"); rtx[3].set_color(C["mute"])
        status.set_text(f"shot {i + 1:2d}/{NSHOT}  COG {cog[i]:5.1f}")
        status.set_color(C["acc"])
        frames.append(grab(fig))
    n_r = sum(len(r) for r in rrec)
    status.set_text(f"{NSHOT} S + {n_r:,} R records written")
    status.set_color(C["ok"])
    lF.set_text(f"line max shift from chord {corr.max():4.1f} m")
    lF.set_color(C["ink"])
    frames.append(grab(fig))
    write_gif(frames, "p190-receivers.gif", hold=50, mid=46)
    return dict(n_s=NSHOT, n_r=n_r, f0=float(fang[0]), f1=float(fang[-1]), corr_max=float(corr.max()),
                s0=srec[0], r0=rrec[0][0])


if __name__ == "__main__":
    which = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"sgy", "p190"}
    if "sgy" in which:
        print(sgycompare())
    if "p190" in which:
        print(p190())
