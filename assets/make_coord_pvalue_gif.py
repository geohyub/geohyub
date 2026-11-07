"""Two more profile animations, same style as make_visuals.py (palette, figure size, helpers copied below).

  coordconverter-datum.gif   CoordConverter  coordconverter/core.py
                               get_utm_zone             (zone of every grid point)
                               convert_batch            (Korean 1985 4162 -> WGS 84 4326, EPSG default and
                                                         datum_mode="sonarwiz_tokyo_m"; WGS 84 -> KGD2002 5186 -> back)
                               calculate_distance       (shift length, method gap, round-trip error)
                               calculate_endpoint       (survey line from a start point + bearing + distance)
                               calculate_grid_convergence (UTM scale factor along the line, zone 51 and 52)
                               get_transform_accuracy   (PROJ accuracy of the pair)
  pvalue-montecarlo.gif      PValue  pvalue/models.py::generate_block_tasks_from_distance
                               pvalue/data.py::validate_metocean
                               pvalue/simulation.py::simulate_campaign, simulate_single_detailed, summarize_pxx
                               pvalue/stats.py::percentile_ci

Everything plotted (zones, shifts, scale factors, converted coordinates, completion times, P50 / P90) comes back
from those functions. The input is synthetic: a regular lon/lat grid and a straight test line in open water
(no coastline, no project), and a 10-year hourly Hs / wind record generated below (seasonal mean, correlated
noise and storms), not a real site. Done in this script, not by the repos: arrow scaling, the cumulative
prefixes of the 1,000-run result table that make the animation, and the drawing.

usage:  python make_coord_pvalue_gif.py <out_dir> [coordconverter_repo] [pvalue_repo] [coord,pvalue]
needs:  numpy, pandas, matplotlib, pillow, pyproj
"""
import sys
sys.dont_write_bytecode = True                     # import from the repos without leaving __pycache__ there
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
COORD = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/Preprocessing/CoordConverter")
PVAL = Path(sys.argv[3] if len(sys.argv) > 3 else "/home/user/repos/Utilities/PValue")
WHICH = (sys.argv[4] if len(sys.argv) > 4 else "coord,pvalue").split(",")
LAST_DIR = OUT
logging.disable(logging.WARNING)

# ---- style: copied from make_visuals.py ---------------------------------------------------
C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
RC = {"font.family": "DejaVu Sans Mono", "font.size": 7, "axes.unicode_minus": True,
      "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
      "xtick.color": C["mute"], "ytick.color": C["mute"]}
plt.rcParams.update(RC)
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
# 1. CoordConverter - zones, datum shift, scale factor, round trip
# =========================================================================================
def coord():
    sys.path[:0] = [str(COORD)]
    import coordconverter.core as cc

    # ---- synthetic grid and test line (open water, no coastline) ----
    lons = np.arange(124.5, 131.01, 0.5); lats = np.arange(33.5, 38.51, 0.5)
    G = [(float(x), float(y)) for y in lats for x in lons]
    zone = np.array([cc.get_utm_zone(x, y)[0] for x, y in G])

    k85 = cc.convert_batch(G, 4162, 4326)                                   # Korean 1985 -> WGS 84 (PROJ default)
    tkm = cc.convert_batch(G, 4162, 4326, datum_mode=cc.DATUM_MODE_SONARWIZ)  # SonarWiz TOKYO-M 3-parameter
    shift = [cc.calculate_distance(x, y, r.x, r.y) for (x, y), r in zip(G, k85)]
    shift_m = np.array([s.distance_m for s in shift])
    gap_m = np.array([cc.calculate_distance(a.x, a.y, b.x, b.y).distance_m for a, b in zip(k85, tkm)])
    acc85 = cc.get_transform_accuracy(4162, 4326)
    acc5186 = cc.get_transform_accuracy(4326, 5186)
    SC = 100.0                                                               # arrow exaggeration (drawing only)
    dx = np.array([(r.x - x) * SC for (x, y), r in zip(G, k85)])
    dy = np.array([(r.y - y) * SC for (x, y), r in zip(G, k85)])

    LAT0, LON0, BRG, LEN = 35.7, 125.15, 82.0, 150_000.0                     # Line A: start, bearing, length
    dist = np.linspace(0, LEN, 61)
    ends = [cc.calculate_endpoint(LAT0, LON0, BRG, float(d)) for d in dist]
    llon = np.array([e.lon for e in ends]); llat = np.array([e.lat for e in ends])
    k51 = np.array([(cc.calculate_grid_convergence(x, y, zone=51).scale_factor - 1) * 1e6 for x, y in zip(llon, llat)])
    k52 = np.array([(cc.calculate_grid_convergence(x, y, zone=52).scale_factor - 1) * 1e6 for x, y in zip(llon, llat)])
    lz = np.array([cc.get_utm_zone(x, y)[0] for x, y in zip(llon, llat)])
    ix = [0, 20, 40, 60]                                                     # table rows: 0 / 50 / 100 / 150 km
    P = [(float(llon[i]), float(llat[i])) for i in ix]
    tm = cc.convert_batch(P, 4326, 5186)
    back = cc.convert_batch([(r.x, r.y) for r in tm], 5186, 4326)
    rt_mm = [cc.calculate_distance(a[0], a[1], b.x, b.y).distance_m * 1000 for a, b in zip(P, back)]
    print(f"shift {shift_m.min():,.1f}-{shift_m.max():,.1f} m, TOKYO-M gap {gap_m.min():.2f}-{gap_m.max():.2f} m,"
          f" acc85 {acc85}, acc5186 {acc5186}, k51 {k51.min():.0f}..{k51.max():.0f}, k52 {k52.min():.0f}..{k52.max():.0f},"
          f" rt max {max(rt_mm):.6f} mm")

    foot = ["coordconverter.core: get_utm_zone, convert_batch (4162->4326 default + TOKYO-M, 4326<->5186), calculate_distance,",
            "calculate_endpoint, calculate_grid_convergence, get_transform_accuracy. Synthetic grid and test line; arrows x100 here."]
    fig, status = fig_base(("CoordConverter", "datum shift and UTM scale"), foot)
    am = fig.add_axes([0.055, 0.17, 0.41, 0.70]); style(am)
    am.set_xlim(124.1, 131.4); am.set_ylim(33.15, 39.15)
    am.set_xticks(range(125, 132, 2)); am.set_yticks(range(34, 39, 2))
    am.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}°E"))
    am.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}°N"))
    lm = label(am, "")
    ak = fig.add_axes([0.575, 0.57, 0.405, 0.30]); style(ak, grid=True)
    ak.set_xlim(0, LEN / 1000); pad = 40
    ak.set_ylim(min(k51.min(), k52.min()) - pad, max(k51.max(), k52.max()) + pad)
    ak.xaxis.set_major_formatter(COMMA); ak.yaxis.set_major_formatter(COMMA)
    ak.set_xlabel("Line A distance (km)", fontsize=6, labelpad=1)
    lk = label(ak, "")
    at = fig.add_axes([0.575, 0.12, 0.405, 0.33]); at.axis("off")
    at.set_xlim(0, 1); at.set_ylim(0, 1)

    zc = {51: C["acc"], 52: C["ok"]}
    frames = []
    # A: grid points by UTM zone (14 frames)
    pts = am.scatter([], [], s=5)
    for f in range(14):
        n = int(len(G) * (f + 1) / 14)
        pts.set_offsets(np.array(G[:n]) if n else np.empty((0, 2)))
        pts.set_color([zc[z] for z in zone[:n]])
        lm.set_text("grid points coloured by get_utm_zone")
        status.set_text(f"zone of {n:,} points")
        if f == 13:
            am.axvline(126.0, color=C["ink"], lw=0.8)
            am.text(125.92, 39.08, "51N", ha="right", va="top", color=C["acc"], fontsize=6.5)
            am.text(126.08, 39.08, "52N", ha="left", va="top", color=C["ok"], fontsize=6.5)
            for cm, code in ((125, 5185), (127, 5186), (129, 5187), (131, 5188)):
                am.axvline(cm, color=C["mute"], lw=0.5, ls=(0, (2, 2)))
                am.text(cm, 33.2, f"{code}", ha="center", va="bottom", color=C["mute"], fontsize=5.5,
                        bbox=dict(fc=C["bg"], ec="none", pad=0.6))
        frames.append(grab(fig))
    for _ in range(4):
        frames.append(grab(fig))
    # B: datum shift arrows grow (22 frames)
    pts.set_sizes([2.5] * len(G))
    q = None
    for f in range(22):
        s = min(1.0, (f + 1) / 16)
        if q is not None:
            q.remove()
        q = am.quiver([g[0] for g in G], [g[1] for g in G], dx * s, dy * s, angles="xy", scale_units="xy",
                      scale=1, color=C["warn"], width=0.004, headwidth=3, headlength=3.5)
        lm.set_text("Korean 1985 -> WGS 84 shift (x100), dashed = TM belts")
        status.set_text(f"shift {shift_m.min() * s:,.0f} - {shift_m.max() * s:,.0f} m")
        frames.append(grab(fig))
    # C: Line A and scale factor along it (30 frames)
    lk.set_text("scale error along Line A (ppm = mm per km)")
    ak.axhline(0, color=C["mute"], lw=0.6)
    (l51,) = ak.plot([], [], color=C["acc"], lw=1.2)
    (l52,) = ak.plot([], [], color=C["ok"], lw=1.2)
    (lline,) = am.plot([], [], color=C["bad"], lw=1.6)
    xcross = dist[np.argmax(lz == 52)] / 1000
    for f in range(30):
        n = max(2, int(len(dist) * (f + 1) / 30))
        lline.set_data(llon[:n], llat[:n])
        l51.set_data(dist[:n] / 1000, k51[:n]); l52.set_data(dist[:n] / 1000, k52[:n])
        status.set_text(f"Line A {dist[n - 1] / 1000:,.0f} km, {lz[n - 1]}N")
        if f == 0:
            am.text(llon[0], llat[0] + 0.16, "Line A", color=C["bad"], fontsize=6, ha="left",
                        bbox=dict(fc=C["bg"], ec="none", pad=0.5))
        if f == 29:
            ak.axvline(xcross, color=C["ink"], lw=0.6, ls=(0, (2, 2)))
            ak.text(xcross + 2, ak.get_ylim()[1] - 15, "126°E", va="top", color=C["ink"], fontsize=5.5)
            ak.text(4, k51[0] + 18, "UTM 51N", color=C["acc"], fontsize=6, va="bottom")
            ak.text(LEN / 1000 - 4, k52[-1] + 18, "UTM 52N", color=C["ok"], fontsize=6, va="bottom", ha="right")
        frames.append(grab(fig))
    for _ in range(4):
        frames.append(grab(fig))
    # D: converted table (4 rows) and summary
    hdr = f"{'km':>4}  {'E 5186 (m)':>12}  {'N 5186 (m)':>12}  {'back':>8}"
    at.text(0.0, 0.97, "Line A: WGS 84 -> KGD2002 5186 -> WGS 84", color=C["mute"], fontsize=6.3, va="top")
    at.text(0.0, 0.84, hdr, color=C["mute"], fontsize=5.9, va="top")
    at.plot([0, 1], [0.745, 0.745], color=C["hair"], lw=0.6)
    y = 0.70
    for r in range(4):
        i = ix[r]
        row = f"{dist[i] / 1000:>4,.0f}  {tm[r].x:>12,.3f}  {tm[r].y:>12,.3f}  {rt_mm[r]:>5.3f} mm"
        at.text(0.0, y, row, color=C["ink"], fontsize=5.9, va="top")
        y -= 0.12
        status.set_text(f"convert_batch row {r + 1}/4")
        for _ in range(3):
            frames.append(grab(fig))
    acc_t = (lambda a: "unknown" if a is None else f"{a:,.1f} m")
    at.text(0.0, 0.17, f"Korean 1985 shift {shift_m.min():,.0f}-{shift_m.max():,.0f} m, PROJ acc. {acc_t(acc85)}",
            color=C["warn"], fontsize=6, va="top")
    at.text(0.0, 0.05, f"EPSG default vs TOKYO-M: {gap_m.min():.1f} - {gap_m.max():.1f} m apart", color=C["bad"],
            fontsize=6, va="top")
    status.set_text("round trip < 0.001 mm" if max(rt_mm) < 0.001 else f"round trip {max(rt_mm):.3f} mm")
    status.set_color(C["ok"])
    for _ in range(3):
        frames.append(grab(fig))
    write_gif(frames, "coordconverter-datum.gif", hold=40, mid=50)


# =========================================================================================
# 2. PValue - Monte Carlo of a survey campaign under weather downtime
# =========================================================================================
def synth_metocean(seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", "2024-12-31 23:00", freq="h")
    n = len(idx); doy = idx.dayofyear.to_numpy(); t = np.arange(n)
    base = 1.05 + 0.55 * np.cos(2 * np.pi * (doy - 20) / 365.25)               # winter rough, summer calm
    yr_f = {y: rng.uniform(0.88, 1.15) for y in range(2015, 2025)}             # year-to-year variation
    base = base * np.array([yr_f[y] for y in idx.year])
    e = np.zeros(n); a = np.exp(-1 / 30.0); w = rng.normal(0, 1, n)
    for i in range(1, n):                                                       # ~30 h correlated noise
        e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * w[i]
    storms = np.zeros(n)
    for c in rng.choice(n, 140, replace=False):                                # 14 storms a year
        L = rng.uniform(18, 60); amp = rng.uniform(0.8, 2.5) * (1.4 if np.cos(2 * np.pi * (doy[c] - 20) / 365.25) > 0 else 0.7)
        storms += amp * np.exp(-0.5 * ((t - c) / L) ** 2)
    hs = np.clip(base * np.exp(0.38 * e) + storms, 0.1, None)
    wind = np.clip(2.5 + 4.2 * hs + rng.normal(0, 1.3, n), 0.3, None)
    return pd.DataFrame({"Hs": hs.round(2), "Wind": wind.round(1)}, index=idx)


def pvalue():
    sys.path[:0] = [str(PVAL)]
    from pvalue.models import generate_block_tasks_from_distance
    from pvalue.data import validate_metocean
    from pvalue.simulation import simulate_campaign, simulate_single_detailed, summarize_pxx
    from pvalue.stats import percentile_ci
    plt.rcParams.update(RC)                        # pvalue.visualization (imported by pvalue/__init__) resets fonts

    met = synth_metocean()
    ok, why = validate_metocean(met); assert ok, why
    THR = {"Hs": 1.5, "Wind": 12.0}
    tasks = generate_block_tasks_from_distance(300.0, 60.0, THR, name_prefix="Segment")   # 5 x 24 h blocks
    NS = 1000
    res = simulate_campaign(met, tasks, n_sims=NS, seed=7, time_interval_min=60)
    days = res["elapsed_days"].to_numpy()
    summ = summarize_pxx(res, [50, 75, 90]).set_index("Metric")["Value_days"]
    ci50 = percentile_ci(days, 50); ci90 = percentile_ci(days, 90)
    print(f"tasks {len(tasks)}, P50 {summ['P50']:.1f} P75 {summ['P75']:.1f} P90 {summ['P90']:.1f} "
          f"min {summ['Min']:.1f} max {summ['Max']:.1f}, CI90 {ci90[1]:.1f}-{ci90[2]:.1f}")
    blocks = {int(y): met[met.index.year == y] for y in sorted(set(met.index.year))}

    def detail(i):
        r = res.iloc[i]
        blk = blocks[int(r["year_sample"])]
        seg = simulate_single_detailed(blk, tasks, int(r["start_idx"]), time_interval_min=60)
        return r, blk, seg

    foot = ["pvalue: models.generate_block_tasks_from_distance, simulation.simulate_campaign (1,000 runs),",
            "simulate_single_detailed, summarize_pxx, stats.percentile_ci. Input: synthetic 10-year hourly Hs / wind."]
    fig, status = fig_base(("PValue", "weather-downtime Monte Carlo"), foot)
    aw = fig.add_axes([0.055, 0.585, 0.56, 0.27]); style(aw)
    lw = label(aw, "")
    ah = fig.add_axes([0.055, 0.17, 0.56, 0.30]); style(ah, grid=True)
    ah.yaxis.set_major_formatter(COMMA)
    lh = label(ah, "completion time of each run (days)")
    xmax = float(np.ceil(np.percentile(days, 99.5) / 5) * 5)
    bins = np.arange(0, xmax + 1, 1.0)
    ac = fig.add_axes([0.70, 0.585, 0.285, 0.27]); style(ac, grid=True)
    ac.set_xscale("log"); ac.set_xlim(8, NS * 1.1)
    ac.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    lc = label(ac, "P50 / P90 vs number of runs")
    at = fig.add_axes([0.70, 0.17, 0.285, 0.30]); at.axis("off"); at.set_xlim(0, 1); at.set_ylim(0, 1)
    WMAX = xmax

    def draw_weather(i):
        aw.cla(); style(aw)
        r, blk, seg = detail(i)
        el = float(r["elapsed_days"])
        span = int(np.ceil(el * 24)) + 1
        k = (int(r["start_idx"]) + np.arange(span)) % len(blk)
        hs = blk["Hs"].to_numpy()[k]
        tday = np.arange(span) / 24.0
        aw.plot(tday, hs, color=C["mute"], lw=0.6)
        aw.axhline(THR["Hs"], color=C["bad"], lw=0.7, ls=(0, (3, 2)))
        top = max(3.2, float(hs.max()) * 1.05)
        for s in seg:
            aw.axvspan(s["wait_start_day"] - seg[0]["wait_start_day"],
                       s["work_start_day"] - seg[0]["wait_start_day"], color=C["warn"], alpha=0.16, lw=0)
            w0 = s["work_start_day"] - seg[0]["wait_start_day"]
            aw.plot([w0, w0 + s["work_days"]], [top * 0.93] * 2, color=C["ok"], lw=3.0, solid_capstyle="butt")
        WX = max(12.0, el * 1.08)
        aw.set_xlim(0, WX); aw.set_ylim(0, top)
        aw.xaxis.set_major_formatter(COMMA)
        aw.text(WX * 0.995, THR["Hs"] + 0.08, "Hs limit 1.5 m", ha="right", va="bottom", color=C["bad"], fontsize=5.5)
        label(aw, f"run {i + 1:,}: start {pd.Timestamp(r['start_time']):%d %b}, Hs (m), "
                  f"green = 24 h segments, amber = waiting")
        return el

    hist_art = []

    def draw_hist(n):
        for a in hist_art:
            a.remove()
        hist_art.clear()
        cnt, _, bars = ah.hist(days[:n], bins=bins, color=C["acc"], alpha=0.75, lw=0)
        hist_art.extend(bars)
        ah.set_xlim(0, xmax); ah.set_ylim(0, max(4, cnt.max() * 1.18))
        if n >= 5:
            s = summarize_pxx(res.iloc[:n], [50, 90]).set_index("Metric")["Value_days"]
            for p, col in (("P50", C["ok"]), ("P90", C["bad"])):
                hist_art.append(ah.axvline(s[p], color=col, lw=1.1))
                hist_art.append(ah.text(s[p], ah.get_ylim()[1] * (0.97 if p == "P90" else 0.78), f" {p} {s[p]:.1f} d",
                                        color=col, fontsize=6, va="top", ha="left"))
            return s
        return None

    hist_t, p50_t, p90_t = [], [], []
    (c50,) = ac.plot([], [], color=C["ok"], lw=1.1)
    (c90,) = ac.plot([], [], color=C["bad"], lw=1.1)
    ac.set_ylim(np.percentile(days, 20), np.percentile(days, 97))
    frames = []
    # phase 1: six single runs drawn one by one (6 x 4 frames)
    for i in range(6):
        el = draw_weather(i)
        draw_hist(i + 1)
        status.set_text(f"run {i + 1:,} / {NS:,}: {el:.1f} days")
        for _ in range(4):
            frames.append(grab(fig))
    # phase 2: runs accumulate, P50 / P90 settle (geometric steps to 1,000)
    steps = np.unique(np.geomspace(8, NS, 46).astype(int))
    for n in steps:
        draw_weather(int(n) - 1)
        s = draw_hist(int(n))
        hist_t.append(n); p50_t.append(s["P50"]); p90_t.append(s["P90"])
        c50.set_data(hist_t, p50_t); c90.set_data(hist_t, p90_t)
        status.set_text(f"{n:,} runs   P50 {s['P50']:.1f} d   P90 {s['P90']:.1f} d")
        frames.append(grab(fig))
    ac.text(NS, p50_t[-1], "P50 ", color=C["ok"], fontsize=5.5, ha="right", va="bottom")
    ac.text(NS, p90_t[-1], "P90 ", color=C["bad"], fontsize=5.5, ha="right", va="bottom")
    # phase 3: summary table from summarize_pxx + percentile_ci
    rows = [("scope", "300 km, 5 x 24 h"), ("limits", "Hs 1.5 m, wind 12 m/s"),
            ("P50", f"{summ['P50']:.1f} d  ({ci50[1]:.1f}-{ci50[2]:.1f})"),
            ("P75", f"{summ['P75']:.1f} d"),
            ("P90", f"{summ['P90']:.1f} d  ({ci90[1]:.1f}-{ci90[2]:.1f})"),
            ("mean", f"{summ['Mean']:.1f} d"),
            ("range", f"{summ['Min']:.1f} - {summ['Max']:.1f} d")]
    at.text(0.0, 0.99, "summarize_pxx, 95% CI", color=C["mute"], fontsize=6.3, va="top")
    y = 0.85
    for kk, (a, b) in enumerate(rows):
        col = C["ok"] if a == "P50" else C["bad"] if a == "P90" else C["ink"]
        at.text(0.0, y, f"{a:<6}", color=C["mute"], fontsize=6.2, va="top")
        at.text(0.25, y, b, color=col, fontsize=6.2, va="top")
        y -= 0.125
        frames.append(grab(fig))
    status.set_text(f"{NS:,} runs   P90 {summ['P90']:.1f} days")
    status.set_color(C["bad"])
    for _ in range(3):
        frames.append(grab(fig))
    write_gif(frames, "pvalue-montecarlo.gif", hold=40, mid=45)


if __name__ == "__main__":
    if "coord" in WHICH:
        coord()
    if "pvalue" in WHICH:
        pvalue()
