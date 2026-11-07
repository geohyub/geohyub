"""Two more profile animations, same style as make_visuals.py (palette, figure size, helpers copied below).

  feathering-crosscurrent.gif  Feathering  src-tauri/python/Parser_2.py::calculate_feathering,
                                 calculate_distance_along_line, detect_feathering_changes (through the summary)
                               backend.py::_compute_run_boundaries, _build_analysis_summary (PASS / WARN / FAIL)
                               cross_current.py::compute_vessel_cog, compute_vessel_speed, estimate_cross_current,
                                 summarise (speed only, see below)
                               forecast.py::kalman_forecast
                               The matched head / tail buoy table is built here; NPD parsing and time matching
                               (parse_npd_file, match_npd_with_track) are not run. The streamer shape comes from a
                               lagged-drift model in this script: each cable point sits where the vessel was s/V
                               seconds ago, moved sideways by the cross-current it has felt since.
                               Direction of the recovered current is not drawn: estimate_cross_current is called
                               with feathering_sign="feathering_app" (positive = starboard) as backend.py does, but
                               calculate_feathering gives positive values for port drift, so the bearing it returns
                               is 180 degrees from the synthetic truth (checked and printed below).
  tidalworks-ellipses.gif      TidalWorks  ocean_toolkit/current/harmonic_analysis.py::harmonic_analysis_uv (UTide,
                                 custom constituent set), _ep2ap (truth ellipses -> u/v amplitude, phase)
                               ocean_toolkit/harmonic/analysis.py::harmonic_predict (synthesises u, v, elevation)
                               ocean_toolkit/current/ellipse_table.py::build_ellipse_table (incl. Z0 mean flow)
                               ocean_toolkit/current/residual.py::compute_residual, lowpass_filter (Godin)
                               ocean_toolkit/current/flood_ebb_velocity.py::principal_axis_deg, calc_flood_ebb
                                 (flood side from the rising tide)
                               The record is synthetic: invented M2 / S2 / N2 / K1 / O1 / M4 ellipses, a 5 cm/s mean
                               flow, a two-day wind-driven pulse and white noise, 30 days at 10 min.

Everything plotted (feathering, verdict, change zones, current speed, ellipses, constants, residual, flood/ebb)
comes back from those functions. No survey data, line, vessel or station names are used.

usage:  python make_feather_tidal_gif.py <out_dir> [feathering_repo] [tidalworks_repo] [feather,tidal]
needs:  numpy, pandas, matplotlib, pillow, scipy, utide
"""
import os
import sys
import types
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
FEATHER = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/Processing/Feathering")
TIDAL = Path(sys.argv[3] if len(sys.argv) > 3 else "/home/user/repos/Utilities/TidalWorks")
LAST_DIR = OUT

# ---- style: copied from make_visuals.py ---------------------------------------------------
C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
RC = {"font.family": "DejaVu Sans Mono", "font.size": 7,
      "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
      "xtick.color": C["mute"], "ytick.color": C["mute"]}
plt.rcParams.update(RC)
COMMA = FuncFormatter(lambda v, p: f"{v:,.0f}")


def reset_rc():                                    # Parser_2 sets a dark theme on import; put ours back
    matplotlib.rcdefaults()
    plt.rcParams.update(RC)


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
# 1. Feathering - streamer pushed by a turning cross-current, judged against +-10 deg
# =========================================================================================
def feathering():
    import pandas as pd
    py = FEATHER / "src-tauri" / "python"
    sys.path[:0] = [str(py)]
    # backend.py opens its log files under %LOCALAPPDATA%\Feathering at import; point that at a scratch folder
    os.environ["LOCALAPPDATA"] = str(OUT / "_feather_logs")
    import Parser_2 as analyzer
    import backend
    import cross_current
    import forecast
    reset_rc()

    rng = np.random.default_rng(11)
    V, SHOT, LINE = 2.4, 12.5, 10_000.0            # m/s, shot interval (m), line length (m)
    LEAD, CABLE = 150.0, 3_000.0                   # vessel -> head buoy, head -> tail buoy (m)
    LIMIT, RUN_IN, RUN_OUT = 10.0, 1_000.0, 1_000.0
    NS = int(LINE / SHOT) + 1
    T = LINE / V
    ts = np.arange(NS) * SHOT / V                   # shot times (s)

    def vc(t):                                     # cross-current (m/s), + = northward = to port of an eastbound vessel
        base = -0.16 + 0.48 * (t / T)
        eddy = 0.45 * np.exp(-0.5 * ((t - 0.52 * T) / 420.0) ** 2)
        return base + eddy + 0.02 * np.sin(2 * np.pi * t / 700.0)

    # vessel and cable history on a fine time grid (starts early enough for the tail at shot 0)
    tg = np.arange(-(LEAD + CABLE) / V - 60, T + 1, 1.0)
    xv = V * tg                                    # along the line, east (m)
    yv = 4.0 * np.sin(2 * np.pi * tg / 900.0)      # small steering weave (m)
    drift = np.cumsum(vc(tg))                      # integral of the cross-current (m), 1 s steps

    def cable_point(t, s):                         # lagged-drift streamer model (this script)
        t0 = t - s / V
        x = np.interp(t0, tg, xv); y = np.interp(t0, tg, yv)
        y = y + (np.interp(t, tg, drift) - np.interp(t0, tg, drift))
        return x, y
    E0, N0 = 500_000.0, 3_800_000.0                # synthetic projected origin (m)
    hx, hy = cable_point(ts, LEAD); tx, tyy = cable_point(ts, LEAD + CABLE)
    gps = lambda n: rng.normal(0, 1.5, n)
    swell = np.convolve(rng.normal(0, 1.0, NS + 8), np.ones(9) / 3.0, mode="valid")   # tail wander (m)
    tyy = tyy + 9.0 * swell[:NS]
    t0 = datetime(2026, 4, 9, 3, 12, 0)
    df = pd.DataFrame({
        "FFID": 1001 + np.arange(NS),
        "Time": [(t0 + timedelta(seconds=float(s))).strftime("%H:%M:%S.%f")[:-3] for s in ts],
        "FRONT_X": E0 + hx + gps(NS), "FRONT_Y": N0 + hy + gps(NS),
        "TAIL_X": E0 + tx + gps(NS), "TAIL_Y": N0 + tyy + gps(NS),
    })
    PLANNED = 270.0                                # cable aft bearing of an eastbound line (operator input)

    # ---- the repo's full-line numbers (used at the end; frames recompute on the shots logged so far)
    feath = analyzer.calculate_feathering(df, PLANNED)
    dist_m = analyzer.calculate_distance_along_line(df) * 1000.0
    ffid = df["FFID"].values
    run_in_ffid, run_out_ffid = backend._compute_run_boundaries(ffid, dist_m, RUN_IN, RUN_OUT)

    def current(sub):
        fx = sub["FRONT_X"].values.astype(float); fy = sub["FRONT_Y"].values.astype(float)
        tsec = analyzer._unwrap_midnight(np.asarray(analyzer.time_to_seconds_vectorized(sub["Time"]), float))
        cog = cross_current.compute_vessel_cog(fx, fy, window=5)
        spd = cross_current.compute_vessel_speed(fx, fy, tsec, window=5)
        f_ = analyzer.calculate_feathering(sub, PLANNED)
        return cross_current.estimate_cross_current(f_, spd, cog, feathering_sign="feathering_app")

    # ---- figure
    foot = ["Feathering calculate_feathering · detect_feathering_changes · _build_analysis_summary"
            " · kalman_forecast",
            "estimate_cross_current (speed) · synthetic line and current · streamer: lagged-drift model in this script"]
    fig, status = fig_base(("Feathering", "streamer feathering vs cross-current"), foot)
    am = fig.add_axes([0.065, 0.615, 0.635, 0.235]); style(am, grid=True)
    af = fig.add_axes([0.065, 0.175, 0.635, 0.33]); style(af, grid=True)
    ac = fig.add_axes([0.765, 0.615, 0.22, 0.235]); style(ac, grid=True)
    ak = fig.add_axes([0.765, 0.12, 0.22, 0.40]); ak.axis("off")

    km = lambda v: v / 1000.0
    EXG = 2.0
    am.set_xlim(-3.4, 10.3); am.set_ylim(-1.3, 1.3)
    am.set_xlabel("east (km)", fontsize=6, labelpad=1)
    label(am, "plan view (km) · streamer blue · tail-buoy track grey · cross-line x2")
    am.set_yticks([-1, 0, 1])
    am.axhline(0, color=C["mute"], lw=0.6, ls=(0, (4, 3)))
    am.axvspan(0, km(RUN_IN), color=C["mute"], alpha=0.06, lw=0)
    am.axvspan(km(LINE - RUN_OUT), km(LINE), color=C["mute"], alpha=0.06, lw=0)
    vtrk, = am.plot([], [], color=C["mute"], lw=0.5)
    ttrk = am.scatter([], [], s=0.8, lw=0, color=C["mute"])
    cab, = am.plot([], [], color=C["acc"], lw=1.6, solid_capstyle="round")
    head_p, = am.plot([], [], "o", ms=2.6, color=C["ink"])
    tail_p, = am.plot([], [], "s", ms=2.6, color=C["ink"])
    boat, = am.plot([], [], marker=">", ms=5, color=C["ink"], ls="none")
    arr = [None]

    af.set_xlim(ffid[0], ffid[-1]); af.set_ylim(-14, 16)
    af.xaxis.set_major_formatter(COMMA)
    af.set_xlabel("FFID", fontsize=6, labelpad=1)
    labf = label(af, f"feathering angle (deg) · limit ±{LIMIT:g}")
    for sgn in (1, -1):
        af.axhline(sgn * LIMIT, color=C["bad"], lw=0.7, ls="--")
    af.axhline(0, color=C["mute"], lw=0.4)
    if run_in_ffid is not None:
        af.axvspan(ffid[0], run_in_ffid, color=C["mute"], alpha=0.08, lw=0)
        af.text(ffid[0] + 8, -12.6, "run-in", fontsize=5.6, color=C["mute"])
    if run_out_ffid is not None:
        af.axvspan(run_out_ffid, ffid[-1], color=C["mute"], alpha=0.08, lw=0)
        af.text(run_out_ffid + 8, -12.6, "run-out", fontsize=5.6, color=C["mute"])
    lf, = af.plot([], [], color=C["ink"], lw=0.7)
    exf = af.scatter([], [], s=3, color=C["bad"], lw=0, zorder=5)

    true_kn = np.abs(vc(ts)) / 0.514444
    ac.set_xlim(ffid[0], ffid[-1]); ac.set_ylim(0, 1.45)
    ac.set_xticks([1200, 1600])
    ac.xaxis.set_major_formatter(COMMA)
    label(ac, "cross-current (kn)")
    ac.plot([], [], color=C["mute"], lw=0.6, ls=(0, (2, 2)), label="true at vessel")
    lt, = ac.plot([], [], color=C["mute"], lw=0.6, ls=(0, (2, 2)))
    le, = ac.plot([], [], color=C["acc"], lw=0.8)
    ac.text(0.02, 0.88, "from cable angle", transform=ac.transAxes, fontsize=5.4, color=C["acc"])
    ac.text(0.02, 0.77, "true at vessel", transform=ac.transAxes, fontsize=5.4, color=C["mute"])

    frames = []
    steps = np.unique(np.linspace(12, NS, 58).astype(int))
    S = np.linspace(LEAD, LEAD + CABLE, 24)
    for k in steps:
        sub = df.iloc[:k]
        f_ = analyzer.calculate_feathering(sub, PLANNED)           # per shot, on what is logged so far
        cs, _cd = current(sub)
        j = k - 1
        X = (sub["FRONT_X"].values - E0, sub["FRONT_Y"].values - N0)
        cx, cy = cable_point(ts[j], np.r_[0.0, S])
        vtrk.set_data(km(V * ts[:k]), km(np.interp(ts[:k], tg, yv)) * EXG)
        cab.set_data(km(cx), km(cy) * EXG)
        head_p.set_data([km(X[0][-1])], [km(X[1][-1]) * EXG])
        tail_p.set_data([km(sub["TAIL_X"].values[-1] - E0)], [km(sub["TAIL_Y"].values[-1] - N0) * EXG])
        boat.set_data([km(V * ts[j])], [km(np.interp(ts[j], tg, yv)) * EXG])
        ex = np.abs(f_) >= LIMIT
        ttrk.set_offsets(np.c_[km(sub["TAIL_X"].values - E0), km(sub["TAIL_Y"].values - N0) * EXG])
        ttrk.set_color(np.where(ex, C["bad"], C["mute"]))
        if arr[0] is not None:
            arr[0].remove()
        v_now = float(vc(ts[j]))
        arr[0] = am.annotate("", xy=(km(V * ts[j]) + 0.35, 0.9 * np.sign(v_now) * min(1, abs(v_now) / 0.3) + 0.0),
                             xytext=(km(V * ts[j]) + 0.35, 0.0),
                             arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=0.9, mutation_scale=6))
        lf.set_data(ffid[:k], f_)
        exf.set_offsets(np.c_[ffid[:k][ex], f_[ex]] if ex.any() else np.empty((0, 2)))
        le.set_data(ffid[:k], cs / 0.514444)
        lt.set_data(ffid[:k], true_kn[:k])
        status.set_text(f"FFID {ffid[j]:,}   {f_[-1]:+.1f}°   current {cs[-1] / 0.514444:.2f} kn")
        status.set_color(C["bad"] if ex[-1] else C["acc"])
        frames.append(grab(fig))
    if arr[0] is not None:
        arr[0].remove()
    boat.set_visible(True)

    # ---- the program's verdict on the full line --------------------------------------------
    summ = backend._build_analysis_summary(
        feath, ffid, dist_m, npd_records=NS, track_records=NS, matched_records=NS, line_name="Line A",
        fast_match=False, tolerance_s=2.0, head_position="Head_Buoy", tail_position="Tail_Buoy",
        feathering_limit=LIMIT, run_in_ffid=run_in_ffid, run_out_ffid=run_out_ffid)
    fc = forecast.kalman_forecast(feath)
    cs_all, cd_all = current(df)
    cc_sum = cross_current.summarise(cs_all, cd_all)
    for ch in summ["changes"]:
        af.axvspan(ch["start_ffid"], ch["end_ffid"], color=C["warn"], alpha=0.16, lw=0, zorder=0)
    if summ["changes"]:
        c0 = summ["changes"][0]
        af.text(c0["start_ffid"], 13.6, c0["detection_type"].lower(), fontsize=5.6, color=C["warn"])
    lim = summ["limit"]
    labf.set_text(f"feathering (deg) · {lim['main_exceeded_count']:,} main-line shots ≥ ±{LIMIT:g}°")
    labf.set_color(C["bad"] if lim["main_exceeded_count"] else C["mute"])
    status.set_text(f"{len(summ['changes'])} change zone(s) · max |feathering| {lim['max_abs']:.1f}°")
    status.set_color(C["ink"])
    frames += [grab(fig)] * 6

    VC = {"PASS": C["ok"], "WARN": C["warn"], "FAIL": C["bad"], "INFO": C["mute"]}
    ms = summ["main_stats"]
    rows = [("main line", f"{summ['window']['included_records']:,} shots"),
            ("mean / std", f"{ms['mean']:+.1f}° / {ms['std']:.1f}°"),
            ("max |feath.|", f"{lim['max_abs']:.1f}°"),
            ("≥ limit", f"{lim['main_exceeded_count']:,} ({lim['main_exceeded_percent']:.1f}%)"),
            ("next shot", f"{fc['forecast_next_deg']:+.1f}° (Kalman)"),
            ("current", f"mean {cc_sum['speed_mean_knots']:.2f} kn")]
    for i, (nm, val) in enumerate(rows):
        y = 0.97 - i * 0.135
        ak.text(0.0, y, nm, fontsize=6.0, color=C["mute"], transform=ak.transAxes, va="center")
        ak.text(0.50, y, val, fontsize=6.0, color=C["ink"], transform=ak.transAxes, va="center")
        frames += [grab(fig)] * 2
    v = summ["verdict"]
    ak.text(0.0, 0.08, f"line verdict  {v}", fontsize=7.5, fontweight="bold", color=VC[v],
            transform=ak.transAxes, va="center")
    status.set_text(f"{lim['main_exceeded_percent']:.1f}% of main line over ±{LIMIT:g}° -> {v}")
    status.set_color(VC[v])
    frames.append(grab(fig))
    write_gif(frames, "feathering-crosscurrent.gif", hold=50, mid=40)

    # sign check of the recovered current bearing against the synthetic truth (main line)
    mi = (ffid >= (run_in_ffid or 0)) & (ffid <= (run_out_ffid or ffid[-1]))
    true_dir = np.where(vc(ts) > 0, 0.0, 180.0)
    dd = (cd_all - true_dir + 180) % 360 - 180
    ok = np.isfinite(dd) & mi & (np.abs(vc(ts)) > 0.05)
    return dict(verdict=v, exceed=lim["main_exceeded_count"], pct=lim["main_exceeded_percent"],
                max_abs=lim["max_abs"], changes=[(c["start_ffid"], c["end_ffid"], c["detection_type"])
                                                 for c in summ["changes"]],
                forecast=fc["forecast_next_deg"], cc=cc_sum, run=(run_in_ffid, run_out_ffid),
                dir_error_median=float(np.median(np.abs(dd[ok]))) if ok.any() else None)


# =========================================================================================
# 2. TidalWorks - a moored current record resolved into tidal ellipses, mean flow and flood / ebb
# =========================================================================================
def tidal():
    import pandas as pd
    sys.path[:0] = [str(TIDAL)]
    from tidalworks.ocean_toolkit.current.harmonic_analysis import harmonic_analysis_uv, _ep2ap
    from tidalworks.ocean_toolkit.harmonic.analysis import harmonic_predict
    from tidalworks.ocean_toolkit.current.ellipse_table import build_ellipse_table
    from tidalworks.ocean_toolkit.current.residual import compute_residual, lowpass_filter
    from tidalworks.ocean_toolkit.current.flood_ebb_velocity import principal_axis_deg, calc_flood_ebb
    from tidalworks.ocean_toolkit.current.tidal_ellipse_plot import _draw_ellipse
    reset_rc()

    LAT = 35.0
    time = pd.date_range("2026-03-02 00:00", periods=30 * 144, freq="10min").to_numpy()
    days = (time - time[0]) / np.timedelta64(1, "D")
    N = time.size
    # invented ellipses: SEMA, SEMI (cm/s, negative = clockwise), INC (deg CCW from east), PHA (deg)
    TRUE = dict(M2=(62.0, -9.0, 28.0, 140.0), S2=(26.0, -4.0, 32.0, 175.0), N2=(12.0, -2.0, 25.0, 120.0),
                K1=(11.0, 5.0, 70.0, 210.0), O1=(8.0, 3.0, 75.0, 190.0), M4=(4.0, 1.0, 30.0, 60.0))
    ELEV = dict(M2=110.0, S2=48.0, N2=22.0, K1=28.0, O1=21.0, M4=3.0)   # tide amplitude (cm)
    cu, cv, ch = {}, {}, {}
    for n, (a, b, inc, pha) in TRUE.items():
        Au, gu, Av, gv = _ep2ap(a, b / a, inc, pha)                    # repo: ellipse -> u / v amplitude, phase
        cu[n] = (float(Au), float(gu)); cv[n] = (float(Av), float(gv))
        ch[n] = (ELEV[n], (pha + 90.0) % 360.0)                         # tide rises while the current runs along +INC
    u_t, mu = harmonic_predict(time, cu, LAT, return_method=True)       # repo: UTide reconstruct, nodal corrections
    v_t, mv = harmonic_predict(time, cv, LAT, return_method=True)
    elev, mh = harmonic_predict(time, ch, LAT, return_method=True)
    rng = np.random.default_rng(7)

    def red(sig):                                                       # AR(1) noise, cm/s
        e = rng.normal(0, sig * np.sqrt(1 - 0.8 ** 2), N); out = np.zeros(N)
        for i in range(1, N):
            out[i] = 0.8 * out[i - 1] + e[i]
        return out
    pulse = 15.0 * np.exp(-0.5 * ((days - 17.5) / 1.1) ** 2)            # wind-driven pulse of a few days toward 215 deg
    u = u_t + 2.9 + pulse * np.sin(np.radians(215)) + red(3.0)          # + mean flow 4.5 cm/s toward 140 deg
    v = v_t - 3.45 + pulse * np.cos(np.radians(215)) + red(3.0)

    CON = ["M2", "S2", "N2", "K1", "O1", "M4"]
    hr = harmonic_analysis_uv(time, u, v, lat=LAT, layer_name="Layer A", constit=CON)
    names = [c.strip() for c in hr.constituents]
    tbl = build_ellipse_table({"Layer A": hr}, constits=("Z0",) + tuple(CON))
    spd = np.hypot(u, v); dirn = np.degrees(np.arctan2(u, v)) % 360
    ru, rv, rst = compute_residual(u, v, hr.u_pred, hr.v_pred, time)
    # Godin on hourly samples: with dt_hours=1/6 the rescaled kernel has an even tap count and
    # _convolve_lowpass returns one sample short (IndexError), so the 10-min record is thinned to hourly here
    lu, lv = lowpass_filter(u[::6], v[::6], dt_hours=1.0, method="godin"); dh = days[::6]
    axis = principal_axis_deg(dirn, spd)
    fe = calc_flood_ebb(time, spd, dirn, flood_dir_range=None, elevation=elev)
    along = lambda uu, vv: uu * np.sin(np.radians(axis)) + vv * np.cos(np.radians(axis))

    foot = ["TidalWorks harmonic_analysis_uv (UTide) · build_ellipse_table · compute_residual · lowpass_filter"
            " · calc_flood_ebb",
            "synthetic 30-day record: harmonic_predict of invented ellipses + mean flow + wind pulse + noise · Godin hourly"]
    fig, status = fig_base(("TidalWorks", "tidal current ellipses"), foot)
    at = fig.add_axes([0.062, 0.64, 0.555, 0.205]); style(at, grid=True)
    ar = fig.add_axes([0.062, 0.455, 0.555, 0.105]); style(ar, grid=True)
    ak = fig.add_axes([0.062, 0.095, 0.555, 0.275]); ak.axis("off")
    ah = fig.add_axes([0.66, 0.15, 0.325, 0.695]); style(ah, grid=True)
    at.set_xlim(0, 30); at.set_ylim(-135, 135); at.set_xticklabels([])
    labt = label(at, f"current along principal axis {axis:.0f}° (cm/s)")
    lo, = at.plot([], [], color=C["mute"], lw=0.4)
    lp, = at.plot([], [], color=C["acc"], lw=0.5)
    ar.set_xlim(0, 30); ar.set_ylim(0, 22); ar.set_yticks([0, 10, 20])
    labr = label(ar, "subtidal speed (cm/s) · Godin low-pass · x: day")
    ar.axvspan(0, 1.5, color=C["mute"], alpha=0.10, lw=0); ar.axvspan(28.5, 30, color=C["mute"], alpha=0.10, lw=0)
    lr, = ar.plot([], [], color=C["warn"], lw=0.8)
    R = 115
    ah.set_xlim(-R, R); ah.set_ylim(-R, R); ah.set_aspect("equal")
    ah.set_xticks([-100, -50, 0, 50, 100]); ah.set_yticks([-100, -50, 0, 50, 100])
    labh = label(ah, "hodograph · east / north (cm/s)")
    ah.axhline(0, color=C["hair"], lw=0.6); ah.axvline(0, color=C["hair"], lw=0.6)
    sc = ah.scatter([], [], s=0.5, lw=0, color=C["mute"], alpha=0.35)

    frames = []
    for k in np.linspace(150, N, 22).astype(int):                       # the record as it is read
        lo.set_data(days[:k], along(u[:k], v[:k]))
        sc.set_offsets(np.c_[u[:k:3], v[:k:3]])
        status.set_text(f"day {days[k - 1]:4.1f}   {spd[k - 1]:,.0f} cm/s toward {dirn[k - 1]:03.0f}°")
        frames.append(grab(fig))

    COL = dict(M2=C["acc"], S2=C["warn"], K1=C["ok"], O1=C["ink"], N2=C["mute"], M4=C["mute"])
    PER = dict(M2=12.42, S2=12.00, N2=12.66, K1=23.93, O1=25.82, M4=6.21)
    XT = 50 * 0.602 * 5.8 / (0.555 * W * 72)                          # x of character column 50 (mono 5.8 pt)
    ak.text(0.0, 0.97, f"{'const':<5}{'period':>8}  {'SEMA':>5} {'±95%':>4}  {'SEMI':>5}  {'INC':>6}  {'PHA':>6}",
            fontsize=5.8, color=C["mute"], transform=ak.transAxes, va="center")
    ak.text(XT, 0.97, "true SEMA / INC", fontsize=5.8, color=C["mute"], transform=ak.transAxes, va="center")
    row = [0]

    def table_row(n, col):
        i = names.index(n); row[0] += 1
        y = 0.97 - row[0] * 0.118
        ci = hr.SEMA_ci[i] if hr.SEMA_ci is not None else np.nan
        a, b, inc, _p = TRUE[n]
        ak.text(0.0, y, f"{n:<5}{PER[n]:6.2f} h  {hr.SEMA[i]:5.1f} ±{ci:3.1f}  {hr.SEMI[i]:+5.1f}  {hr.INC[i]:5.1f}°"
                        f"  {hr.PHA[i]:5.1f}°", fontsize=5.8, color=col, transform=ak.transAxes, va="center")
        ak.text(XT, y, f"{a:5.1f} / {inc:4.0f}°", fontsize=5.8, color=C["mute"], transform=ak.transAxes,
                va="center")

    trace, = ah.plot([], [], color=C["ink"], lw=1.0)
    dot, = ah.plot([], [], "o", ms=3, color=C["ink"])
    for n in ["M2", "S2", "K1", "O1"]:
        i = names.index(n)
        a, b, inc, pha = hr.SEMA[i], hr.SEMI[i], hr.INC[i], hr.PHA[i]
        th = np.linspace(0, 2 * np.pi, 181); ir = np.radians(inc)
        # same parametrisation as tidal_ellipse_plot._draw_ellipse; time runs one way round, sign of SEMI sets it
        x0, y0 = a * np.cos(th), b * np.sin(th)
        ex, ey = x0 * np.cos(ir) - y0 * np.sin(ir), x0 * np.sin(ir) + y0 * np.cos(ir)
        trace.set_color(COL[n]); dot.set_color(COL[n])
        sense = "clockwise" if b < 0 else "anticlockwise"
        for f in np.linspace(0.18, 1.0, 6):
            m = int(f * 180) + 1
            trace.set_data(ex[:m], ey[:m]); dot.set_data([ex[m - 1]], [ey[m - 1]])
            status.set_text(f"{n}   SEMA {a:.1f} cm/s   INC {inc:.0f}°   {sense}")
            status.set_color(COL[n])
            frames.append(grab(fig))
        trace.set_data([], []); dot.set_data([], [])
        _draw_ellipse(ah, a, b, inc, pha, COL[n], n)                    # repo drawing of the finished ellipse
        for ln in ah.lines[-3:]:
            ln.set_linewidth(min(ln.get_linewidth(), 1.1))
        ah.lines[-1].set_markersize(3)
        table_row(n, COL[n])
        frames += [grab(fig)] * 2
    for n in ["N2", "M4"]:
        table_row(n, C["mute"])
    frames += [grab(fig)] * 3

    # mean flow (Z0), tidal prediction and the subtidal residual
    z0 = tbl[tbl["분조"] == "Z0"].iloc[0]
    zs, zd = float(z0["장축길이(cm/s)"]), float(z0["기울기(°)"])
    row[0] += 0
    ak.text(0.0, 0.97 - 7 * 0.118, f"Z0   mean flow {zs:.1f} cm/s toward {zd:03.0f}°", fontsize=5.8,
            color=C["ink"], transform=ak.transAxes, va="center")
    ah.annotate("", xy=(8 * zs * np.sin(np.radians(zd)), 8 * zs * np.cos(np.radians(zd))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.2, mutation_scale=7))
    ah.text(8 * zs * np.sin(np.radians(zd)) + 3, 8 * zs * np.cos(np.radians(zd)) - 9, "Z0 x8", fontsize=5.8,
            color=C["warn"], fontweight="bold")
    pa, lpa = along(hr.u_pred, hr.v_pred), np.hypot(lu, lv)
    for k in np.linspace(N // 6, N, 6).astype(int):
        lp.set_data(days[:k], pa[:k]); lr.set_data(dh[:k // 6], lpa[:k // 6])
        status.set_text(f"tidal prediction · {rst.variance_explained_tidal:.0f}% of variance is tidal")
        status.set_color(C["acc"])
        frames.append(grab(fig))
    labt.set_text(f"along {axis:.0f}° (cm/s) · observed grey · UTide prediction blue")
    inner = (dh > 1.5) & (dh < 28.5)                                   # skip the filter half-width at each end
    pk = int(np.flatnonzero(inner)[np.nanargmax(lpa[inner])])
    labr.set_text(f"subtidal speed (cm/s) · Godin · peak {lpa[pk]:.0f} cm/s on day {dh[pk]:.1f}")
    frames += [grab(fig)] * 4

    # flood / ebb on the hodograph
    for key, colr in (("flood", C["ink"]), ("ebb", C["ink"])):
        s_, d_ = fe[f"{key}_max_speed"], fe[f"{key}_dir"]
        ah.annotate("", xy=(s_ * np.sin(np.radians(d_)), s_ * np.cos(np.radians(d_))), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.3, mutation_scale=8))
        tx_, ty_ = s_ * np.sin(np.radians(d_)), s_ * np.cos(np.radians(d_))
        ah.text(tx_ - 4 if key == "flood" else tx_ + 4, ty_ + 7 if key == "flood" else ty_ - 12, f"{key} {s_:.0f}",
                fontsize=6, color=colr, fontweight="bold", ha="right" if key == "flood" else "left")
        frames += [grab(fig)] * 3
    basis = {"rising_tide": "rising tide", "axis_only": "axis only", "range": "fixed range"}[fe["flood_basis"]]
    labh.set_text(f"hodograph · flood side: {basis}")
    status.set_text(f"flood {fe['flood_dir']:03.0f}° {fe['flood_max_speed']:.0f} cm/s · ebb {fe['ebb_dir']:03.0f}°"
                    f" {fe['ebb_max_speed']:.0f} cm/s")
    status.set_color(C["ink"])
    frames.append(grab(fig))
    write_gif(frames, "tidalworks-ellipses.gif", hold=50, mid=40)
    rec = {n: (round(float(hr.SEMA[names.index(n)]), 1), round(float(hr.SEMI[names.index(n)]), 1),
               round(float(hr.INC[names.index(n)]), 1), round(float(hr.PHA[names.index(n)]), 1)) for n in CON}
    return dict(methods=(mu, mv, mh), recovered=rec, true=TRUE, z0=(zs, zd), axis=axis,
                tidal_pct=rst.variance_explained_tidal, resid_mean=rst.mean_speed, flood=fe)


if __name__ == "__main__":
    which = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"feather", "tidal"}
    if "feather" in which:
        print(feathering())
    if "tidal" in which:
        print(tidal())
