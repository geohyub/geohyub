"""Three more profile animations, same style as make_visuals.py (whose helpers this imports).

  navqc-track.gif      NavQC    desktop/services/analysis_service.py
                                  enrich_records, detect_anomalies, analyze_timing, calculate_cross_track_error
                                desktop/services/line_verdicts.py::line_diagnostics
                                desktop/verdict.py::line_statuses, rollup
                                desktop/writers/p190_writer.py::write_p190
  gt-cpt.gif           GT suite (GeoViewShared/geoview_cpt)
                                reportgen/robertson.py::compute_robertson_derived (Qt, Fr, Ic, 9-zone SBT)
                                reportgen/charts.py::draw_robertson_sbt_chart
  ai-firstbreak.gif    MarineSeismicAI
                                models/v7_uncertainty.py::V7UncertaintyModel (+ predict_with_uncertainty, MC-Dropout)
                                training/losses.py::CombinedV7Loss, core/preprocessing.py::normalize_trace,
                                generate_gaussian_heatmap; QC thresholds of deployment/fold_ensemble.py

Everything plotted (flags, verdicts, zones, picks, uncertainty) comes back from those functions;
the data they are fed is synthetic and generated below. The first-break model has no shipped weights,
so it is trained here, briefly, on CPU, on synthetic gathers -- it is not the production model.

usage:  python make_visuals_2.py <out_dir> [repos_root] [geoview_shared] [last_png_dir] [nav,gt,fb] [fb_cache_dir]
needs:  numpy, scipy, matplotlib, pillow, pyproj, torch (CPU)
"""
import sys, types, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_visuals as MV                      # palette, rcParams, fig_base/style/label/grab/write_gif
from make_visuals import C, fig_base, style, label, grab, write_gif, R, SHARED
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def pkg(name, path=None):
    m = types.ModuleType(name); m.__path__ = [str(path)] if path else []; sys.modules[name] = m


# =========================================================================================
# 1. NavQC - vessel runs two planned lines; nav QC flags the problems; line verdicts
# =========================================================================================
def navqc():
    sys.path[:0] = [str(R / "navqc"), str(SHARED)]
    from navqc.desktop.services import analysis_service as A
    from navqc.desktop.services.line_verdicts import line_diagnostics
    from navqc.desktop import verdict as V
    from navqc.desktop.writers.p190_writer import write_p190
    from pyproj import Transformer

    rng = np.random.default_rng(7)
    EPSG, E0, N0, LEN, SEP = 32652, 512000.0, 3881000.0, 1500.0, 120.0
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    TOL = 10.0                                              # contract CTE tolerance (m), set by me
    tol = V.Tolerances(cte_tol_m=TOL)
    v = 4.6 * 0.514444                                      # 4.6 kn, 1 Hz fixes
    n = int(LEN / v)
    lines = []
    for li, (name, y, dirn, t0) in enumerate([("NL-1001", N0, +1, 9 * 3600 + 30 * 60),
                                               ("NL-1002", N0 + SEP, -1, 9 * 3600 + 49 * 60)]):
        k = np.arange(n)
        along = k * v + rng.normal(0, 0.03, n)                # RTK-grade fixes
        x = E0 + along if dirn > 0 else E0 + LEN - along
        off = 2.0 * np.sin(k / 70 + li) + rng.normal(0, 0.03, n)
        hdg = (90.0 if dirn > 0 else 270.0) + rng.normal(0, 0.6, n) + np.gradient(off) * 20
        tsec = t0 + k.astype(float)
        keep = np.ones(n, bool)
        if li == 0:
            hdg[300] += 58                                  # gyro glitch, one record
        else:
            off += 15.5 * np.exp(-((k - 230) / 55.0) ** 2)   # current pushes the vessel off-line
            keep[470:482] = False                           # 12 s logging gap
            x[560] += 100; off[560] += 30                     # GNSS position jump, one fix
        x, off, hdg, tsec, k = x[keep], off[keep], hdg[keep], tsec[keep], k[keep]
        yy = y - dirn * off                                 # starboard drift
        lon, lat = to_ll.transform(x, yy)
        recs = [dict(line_name=name, point_number=int(1000 + kk), easting=float(e), northing=float(nn),
                     latitude=float(la), longitude=float(lo), heading=float(h % 360), time_s=float(t),
                     depth=float(48 - 0.004 * (e - E0) + rng.normal(0, 0.1)), quality=4)
                for kk, e, nn, la, lo, h, t in zip(k, x, yy, lat, lon, hdg, tsec)]
        lines.append(dict(name=name, y=y, recs=recs, pre=[(E0, y), (E0 + LEN, y)]))

    def qc(recs, pre):
        rr = A.enrich_records([dict(r) for r in recs], in_place=True)
        an = A.detect_anomalies(rr)
        cte = np.array(A.calculate_cross_track_error(pre, [(r["easting"], r["northing"]) for r in rr]))
        return rr, an, cte

    fig, status = fig_base(("NavQC", "line QC as the vessel runs"),
                           "detect_anomalies · calculate_cross_track_error · line_statuses · write_p190"
                           "  ·  synthetic data, limit mine")
    a1 = fig.add_axes([0.06, 0.135, 0.50, 0.73]); style(a1)
    a2 = fig.add_axes([0.615, 0.565, 0.37, 0.30]); style(a2, grid=True)
    a3 = fig.add_axes([0.615, 0.235, 0.37, 0.235]); style(a3, grid=True)
    a1.set_xlim(E0 - 60, E0 + LEN + 60); a1.set_ylim(N0 - 80, N0 + SEP + 80)
    a1.ticklabel_format(useOffset=False, style="plain")
    a1.set_xticks(E0 + np.array([0, 500, 1000, 1500])); a1.set_xticklabels(["0", "500", "1000", "1500 m"])
    a1.set_yticks([N0, N0 + SEP]); a1.set_yticklabels(["L1001", "L1002"])
    a1.set_xlabel(f"easting (UTM 52N, EPSG:{EPSG})", fontsize=6, labelpad=1)
    label(a1, f"plan · preplot (dashed) · ±{TOL:.0f} m corridor")
    for L_ in lines:
        a1.plot([E0, E0 + LEN], [L_["y"]] * 2, color=C["mute"], lw=0.6, ls=(0, (3, 3)), zorder=1)
        a1.fill_between([E0, E0 + LEN], L_["y"] - TOL, L_["y"] + TOL, color=C["ok"], alpha=0.07, lw=0)
    trk, = a1.plot([], [], color=C["acc"], lw=1.0, zorder=3)
    trk_done = []
    boat, = a1.plot([], [], marker="o", ms=3.5, color=C["ink"], lw=0, zorder=6)
    flags_sc = a1.scatter([], [], s=14, facecolors="none", edgecolors=C["bad"], lw=0.9, zorder=5)
    a2.set_ylim(0, 22); a2.axhline(TOL, color=C["bad"], lw=0.6, ls="--")
    a2.set_xticklabels([]); lab2 = a2.text(0, 1.03, "", transform=a2.transAxes, fontsize=6.5, color=C["mute"])
    a3.set_ylim(0, 8); a3.set_xlabel("fix (1 Hz)", fontsize=6, labelpad=1)
    lab3 = a3.text(0, 1.04, "", transform=a3.transAxes, fontsize=6.5, color=C["mute"])
    l2, = a2.plot([], [], color=C["acc"], lw=0.8); s2 = a2.scatter([], [], s=5, color=C["bad"], lw=0, zorder=5)
    l3, = a3.plot([], [], color=C["ink"], lw=0.8); s3 = a3.scatter([], [], s=12, marker="^", color=C["bad"], lw=0, zorder=5, clip_on=False)
    chips_ax = fig.add_axes([0.615, 0.075, 0.37, 0.08]); chips_ax.axis("off")
    dims = [d.key for d in V.LINE_DIMS]
    SC = {V.STATUS_OK: C["ok"], V.STATUS_WARN: C["warn"], V.STATUS_BAD: C["bad"], V.STATUS_NA: C["mute"]}
    VT = {V.VERDICT_PASS: ("PASS", C["ok"]), V.VERDICT_COND: ("CONDITIONAL", C["warn"]),
          V.VERDICT_FAIL: ("FAIL", C["bad"]), V.VERDICT_NONE: ("NOT MEASURED", C["mute"])}
    chip_t = [chips_ax.text(i / len(dims), 0.5, d, fontsize=5.6, color=C["hair"], va="center",
                            transform=chips_ax.transAxes) for i, d in enumerate(dims)]
    vtexts = []

    frames, results = [], []
    for li, L_ in enumerate(lines):
        rr, an, cte = qc(L_["recs"], L_["pre"])
        st = V.line_statuses({**line_diagnostics(rr), "cte_max_m": float(cte.max())}, tol)
        roll = V.rollup(st, V.LINE_DIMS)
        results.append((L_["name"], roll.verdict, st, an["summary"], float(cte.max())))
        e = np.array([r["easting"] for r in rr]); nn = np.array([r["northing"] for r in rr])
        spd = np.array([r.get("speed") or 0.0 for r in rr]); idx = np.arange(len(rr))
        flag_seq = {}                                       # record index -> reason, from detect_anomalies
        for key, tag in (("heading_spikes", "hdg"), ("speed_spikes", "spd"), ("time_gaps", "gap"),
                         ("position_jumps", "jump"), ("turn_rate_spikes", "turn"), ("accel_spikes", "acc")):
            for item in an[key]:
                flag_seq.setdefault(item[0], tag)
        cte_bad = cte > TOL
        a2.set_xlim(0, len(rr)); a3.set_xlim(0, len(rr))
        for t_ in chip_t:
            t_.set_color(C["hair"])
        steps = np.linspace(20, len(rr), 34).astype(int)
        for kk in steps:
            # the program's QC re-run on the records logged so far
            _, an_k, _ = qc(L_["recs"][:kk], L_["pre"])
            trk.set_data(e[:kk], nn[:kk]); boat.set_data([e[kk - 1]], [nn[kk - 1]])
            fl = [s for s in flag_seq if s < kk]
            pts = [np.c_[e[fl], nn[fl]]] if fl else []
            for lp in trk_done:
                pts.append(lp)
            flags_sc.set_offsets(np.vstack(pts) if pts else np.empty((0, 2)))
            l2.set_data(idx[:kk], cte[:kk]); b = cte_bad[:kk]
            s2.set_offsets(np.c_[idx[:kk][b], cte[:kk][b]] if b.any() else np.empty((0, 2)))
            l3.set_data(idx[:kk], np.clip(spd[:kk], 0, 8))
            sp = [s for s in fl if flag_seq[s] in ("spd", "acc")]
            s3.set_offsets(np.c_[sp, np.clip(spd[sp], 0, 7.85)] if sp else np.empty((0, 2)))
            lab2.set_text(f"cross-track (m) · max {cte[:kk].max():4.1f}")
            lab2.set_color(C["bad"] if b.any() else C["mute"])
            lab3.set_text("speed (kn)" + (f" · {len(sp)} flagged" if sp else ""))
            lab3.set_color(C["bad"] if sp else C["mute"])
            cnt = an_k["summary"]["total_anomalies"]
            parts = [f"{len(an_k[k_])} {t_}" for k_, t_ in (("heading_spikes", "hdg spike"),
                     ("time_gaps", "gap"), ("position_jumps", "jump")) if an_k[k_]]
            if b.any():
                parts.insert(0, f"CTE > {TOL:.0f} m")
            status.set_text(f"{L_['name']}  fix {kk:3d}   " + (" · ".join(parts) if parts else "clean so far"))
            status.set_color(C["bad"] if parts else C["acc"])
            frames.append(grab(fig))
        # verdict for the line, from verdict.line_statuses + rollup
        for t_, d in zip(chip_t, dims):
            t_.set_color(SC[st[d]])
        word, col = VT[roll.verdict]
        vtexts.append(a1.text(E0 + LEN, L_["y"] + 30, word, fontsize=7, fontweight="bold", color=col,
                              ha="right", va="center", zorder=7,
                              bbox=dict(boxstyle="square,pad=0.2", fc=C["bg"], ec="none", alpha=0.85)))
        status.set_text(f"{L_['name']}  {roll.ok} ok · {roll.warn} warn · {roll.bad} bad · {roll.na} n/a"
                        f" -> {word}")
        status.set_color(col)
        frames += [grab(fig)] * 14
        a1.plot(e, nn, color=C["acc"], lw=1.0, zorder=3)
        if flag_seq:
            trk_done.append(np.c_[e[list(flag_seq)], nn[list(flag_seq)]])
        trk.set_data([], [])
    boat.set_data([], [])

    # P1/90 export of the first line, written by the program's own writer, typed out
    with tempfile.TemporaryDirectory() as td:
        rr0 = A.enrich_records([dict(r) for r in lines[0]["recs"]])
        p = write_p190(rr0[::25], str(Path(td) / "NL-1001.p190"), line_name="NL-1001",
                       date_str="2026-05-14", epsg=EPSG, record_type="V",
                       headers={"area": "Synthetic demo block", "vessel": "Demo vessel"})
        p190 = Path(p).read_text().splitlines()
    hsel = [l for l in p190 if l.startswith(("H0100", "H1800", "H1900"))]
    vsel = [l for l in p190 if l.startswith("V")][:6]
    show = hsel + vsel
    pa = fig.add_axes([0.06, 0.135, 0.925, 0.73]); pa.set_facecolor(C["bg"])
    for s_ in pa.spines.values():
        s_.set_color(C["hair"])
    pa.set_xticks([]); pa.set_yticks([]); pa.set_xlim(0, 1); pa.set_ylim(0, 1)
    pa.text(0.012, 0.93, "write_p190  ->  NL-1001.p190   (UKOOA P1/90, 80 columns, every 25th fix)",
            fontsize=6.5, color=C["mute"], transform=pa.transAxes)
    rows = [pa.text(0.012, 0.82 - i * 0.082, "", fontsize=6.6, color=C["ink"] if l.startswith("V") else C["mute"],
                    family="DejaVu Sans Mono", transform=pa.transAxes) for i, l in enumerate(show)]
    for ax_ in (a1, a2, a3, chips_ax):
        ax_.set_visible(False)
    status.set_text("P1/90 export"); status.set_color(C["acc"])
    for i, l in enumerate(show):
        for c in (40, 80):
            rows[i].set_text(l[:c]); frames.append(grab(fig))
    status.set_text("   ".join(f"{r[0]} {VT[r[1]][0]}" for r in results))
    frames.append(grab(fig))
    write_gif(frames, "navqc-track.gif", hold=45)
    return results


# =========================================================================================
# 2. GT suite (geoview_cpt) - CPT sounding draws down; Robertson SBT fills a column + the chart
# =========================================================================================
def gtcpt():
    from scipy.ndimage import gaussian_filter1d
    sys.path[:0] = [str(SHARED)]
    pkg("reportgen", SHARED / "geoview_cpt/reportgen")      # skip reportgen/__init__ (PDF/Word stack)
    from reportgen.robertson import compute_robertson_derived, ZONE_COLORS
    from reportgen.charts import draw_robertson_sbt_chart

    rng = np.random.default_rng(3)
    z = np.arange(0.5, 20.0 + 1e-9, 0.02)
    # (top m, qc at top, qc at base [MPa], Rf %, u2 excess as a fraction of qc)
    layers = [(0.0, 0.08, 0.20, 3.3, 0.55), (4.2, 8.0, 15.0, 0.55, 0.0), (7.8, 3.0, 4.5, 1.7, 0.10),
              (10.6, 1.3, 1.9, 4.0, 0.45), (15.0, 20.0, 30.0, 0.65, 0.0)]
    qc = np.zeros_like(z); rf = np.zeros_like(z); ue = np.zeros_like(z)
    for i, (top, a, b, r, u) in enumerate(layers):
        bot = layers[i + 1][0] if i + 1 < len(layers) else 99
        m = (z >= top) & (z < bot); f = (z[m] - top) / (min(bot, z[-1]) - top)
        qc[m] = a + (b - a) * f; rf[m] = r; ue[m] = u
    qc = gaussian_filter1d(qc, 3) * np.exp(gaussian_filter1d(rng.normal(0, 0.12, z.size), 1.5))
    rf = gaussian_filter1d(rf, 3) * np.exp(gaussian_filter1d(rng.normal(0, 0.12, z.size), 1.5))
    fs = qc * rf / 100
    u0 = 0.00981 * z
    u2 = u0 + gaussian_filter1d(ue, 3) * qc + rng.normal(0, 0.002, z.size)
    d = compute_robertson_derived(z, qc, fs, u2, water_depth_m=60.0)
    Qt, Fr, Ic, zones = d.Qt, d.Fr, d.Ic, d.zones

    fig, status = fig_base(("GT suite", "CPT soil behaviour type, live"),
                           "geoview_cpt: robertson.compute_robertson_derived (Ic, 9-zone SBT)"
                           " · charts.draw_robertson_sbt_chart  ·  synthetic CPT")
    y0, hh = 0.135, 0.73
    aq = fig.add_axes([0.055, y0, 0.15, hh]); af = fig.add_axes([0.225, y0, 0.11, hh])
    au = fig.add_axes([0.355, y0, 0.11, hh]); az = fig.add_axes([0.49, y0, 0.035, hh])
    ac = fig.add_axes([0.655, y0, 0.33, hh])
    for a, ttl in ((aq, "qc (MPa)"), (af, "Rf (%)"), (au, "u2 (MPa)"), (az, "SBT")):
        style(a, grid=a is not az); label(a, ttl); a.set_ylim(20, 0)
    aq.set_ylabel("depth below seabed (m)", fontsize=6, labelpad=1)
    for a in (af, au, az):
        a.set_yticklabels([])
    aq.set_xlim(0, 35); af.set_xlim(0, 6); au.set_xlim(-0.05, float(np.ceil(u2.max() * 5) / 5)); az.set_xticks([]); az.set_xlim(0, 1)
    au.plot(u0, z, color=C["mute"], lw=0.6, ls="--")
    draw_robertson_sbt_chart(ac, zone_alpha=0.30, zone_label_fontsize=6, title="", draw_nc_diagonal=False,
                             resolution=240, boundary_linewidth=0.35)
    ac.set_xlabel("Fr (%)", fontsize=6, labelpad=1); ac.set_ylabel("Qt", fontsize=6, labelpad=0)
    ac.tick_params(labelsize=6, length=2, pad=1.5, colors=C["mute"]); ac.grid(False)
    for s_ in ac.spines.values():
        s_.set_color(C["hair"])
    label(ac, "Robertson 1990 SBT chart")
    lq, = aq.plot([], [], color=C["ink"], lw=0.7); lf, = af.plot([], [], color=C["acc"], lw=0.7)
    lu, = au.plot([], [], color=C["ink"], lw=0.7)
    zc = np.array([matplotlib.colors.to_rgba(ZONE_COLORS[int(k)]) if k > 0 else (1, 1, 1, 0) for k in zones])
    col_img = np.ones((z.size, 1, 4)); col_img[..., 3] = 0
    im = az.imshow(col_img, aspect="auto", extent=[0, 1, z[-1], z[0]], interpolation="nearest")
    pts = ac.scatter([], [], s=5, lw=0.25, edgecolors=C["ink"], zorder=10)
    cur = ac.scatter([], [], s=34, facecolors="none", edgecolors=C["ink"], lw=1.0, zorder=11)
    cur_lines = [a.axhline(-1, color=C["acc"], lw=0.6) for a in (aq, af, au)]
    ok = np.isfinite(Qt) & np.isfinite(Fr) & (Qt > 0) & (Fr > 0)
    names = {2: "organic", 3: "clay", 4: "silt mix", 5: "sand mix", 6: "sand", 7: "gravelly sand",
             1: "sensitive", 8: "v.stiff sand", 9: "v.stiff fine"}
    frames = []
    for k in np.linspace(10, z.size, 70).astype(int):
        sl = slice(0, k)
        lq.set_data(qc[sl], z[sl]); lf.set_data(rf[sl], z[sl]); lu.set_data(u2[sl], z[sl])
        col_img[:k, 0] = zc[:k]; im.set_data(col_img)
        m = ok[:k]
        pts.set_offsets(np.c_[Fr[:k][m], Qt[:k][m]]); pts.set_facecolors(zc[:k][m])
        j = k - 1
        cur.set_offsets([[Fr[j], Qt[j]]] if ok[j] else np.empty((0, 2)))
        for c_ in cur_lines:
            c_.set_ydata([z[j]] * 2)
        zn = int(zones[j])
        status.set_text(f"{z[j]:5.2f} m   Ic {Ic[j]:4.2f}   zone {zn} {names.get(zn, '')}")
        status.set_color(C["acc"])
        frames.append(grab(fig))
    cur.set_offsets(np.empty((0, 2)))
    for c_ in cur_lines:
        c_.set_visible(False)
    # label each zone run thicker than 1 m beside the column, from the program's zones
    edges = np.flatnonzero(np.diff(zones)) + 1
    runs = np.split(np.arange(z.size), edges)
    seen = []
    for r_ in runs:
        if z[r_[-1]] - z[r_[0]] >= 1.0 and zones[r_[0]] > 0:
            az.text(1.25, z[r_].mean(), str(int(zones[r_[0]])), fontsize=6.5, va="center",
                    color=C["ink"], clip_on=False)
            seen.append(int(zones[r_[0]]))
    status.set_text("   ".join(f"{k_} {names[k_]}" for k_ in sorted(set(seen))))
    status.set_color(C["ink"])
    frames.append(grab(fig))
    write_gif(frames, "gt-cpt.gif", hold=45)
    return [(round(float(z[r_[0]]), 2), round(float(z[r_[-1]]), 2), int(zones[r_[0]])) for r_ in runs]


# =========================================================================================
# 3. MarineSeismicAI - V7 first-break model learns on synthetic gathers; MC-Dropout QC at the end
# =========================================================================================
FB_L, FB_DT, FB_NTR = 512, 0.002, 96                    # samples, 2 ms, traces per shot
FB_OFF = 30 + 12.5 * np.arange(FB_NTR)


def shot_gather(rng, bad=None):
    """Synthetic marine shot gather (FB_L, FB_NTR): direct wave + two head waves, seabed reflection and
    multiples, sub-seabed reflectors, swell + random noise. Returns data and first-break sample per trace."""
    t = np.arange(FB_L) * FB_DT
    h = rng.uniform(25, 90); v1 = 1500.0; v2 = rng.uniform(1650, 2300); z2 = rng.uniform(15, 70)
    v3 = v2 + rng.uniform(300, 1000); t0 = rng.uniform(0.01, 0.06)
    td = FB_OFF / v1
    th1 = FB_OFF / v2 + 2 * h * np.sqrt(1 / v1 ** 2 - 1 / v2 ** 2)
    th2 = FB_OFF / v3 + 2 * h * np.sqrt(1 / v1 ** 2 - 1 / v3 ** 2) + 2 * z2 * np.sqrt(1 / v2 ** 2 - 1 / v3 ** 2)
    fb = t0 + np.minimum(td, np.minimum(th1, th2))
    f = rng.uniform(25, 45); tau = rng.uniform(0.012, 0.025)

    def wav(ta, amp):                                   # causal damped wavelet starting at ta
        s_ = t[:, None] - ta[None]
        return amp[None] * np.where(s_ >= 0, np.sin(2 * np.pi * f * s_) * np.exp(-s_ / tau), 0.0)
    a_d = 1.0 / (1 + FB_OFF / 150); a_h = 0.5 / (1 + FB_OFF / 400)
    D = wav(t0 + td, a_d) + wav(t0 + th1, a_h * (th1 < td + 0.3)) + wav(t0 + th2, 0.8 * a_h)
    for k in range(1, 6):
        D += wav(t0 + np.sqrt(FB_OFF ** 2 + (2 * k * h) ** 2) / v1, (-0.6) ** k * a_d * 1.4)
    for _ in range(rng.integers(3, 8)):
        tz = rng.uniform(2 * h / v1 + 0.02, 0.9); vr = rng.uniform(1600, 2400)
        D += wav(t0 + np.sqrt(tz ** 2 + (FB_OFF / vr) ** 2), rng.uniform(-0.25, 0.25) * a_d * 2)
    sig = np.abs(D).max(0) + 1e-9
    snr = rng.uniform(2.5, 25) * np.exp(-FB_OFF / rng.uniform(800, 3000))
    swell = np.cumsum(rng.normal(0, 1, (FB_L, FB_NTR)), 0); swell -= swell.mean(0); swell /= swell.std(0) + 1e-9
    nz = 0.6 * rng.normal(0, 1, (FB_L, FB_NTR)) + rng.uniform(0, 1.2) * swell
    for a, b, g in (bad or []):                         # a patch of noisy traces
        snr[a:b] *= g
    return (D + nz * (sig / snr)[None] * 0.35).astype(np.float32), (fb / FB_DT).astype(np.float32)


def fb_train(cache):
    import torch
    src = R / "marineseismicai/src"
    pkg("marine_seismic_ai", src / "marine_seismic_ai")          # skip package __init__s (Qt/segyio stack)
    pkg("marine_seismic_ai.core", src / "marine_seismic_ai/core")
    sys.path.insert(0, str(src))
    from marine_seismic_ai.models.v7_uncertainty import V7UncertaintyModel
    from marine_seismic_ai.training.losses import CombinedV7Loss
    from marine_seismic_ai.core.preprocessing import normalize_trace, generate_gaussian_heatmap
    cf = Path(cache) / "fb_run.npz" if cache else None
    torch.manual_seed(0); torch.set_num_threads(2)
    rng = np.random.default_rng(0)

    def prep(D):
        return np.stack([normalize_trace(D[:, i].copy()) for i in range(D.shape[1])])[:, None].astype(np.float32)

    def make(ng):
        X, Y = zip(*[(prep(D), fb) for D, fb in (shot_gather(rng) for _ in range(ng))])
        return np.concatenate(X), np.concatenate(Y) / (FB_L - 1)
    Xtr, Ytr = make(32); Xva, Yva = make(4)
    Dh, fbh = shot_gather(np.random.default_rng(99), bad=[(60, 70, 0.15)])   # held-out display gather
    Xh = torch.tensor(prep(Dh))
    model = V7UncertaintyModel(input_length=FB_L, beta=6.0)
    if cf and cf.exists():
        z = np.load(cf, allow_pickle=True)
        model.load_state_dict(torch.load(Path(cache) / "fb_model.pth"))
        return dict(z), Dh, model, Xh

    HS = 4.0                                            # heatmap sigma (samples) for a 512-sample trace
    hm = lambda y: torch.tensor(np.stack([generate_gaussian_heatmap(float(v), FB_L, HS) for v in y]))[:, None].float()
    loss_fn = CombinedV7Loss(position_loss="gnll", lambda_hmap=0.5, lambda_sigma=0.1)   # as train_v7_unified
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    B, EPOCHS = 32, 6
    spe = len(Xtr) // B
    xv, yv, hv = torch.tensor(Xva), torch.tensor(Yva).float(), hm(Yva)
    snaps, steps, tr_l, va_l, va_mae, run = [], [], [], [], [], []
    t0 = time.time(); step = 0

    def snapshot():
        model.eval()
        with torch.no_grad():
            snaps.append(model(Xh)["position"].numpy().ravel() * (FB_L - 1))
            vl = []; err = []
            for i in range(0, len(xv), 192):
                o = model(xv[i:i + 192]); vl.append(loss_fn(o, yv[i:i + 192], hv[i:i + 192])[0].item())
                err.append((o["position"].squeeze(1) - yv[i:i + 192]).abs().numpy() * (FB_L - 1))
        steps.append(step); va_l.append(float(np.mean(vl))); va_mae.append(float(np.concatenate(err).mean()))
        tr_l.append(float(np.mean(run)) if run else np.nan); run.clear()
    snapshot()
    for ep in range(EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        for b in range(spe):
            idx = perm[b * B:(b + 1) * B]
            x, y = torch.tensor(Xtr[idx]), torch.tensor(Ytr[idx]).float()
            model.train(); opt.zero_grad()
            loss, _ = loss_fn(model(x), y, hm(Ytr[idx]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
            run.append(loss.item()); step += 1
            if (ep == 0 and step % 4 == 0) or step % 16 == 0:
                snapshot()
        sched.step()
        print(f"epoch {ep + 1}: val loss {va_l[-1]:.3f}  val MAE {va_mae[-1]:.2f} samples  {time.time() - t0:.0f} s", flush=True)
    res = dict(snaps=np.array(snaps), steps=np.array(steps), tr=np.array(tr_l), va=np.array(va_l),
               mae=np.array(va_mae), spe=spe, minutes=(time.time() - t0) / 60, n_train=len(Xtr), fbh=fbh)
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cf, **res); torch.save(model.state_dict(), Path(cache) / "fb_model.pth")
    return res, Dh, model, Xh


def firstbreak(cache=None):
    import torch
    res, Dh, model, Xh = fb_train(cache)
    snaps, steps, spe = res["snaps"], res["steps"], int(res["spe"])
    torch.manual_seed(1)                                          # reproducible dropout masks
    mc = model.predict_with_uncertainty(Xh, T=30)                 # MC-Dropout, 30 stochastic passes
    pm = mc["position_mean"].numpy().ravel() * (FB_L - 1)
    ps = mc["position_std"].numpy().ravel() * (FB_L - 1)         # epistemic spread, samples
    flag = np.where(ps < 3.0, "GOOD", np.where(ps < 10.0, "REVIEW", "REJECT"))   # fold_ensemble.py thresholds
    FC = {"GOOD": C["ok"], "REVIEW": C["warn"], "REJECT": C["bad"]}

    fig, status = fig_base(("MarineSeismicAI", "first-break model learning"),
                           f"V7UncertaintyModel + CombinedV7Loss, trained {float(res['minutes']):.1f} min on CPU on synthetic"
                           " gathers here, not the production weights")
    a1 = fig.add_axes([0.065, 0.135, 0.55, 0.70]); style(a1)
    a2 = fig.add_axes([0.685, 0.56, 0.30, 0.29]); style(a2, grid=True)
    a3 = fig.add_axes([0.685, 0.135, 0.30, 0.29]); style(a3, grid=True)
    tmax = 0.9
    clip = np.percentile(np.abs(Dh), 98)
    gcm = matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#24292F", "#FFFFFF", "#24292F"])
    a1.imshow(Dh, aspect="auto", cmap="Greys", vmin=-clip, vmax=clip, interpolation="bilinear",
              extent=[-0.5, FB_NTR - 0.5, FB_L * FB_DT, 0], alpha=0.85)
    a1.set_ylim(tmax, 0); a1.set_xlim(-0.5, FB_NTR - 0.5)
    a1.set_xticks([0, 32, 64, 95]); a1.set_xticklabels([f"{FB_OFF[i]:.0f}" for i in (0, 32, 64, 95)])
    a1.set_xlabel("offset (m)", fontsize=6, labelpad=1); a1.set_ylabel("time (s)", fontsize=6, labelpad=1)
    lab1 = a1.text(0.0, 1.015, "held-out synthetic shot gather · model picks", transform=a1.transAxes,
                   fontsize=6.5, color=C["mute"])
    tr_idx = np.arange(FB_NTR)
    pk = a1.scatter([], [], s=5, color=C["acc"], lw=0, zorder=5)
    EPX = dict(xscale="log", xlim=(0.035, steps[-1] / spe * 1.08), xticks=[0.1, 1, steps[-1] / spe])
    a2.set(**EPX); a2.set_xticklabels([]); a2.minorticks_off()
    fin = np.isfinite(res["tr"])
    lo = min(np.nanmin(res["tr"][fin]), res["va"].min()); hi = max(np.nanmax(res["tr"][fin]), res["va"].max())
    a2.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    label(a2, "CombinedV7Loss")
    ltr, = a2.plot([], [], color=C["acc"], lw=0.9, label="train")
    lva, = a2.plot([], [], color=C["warn"], lw=0.9, label="val")
    a2.legend(loc="upper right", fontsize=5.5, frameon=False, handlelength=1.2)
    a3.set(**EPX); a3.set_yscale("log"); a3.set_ylim(0.3, 300); a3.minorticks_off()
    a3.set_xticklabels(["0.1", "1", f"{steps[-1] / spe:.0f}"]); a3.set_xlabel("epoch (log)", fontsize=6, labelpad=1)
    label(a3, "val pick error (samples, 2 ms)")
    lmae, = a3.plot([], [], color=C["ink"], lw=0.9)
    frames = []
    for i in range(len(steps)):
        p_ = snaps[i]
        pk.set_offsets(np.c_[tr_idx, p_ * FB_DT])
        ep = steps[i] / spe
        ltr.set_data(steps[:i + 1][fin[:i + 1]] / spe, res["tr"][:i + 1][fin[:i + 1]])   # step 0 off the log axis
        lva.set_data(steps[:i + 1] / spe, res["va"][:i + 1])
        lmae.set_data(steps[:i + 1] / spe, res["mae"][:i + 1])
        status.set_text(f"epoch {ep:4.2f}   val MAE {res['mae'][i]:5.1f} smp")
        frames.append(grab(fig))
    # MC-Dropout uncertainty + QC flags on the trained model
    pk.set_visible(False)
    a1.fill_between(tr_idx, (pm - 2 * ps) * FB_DT, (pm + 2 * ps) * FB_DT, color=C["acc"], alpha=0.25, lw=0,
                    step="mid", zorder=4)
    a1.plot(tr_idx, pm * FB_DT, color=C["acc"], lw=0.8, zorder=5, drawstyle="steps-mid")
    lab1.set_text("MC-Dropout mean pick ± 2σ (30 passes) · QC flag per trace")
    a1.scatter(tr_idx, np.full(FB_NTR, 0.012), c=[FC[f_] for f_ in flag], s=9, marker="s", lw=0, zorder=6)
    for f_, ii in (("REVIEW", flag == "REVIEW"), ("REJECT", flag == "REJECT")):
        for j in np.flatnonzero(ii):
            a1.axvspan(j - 0.5, j + 0.5, color=FC[f_], alpha=0.18, lw=0, zorder=3)
    a3.cla(); style(a3, grid=True)
    a3.bar(tr_idx, ps, width=1.0, color=[FC[f_] for f_ in flag], lw=0)
    a3.set_yscale("log"); a3.set_ylim(0.1, 100); a3.set_xlim(-0.5, FB_NTR - 0.5)
    for v_ in (3.0, 10.0):
        a3.axhline(v_, color=C["ink"], lw=0.5, ls="--")
    a3.set_xticks([0, 32, 64, 95]); a3.set_xticklabels([f"{FB_OFF[i]:.0f}" for i in (0, 32, 64, 95)])
    a3.set_xlabel("offset (m)", fontsize=6, labelpad=1)
    label(a3, "MC-Dropout σ (samples)")
    cnt = {k_: int((flag == k_).sum()) for k_ in ("GOOD", "REVIEW", "REJECT")}
    status.set_text(f"GOOD {cnt['GOOD']}   REVIEW {cnt['REVIEW']}   REJECT {cnt['REJECT']}")
    status.set_color(C["ink"])
    frames.append(grab(fig))
    write_gif(frames, "ai-firstbreak.gif", hold=50)
    return cnt, ps.round(1), (pm - res["fbh"]).round(1), float(res["minutes"])


if __name__ == "__main__":
    which = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"nav", "gt", "fb"}
    if "nav" in which:
        for r in navqc():
            print(r)
    if "gt" in which:
        print(gtcpt())
    if "fb" in which:
        print(firstbreak(sys.argv[6] if len(sys.argv) > 6 else None))
