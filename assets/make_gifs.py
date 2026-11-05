"""Profile animations driven by the real GeoView algorithms on synthetic data.

- SeismicProcessor  seismic_processor/auto/analyzer.py::_sta_lta_first_break (exec'd from source)
- MagQC             magqc/core.py::calc_fourth_diff, fourth_diff_stats
- TidalWorks        tidalworks/ocean_toolkit/harmonic/analysis.py::harmonic_predict
"""
import ast, sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
R = "/home/user/repos"

# ---- real functions ---------------------------------------------------------
src = open(f"{R}/seismicprocessor/seismic_processor/auto/analyzer.py").read()
fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_sta_lta_first_break"][0]
ns = {"np": np}; exec(compile(ast.Module([fn], []), "analyzer.py", "exec"), ns)
sta_lta = ns["_sta_lta_first_break"]
spec = importlib.util.spec_from_file_location("mqcore", f"{R}/magqc/magqc/core.py")
mq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mq)
sys.path.insert(0, f"{R}/tidalworks")
from tidalworks.ocean_toolkit.harmonic.analysis import harmonic_predict

THEMES = {
    "light": dict(bg="#F6F1E7", ink="#1F2A33", mute="#7A776E", hair="#DDD4C2", acc="#9A4F2B", bad="#B0342A"),
    "dark":  dict(bg="#11161B", ink="#E8E2D6", mute="#8E8E86", hair="#2A333C", acc="#D08A5C", bad="#E0675B"),
}
W, H, DPI = 6.4, 2.6, 100
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 8})


def frame(th, title, foot):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=th["bg"])
    fig.text(0.015, 0.93, title, color=th["ink"], fontsize=9, fontweight="bold")
    fig.text(0.015, 0.03, foot, color=th["mute"], fontsize=7)
    return fig


def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_color(th["hair"])
    ax.tick_params(colors=th["mute"], labelsize=7, length=2)
    ax.grid(color=th["hair"], lw=0.5)


def save(anim, name, fps):
    anim.save(OUT / name, writer=PillowWriter(fps=fps))
    plt.close("all")


# ---- 1. seismic first-break -------------------------------------------------
def seismic(th, name):
    rng = np.random.default_rng(7)
    dt, ns_, ntr = 0.00025, 760, 24
    off = np.linspace(10, 190, ntr); t = np.arange(ns_) * dt
    tfb = np.sqrt(0.045**2 + (off / 1500.0) ** 2)
    data = np.zeros((ntr, ns_))
    for i in range(ntr):
        for t0, a in ((tfb[i], 1.0), (tfb[i] * 1.9, 0.45)):
            arg = (np.pi * 140 * (t - t0)) ** 2
            data[i] += a * (1 - 2 * arg) * np.exp(-arg)
        data[i] += 0.035 * rng.standard_normal(ns_)
    picks = np.array([sta_lta(tr, dt) for tr in data])
    fig = frame(th, "SeismicProcessor  ·  STA/LTA first-break picker",
                "synthetic marine shot gather  ·  picks from the program's own picker")
    ax = fig.add_axes([0.07, 0.17, 0.91, 0.70]); style(ax, th)
    ax.set_xlim(0, ntr + 1); ax.set_ylim(t[-1] * 1000, 25); ax.set_ylabel("ms", color=th["mute"], fontsize=7)
    ax.set_xticks([])
    for i, tr in enumerate(data):
        x = i + 1 + tr * 0.9
        ax.plot(x, t * 1000, color=th["ink"], lw=0.5)
        ax.fill_betweenx(t * 1000, i + 1, x, where=tr > 0, color=th["ink"], lw=0)
    sc = ax.scatter([], [], s=14, color=th["acc"], zorder=5)
    ln, = ax.plot([], [], color=th["acc"], lw=1.2, zorder=4)
    txt = fig.text(0.98, 0.93, "", ha="right", color=th["acc"], fontsize=8)

    def upd(k):
        n = min(k, ntr)
        ok = picks[:n] > 0
        xs = np.arange(1, n + 1)[ok]; ys = picks[:n][ok] * 1000
        sc.set_offsets(np.c_[xs, ys] if len(xs) else np.empty((0, 2)))
        ln.set_data(xs, ys)
        txt.set_text(f"picked {int(ok.sum())}/{ntr}")
        return sc, ln, txt
    save(FuncAnimation(fig, upd, frames=ntr + 14, blit=False), name, 12)


# ---- 2. MagQC 4th difference ------------------------------------------------
def mag(th, name):
    rng = np.random.default_rng(3)
    n = 900; x = np.arange(n)
    field = 50120 + 0.004 * x + 8 * np.exp(-((x - 520) / 35) ** 2) - 4 * np.exp(-((x - 570) / 30) ** 2)
    field += 0.02 * rng.standard_normal(n)
    for s in (210, 690, 691):
        field[s] += 0.6
    fd = np.array(mq.calc_fourth_diff(list(field)))
    thr = 0.5
    st = mq.fourth_diff_stats(list(fd), threshold=thr)
    fig = frame(th, "MagQC  ·  4th-difference noise check",
                f"synthetic magnetometer line  ·  limit ±{thr} nT  ·  calc_fourth_diff / fourth_diff_stats")
    a1 = fig.add_axes([0.08, 0.50, 0.90, 0.37]); a2 = fig.add_axes([0.08, 0.15, 0.90, 0.30])
    for a in (a1, a2):
        style(a, th); a.set_xlim(0, n)
    a1.set_xticks([]); a1.set_ylim(field.min() - 1, field.max() + 1); a1.set_ylabel("nT", color=th["mute"], fontsize=7)
    a2.set_ylim(-3, 3); a2.set_xticks([]); a2.set_ylabel("FD", color=th["mute"], fontsize=7)
    for y in (thr, -thr):
        a2.axhline(y, color=th["mute"], lw=0.6, ls="--")
    l1, = a1.plot([], [], color=th["ink"], lw=0.9)
    l2, = a2.plot([], [], color=th["acc"], lw=0.7)
    bad = a2.scatter([], [], s=16, color=th["bad"], zorder=5)
    txt = fig.text(0.98, 0.93, "", ha="right", color=th["acc"], fontsize=8)
    steps = 40

    def upd(k):
        m = int(min(k, steps) / steps * n)
        l1.set_data(x[:m], field[:m])
        mf = max(0, m - 4); xf = x[2:2 + mf]; f = fd[:mf]
        l2.set_data(xf, f)
        over = np.abs(f) > thr
        bad.set_offsets(np.c_[xf[over], f[over]] if over.any() else np.empty((0, 2)))
        if k >= steps:
            txt.set_text(f"exceed {st['exceedance_count']}/{st['n']}  ({st['exceedance_pct']:.2f}%)")
        else:
            txt.set_text(f"exceed {int(over.sum())}")
        return l1, l2, bad, txt
    save(FuncAnimation(fig, upd, frames=steps + 15, blit=False), name, 12)


# ---- 3. TidalWorks harmonic prediction --------------------------------------
def tide(th, name):
    tt = np.arange("2026-03-01", "2026-03-16", np.timedelta64(30, "m"), dtype="datetime64[m]")
    cons = {"M2": (1.20, 110.0), "S2": (0.48, 140.0), "K1": (0.32, 250.0), "O1": (0.24, 220.0)}
    y, method = harmonic_predict(tt, cons, 35.1, reftime=tt[0], return_method=True)
    hrs = (tt - tt[0]).astype(float) / 60.0
    fig = frame(th, "TidalWorks  ·  harmonic tide prediction",
                f"M2 S2 K1 O1, lat 35.1°N  ·  harmonic_predict ({method})  ·  15 days")
    ax = fig.add_axes([0.07, 0.17, 0.91, 0.70]); style(ax, th)
    ax.set_xlim(0, hrs[-1]); ax.set_ylim(y.min() * 1.15, y.max() * 1.15)
    ax.set_xticks(np.arange(0, hrs[-1] + 1, 48)); ax.set_xticklabels([f"d{int(h/24)}" for h in np.arange(0, hrs[-1] + 1, 48)])
    ax.set_ylabel("m", color=th["mute"], fontsize=7)
    ln, = ax.plot([], [], color=th["ink"], lw=0.8)
    dot = ax.scatter([], [], s=14, color=th["acc"], zorder=5)
    txt = fig.text(0.98, 0.93, "", ha="right", color=th["acc"], fontsize=8)
    steps = 50

    def upd(k):
        m = max(1, int(min(k, steps) / steps * len(hrs)))
        ln.set_data(hrs[:m], y[:m]); dot.set_offsets([[hrs[m - 1], y[m - 1]]])
        txt.set_text(f"{str(tt[m-1])[:16].replace('T', ' ')}   {y[m-1]:+.2f} m")
        return ln, dot, txt
    save(FuncAnimation(fig, upd, frames=steps + 12, blit=False), name, 12)


for theme, th in THEMES.items():
    seismic(th, f"seismic-firstbreak-{theme}.gif")
    mag(th, f"magqc-fourthdiff-{theme}.gif")
    tide(th, f"tidalworks-tide-{theme}.gif")
print("done")
