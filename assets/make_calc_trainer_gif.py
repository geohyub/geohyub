"""Profile animation for GeoView Calculator Pro, same style as make_visuals.py (palette, figure size, helpers
copied below). QCInterpretationTrainer is skipped (see report): it is a Next.js case-reading app with authored
JSON cases and no headless computation to animate.

  calculator-mbes-plan.gif   GeoView_Calculator_Pro  geoview_calculator_pro/formulas.py
                               mackenzie_sv          (sound speed of a synthetic CTD cast, harmonic mean to depth)
                               mbes_calculate        (swath, PRF, along-track step, sounding density)
                               mbes_survey_planning  (line count, line km, survey hours incl. turns)
                               iho_uncertainty       (IHO S-44 TVU allowance, presets.IHO_ORDERS)
                             inputs are the MBES Planning tab defaults (tab_planning._calculate): 10 km2 block,
                             20 % overlap, 5 min turns, 130 deg fan, 256 beams, 10 % range margin, 4 kn.

Water depth is swept 10 -> 100 m and back to 40 m; every number drawn (fan width, line spacing, PRF, density,
hours, TVU) comes back from those functions. The CTD cast (temperature / salinity profile) is synthetic and
generated below; the beam fan, the square block and the stripe layout are drawn here from the returned swath
and line count (the app draws its own diagram). No survey, vessel or project names are used.

usage:  python make_calc_trainer_gif.py <out_dir> [calculator_repo] [shared_root]
needs:  numpy, matplotlib, pillow
"""
import sys, types, math
sys.dont_write_bytecode = True                     # import from the repo without leaving __pycache__ there
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
CALC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/Utilities/GeoView_Calculator_Pro")
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
    foot = [foot] if isinstance(foot, str) else foot
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
# GeoView Calculator Pro - MBES coverage and survey time as the water gets deeper
# =========================================================================================
def calculator():
    sys.path[:0] = [str(CALC), str(SHARED)]
    pkg("geoview_common", SHARED / "geoview_common")             # skip the heavy package __init__s,
    pkg("geoview_common.geo", SHARED / "geoview_common/geo")     # formulas.py only needs geo/crs.py
    from geoview_calculator_pro import formulas as F
    from geoview_calculator_pro.presets import IHO_ORDERS

    # MBES Planning tab defaults (tab_planning._calculate)
    AREA, OVL, TURN, ANG, BEAMS, MARGIN, KN = 10.0, 20.0, 5.0, 130.0, 256, 10.0, 4.0
    SIDE = math.sqrt(AREA * 1e6)

    # synthetic CTD cast: warm mixed layer, thermocline 15-35 m, slight salinity rise
    zc = np.arange(0.0, 101.0, 1.0)
    T = 9.0 + 9.0 / (1 + np.exp((zc - 25) / 4.0))
    S = 33.6 + 0.6 / (1 + np.exp(-(zc - 30) / 6.0))
    sv = np.array([F.mackenzie_sv(t, s, z) for t, s, z in zip(T, S, zc)])

    def sv_mean(d):                                # harmonic mean of the cast down to d (travel-time mean)
        m = zc <= d
        return float(m.sum() / np.sum(1.0 / sv[m]))

    def solve(d):
        c = sv_mean(d)
        r = F.mbes_calculate(c, KN, BEAMS, ANG, MARGIN, d)
        n, km, h = F.mbes_survey_planning(AREA, r["swath"], OVL, KN, TURN)
        return dict(d=d, c=c, n=n, km=km, h=h, sp=r["swath"] * (1 - OVL / 100),
                    tvu_s=F.iho_uncertainty(d, IHO_ORDERS["Special"]),
                    tvu_1a=F.iho_uncertainty(d, IHO_ORDERS["Order 1a"]), **r)

    dgrid = np.linspace(10, 100, 361)
    G = [solve(d) for d in dgrid]
    gh = np.array([g["h"] for g in G]); gn = np.array([g["n"] for g in G])
    gprf = np.array([g["prf"] for g in G]); gden = np.array([g["density"] for g in G])

    foot = ["GeoView Calculator Pro formulas: mackenzie_sv · mbes_calculate · mbes_survey_planning"
            " · iho_uncertainty",
            "Planning tab defaults: 10 km², 20 % overlap, 130°, 256 beams, 4 kn"
            "  ·  synthetic CTD  ·  diagrams drawn here"]
    fig, status = fig_base(("GeoView Calculator Pro", "MBES coverage vs depth"), foot)
    a1 = fig.add_axes([0.065, 0.20, 0.285, 0.65]); style(a1)
    a2 = fig.add_axes([0.385, 0.20, 0.20, 0.65]); style(a2)
    a3 = fig.add_axes([0.655, 0.60, 0.33, 0.25]); style(a3, grid=True)
    ac = fig.add_axes([0.655, 0.14, 0.33, 0.36]); ac.axis("off")

    # cross-section: fan from the transducer to a flat seabed
    HALF = 0.5 * 2 * 100 * math.tan(math.radians(ANG / 2))
    a1.set_xlim(-HALF * 1.04, HALF * 1.04); a1.set_ylim(112, -6)
    a1.xaxis.set_major_formatter(COMMA)
    a1.set_xticks([-200, -100, 0, 100, 200])
    a1.set_xlabel("across-track (m)", fontsize=6, labelpad=1)
    a1.set_ylabel("depth (m)", fontsize=6, labelpad=1)
    lab1 = label(a1, "130° fan · swath")
    angs = np.radians(np.linspace(-ANG / 2, ANG / 2, 27))
    rays = [a1.plot([], [], color=C["acc"], lw=0.45, alpha=0.55)[0] for _ in angs]
    bed, = a1.plot([], [], color=C["ink"], lw=1.0)
    sw_bar, = a1.plot([], [], color=C["warn"], lw=1.6, solid_capstyle="butt")
    sw_txt = a1.text(0, 0, "", ha="center", va="top", fontsize=6.3, color=C["warn"])
    a1.scatter([0], [0], marker="v", s=18, color=C["ink"], zorder=5)
    a1.axhline(0, color=C["mute"], lw=0.5, ls=(0, (3, 3)))

    # plan view: square block with planned lines and swath stripes
    a2.set_xlim(0, SIDE); a2.set_ylim(0, SIDE); a2.set_aspect("equal", adjustable="box")
    a2.set_xticks([0, 1000, 2000, 3000]); a2.set_yticks([0, 1000, 2000, 3000])
    a2.xaxis.set_major_formatter(COMMA); a2.yaxis.set_major_formatter(COMMA)
    a2.set_xlabel("east (m)", fontsize=6, labelpad=1)
    lab2 = label(a2, "10 km² block")
    stripes = []

    # survey hours vs depth, drawn as the sweep goes
    a3.set_xlim(10, 100); a3.set_ylim(0, gh.max() * 1.08)
    a3.yaxis.set_major_formatter(COMMA)
    a3.set_xlabel("water depth (m)", fontsize=6, labelpad=1)
    label(a3, "survey time incl. turns (h)")
    lh, = a3.plot([], [], color=C["acc"], lw=0.9)
    dot, = a3.plot([], [], "o", ms=3.2, color=C["acc"], zorder=5)

    ROWS = [("sound speed", lambda g: f"{g['c']:,.1f} m/s"),
            ("swath", lambda g: f"{g['swath']:,.1f} m"),
            ("line spacing", lambda g: f"{g['sp']:,.1f} m"),
            ("PRF", lambda g: f"{g['prf']:,.2f} Hz"),
            ("along-track step", lambda g: f"{g['along_track_step']:,.2f} m"),
            ("density", lambda g: f"{g['density']:,.1f} /m²"),
            ("TVU Special / 1a", lambda g: f"{g['tvu_s']:,.2f} / {g['tvu_1a']:,.2f} m")]
    rt = []
    for i, (nm, _) in enumerate(ROWS):
        y = 0.95 - i * 0.133
        ac.text(0.0, y, nm, fontsize=6.2, color=C["mute"], transform=ac.transAxes, va="center")
        rt.append(ac.text(1.0, y, "", fontsize=6.2, color=C["ink"], ha="right",
                          transform=ac.transAxes, va="center"))
    plan_txt = ac.text(0.0, 0.0, "", fontsize=7, color=C["ink"], fontweight="bold",
                       transform=ac.transAxes, va="center")

    def draw(g, trail):
        d, sw = g["d"], g["swath"]
        for a_, r_ in zip(angs, rays):
            r_.set_data([0, d * math.tan(a_)], [0, d])
        bed.set_data([-HALF * 1.04, HALF * 1.04], [d, d])
        sw_bar.set_data([-sw / 2, sw / 2], [d + 3, d + 3])
        sw_txt.set_position((0, d + 5)); sw_txt.set_text(f"{sw:,.0f} m")
        sw_txt.set_y(d + 5 if d < 95 else d + 4)
        lab1.set_text(f"130° fan at {d:,.0f} m · PRF {g['prf']:,.1f} Hz")
        for s_ in stripes:
            s_.remove()
        stripes.clear()
        n, sp = g["n"], g["sp"]
        for i in range(n):                         # line i at the centre of its effective strip
            xc = sp * (i + 0.5)
            stripes.append(a2.axvspan(max(0, xc - sw / 2), min(SIDE, xc + sw / 2),
                                      color=C["acc"], alpha=0.16, lw=0))
            if n <= 60:
                stripes.append(a2.axvline(xc, color=C["acc"], lw=0.4))
        lab2.set_text(f"{n:,} lines · {g['km']:,.1f} line km")
        m = dgrid <= d + 1e-9
        if trail:
            lh.set_data(dgrid[m], gh[m])
        dot.set_data([d], [g["h"]])
        for (nm, fm), t_ in zip(ROWS, rt):
            t_.set_text(fm(g))
        status.set_text(f"depth {d:,.0f} m   {n:,} lines   {g['h']:,.1f} h")

    frames = []
    ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * u)
    for u in np.linspace(0, 1, 60):                # sweep 10 -> 100 m
        draw(solve(10 + 90 * ease(u)), True)
        frames.append(grab(fig))
    frames += [grab(fig)] * 4
    for u in np.linspace(0, 1, 16)[1:]:            # back to a 40 m block
        draw(solve(100 - 60 * ease(u)), True)
        lh.set_data(dgrid, gh)
        frames.append(grab(fig))
    g = solve(40.0)
    draw(g, True); lh.set_data(dgrid, gh)
    frames += [grab(fig)] * 3
    for t_ in rt:                                  # read out the 40 m answer row by row
        t_.set_color(C["acc"]); t_.set_fontweight("bold")
        frames += [grab(fig)] * 2
        t_.set_fontweight("normal")
    plan_txt.set_text(f"40 m: {g['n']:,} lines · {g['km']:,.1f} km · {g['h']:,.1f} h")
    plan_txt.set_color(C["ok"])
    status.set_text(f"plan at 40 m -> {g['h']:,.1f} h"); status.set_color(C["ok"])
    frames.append(grab(fig))
    write_gif(frames, "calculator-mbes-plan.gif", hold=50, mid=40)
    g10, g100 = solve(10.0), solve(100.0)
    return dict(at40={k: g[k] for k in ("c", "swath", "sp", "prf", "density", "n", "km", "h", "tvu_s", "tvu_1a")},
                at10=(g10["n"], g10["h"]), at100=(g100["n"], g100["h"]), prf_range=(gprf.min(), gprf.max()),
                dens_range=(gden.min(), gden.max()))


if __name__ == "__main__":
    print(calculator())
