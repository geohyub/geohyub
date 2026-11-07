"""UHRSeismicAI profile animation. Trains the real repo refiner on synthetic 96-channel shot gathers, on CPU.

  uhr-picking.gif   UHRSeismicAI  models/refiner.py::SeabedMultiChannelRefiner, SeabedMultiChannelConfig
                                  (one model per target, per-channel residual head - as scripts/uhr_train_shot.py)
                                  training/losses.py::picking_loss_direct, picking_loss_seabed
                                  core/geometry.py::direct_arrival_samples, hyperbolic_seabed_samples
                                  (physics guess that centres each channel's window, as uhr_build_shot_cache.py)

Synthetic gathers only: water-column direct arrival, a dipping seabed hyperbola, two weaker sub-bottom
reflectors, gaussian + swell noise. Per-channel truth = physics + unknown water velocity, trigger delay,
streamer feathering and statics, so the physics guess is several samples off and the model has to learn
the residual. Weights are trained here for a few minutes - they are not the production weights.
The model is smaller than production (base_channels=12, 96 x 64 window instead of 32 / 128-256) to fit a
CPU budget. Style (palette, write_gif, title/footer) is copied from make_visuals_ml.py.

usage:  python make_uhr_gif.py <out_dir> [uhr_src] [png_dir] [ckpt_dir]
needs:  numpy, matplotlib, pillow, torch (CPU is fine)
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image
import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/AI/uhr_seismic_ai/src")
LAST_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if LAST_DIR:
    LAST_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(sys.argv[4]) if len(sys.argv) > 4 else (LAST_DIR or OUT) / "_ckpt"
SCRATCH.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)

sys.path.insert(0, str(SRC))
from uhr_seismic_ai.models.refiner import SeabedMultiChannelRefiner, SeabedMultiChannelConfig
from uhr_seismic_ai.training.losses import picking_loss_direct, picking_loss_seabed
from uhr_seismic_ai.core.geometry import direct_arrival_samples, hyperbolic_seabed_samples

C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
SEIS = LinearSegmentedColormap.from_list("seis", ["#0A3069", "#54AEFF", "#FFFFFF", "#FF8182", "#82071E"])
DCOL, SCOL = C["acc"], C["bad"]                 # direct arrival / seabed


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
    for hx in C.values():
        rgb += [int(hx[i:i + 2], 16) for i in (1, 3, 5)]
    ref = Image.new("P", (1, 1)); ref.putpalette(rgb + [0] * (768 - len(rgb)))
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / FPS)] * len(pal); dur[-1] = int(1000 / FPS) * hold
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=dur, loop=0,
                optimize=True, disposal=1)
    if LAST_DIR:
        frames[-1].save(LAST_DIR / (Path(name).stem + "-last.png"))
        frames[len(frames) // 2].save(LAST_DIR / (Path(name).stem + "-mid.png"))
    print(f"{name}: {len(pal)} frames, {(OUT / name).stat().st_size / 1024:.0f} KB")
    plt.close("all")


def took(sec):
    return f"{sec / 60:.0f} min" if sec >= 90 else f"{sec:.0f} s"


def ricker(f, n=41):
    t = np.arange(n) - n // 2
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


# ---- synthetic 96-channel UHR shot gather --------------------------------------------------
NCH, NS, DT_US, PW = 96, 640, 250, 64          # channels, samples, 0.25 ms, model window
OFF = 8.0 + 1.5625 * np.arange(NCH)            # nominal source-receiver offsets (m)
MAXR = 9.0                                     # keep |truth - guess| inside the tanh x 10 head


def make_shot(rng):
    """Returns gather (96, NS), truth picks and physics-guess picks for direct + seabed."""
    dt_ms = DT_US / 1000.0
    k = np.arange(NCH) / (NCH - 1)
    feather = rng.uniform(-1.5, 1.5) * k ** 2 + rng.normal(0, 0.15, NCH)   # actual - nominal offset (m)
    off = OFF + feather
    stat = np.convolve(rng.normal(0, 0.6, NCH + 8), np.ones(9) / 9, "valid")  # smooth per-channel statics
    v_w = rng.uniform(1475, 1530); delay = rng.uniform(-3.0, 3.0)
    da_true = off / v_w * 1000.0 / dt_ms + delay + 0.5 * stat
    da_guess = direct_arrival_samples(OFF, DT_US, v_water=1500.0)          # what the cache builder does
    depth = rng.uniform(30, 50); v_sb = rng.uniform(1490, 1560); dip = rng.uniform(-0.05, 0.05)
    t0 = 2 * depth / v_sb * 1000.0 / dt_ms
    sb_true = hyperbolic_seabed_samples(off, t0, DT_US, v_nmo=v_sb) + dip * off + stat
    t0_guess = t0 + rng.uniform(-5, 5)                                       # e.g. echo-sounder depth
    sb_guess = hyperbolic_seabed_samples(OFF, t0_guess, DT_US, v_nmo=1500.0)
    da_true = da_guess + np.clip(da_true - da_guess, -MAXR, MAXR)
    sb_true = sb_guess + np.clip(sb_true - sb_guess, -MAXR, MAXR)

    t = np.arange(NS)
    g = np.zeros((NCH, NS))
    w = ricker(rng.uniform(0.07, 0.10))
    half = len(w) // 2

    def spike(row, pos, amp):
        i = int(np.floor(pos)); fr = pos - i
        for j, a in ((i, 1 - fr), (i + 1, fr)):
            if 0 <= j < NS:
                row[j] += amp * a
    sub = [(rng.uniform(25, 60), rng.uniform(0.15, 0.3)), (rng.uniform(70, 130), rng.uniform(0.1, 0.2))]
    for c in range(NCH):
        row = np.zeros(NS)
        spike(row, da_true[c], 1.6 / (1 + off[c] / 40))
        spike(row, sb_true[c], -0.8 * rng.uniform(0.85, 1.15))
        for dz, amp in sub:
            spike(row, sb_true[c] + dz * (1 - 0.15 * k[c]), amp * rng.choice([-1, 1]))
        g[c] = np.convolve(row, w, "same")
    # picks sit on the wavelet peak; noise: gaussian + low-frequency swell
    g += rng.normal(0, rng.uniform(0.03, 0.09), g.shape)
    sw = np.sin(2 * np.pi * (t[None, :] / rng.uniform(150, 300) + rng.uniform(0, 1) + 0.02 * np.arange(NCH)[:, None]))
    g += rng.uniform(0.0, 0.06) * sw
    g /= (np.percentile(np.abs(g), 99) or 1.0)                               # as uhr_build_shot_cache.py
    return g.astype(np.float32), off.astype(np.float32), da_true, da_guess, sb_true, sb_guess


def window(g, guess):
    centers = np.round(guess).astype(int)
    out = np.zeros((NCH, PW), np.float32)
    for c in range(NCH):
        s0 = centers[c] - PW // 2
        a, b = max(0, s0), min(NS, s0 + PW)
        if b > a:
            out[c, a - s0:b - s0] = g[c, a:b]
    return out, centers


def build(n, seed):
    rng = np.random.default_rng(seed)
    P = {k: [] for k in ("gather", "off", "pd", "td", "cd", "ps", "ts", "cs")}
    for _ in range(n):
        g, off, dt_, dg, st, sg = make_shot(rng)
        pd_, cd = window(g, dg); ps, cs = window(g, sg)
        P["gather"].append(g); P["off"].append(off)
        P["pd"].append(pd_); P["td"].append(dt_ - cd); P["cd"].append(cd)
        P["ps"].append(ps); P["ts"].append(st - cs); P["cs"].append(cs)
    return {k: np.stack(v) for k, v in P.items()}


# ---- training -------------------------------------------------------------------------------
N_TRAIN, N_VAL, EPOCHS, BATCH, LR, EVERY = 640, 48, 14, 16, 1e-3, 8


def train():
    cache = SCRATCH / "uhr_snaps.npz"
    val = build(N_VAL, seed=2026)                                   # held out: own seed, never trained on
    if cache.exists():
        z = np.load(cache)
        return val, z["step"], z["pred_d"], z["pred_s"], float(z["train_s"]), int(z["spe"])
    tr = build(N_TRAIN, seed=7)
    torch.manual_seed(0)
    models, opts, scheds = {}, {}, {}
    spe = N_TRAIN // BATCH
    for key in ("d", "s"):
        cfg = SeabedMultiChannelConfig(patch_h=NCH, patch_w=PW, base_channels=12,
                                       max_offset_samples=(8.0 if key == "d" else 10.0),
                                       use_recon_aux=True, per_channel_residual=True)
        models[key] = SeabedMultiChannelRefiner(cfg)
        opts[key] = torch.optim.AdamW(models[key].parameters(), lr=LR, weight_decay=0.05)
        scheds[key] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[key], EPOCHS * spe, eta_min=LR * 0.01)
    vd = torch.from_numpy(val["pd"])[:, None]; vs = torch.from_numpy(val["ps"])[:, None]

    def predict():
        out = []
        for key, x in (("d", vd), ("s", vs)):
            models[key].eval()
            with torch.no_grad():
                out.append(models[key](x)["residual"].numpy().copy())
            models[key].train()
        return out
    steps, pdl, psl = [0], *[[a] for a in predict()]
    rng = np.random.default_rng(11)
    t0 = time.time(); step = 0
    for ep in range(EPOCHS):
        perm = rng.permutation(N_TRAIN)
        for b in range(spe):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            off = torch.from_numpy(tr["off"][idx])
            for key in ("d", "s"):
                x = torch.from_numpy(tr["p" + key][idx])[:, None]
                y = torch.from_numpy(tr["t" + key][idx].astype(np.float32))
                out = models[key](x)
                if key == "d":
                    loss, _ = picking_loss_direct(out, y, off, target_patch=x)
                else:
                    # lambda_hyper=0: the default 0.3 hyperbolic term is fed residual-space values
                    # (target + residual, often negative) and pushed every pick late; held-out error
                    # rose from 4.0 (guess) to 7.3 samples. With the term off: 0.3-0.8 (5-epoch check).
                    loss, _ = picking_loss_seabed(out, y, off, DT_US, target_patch=x, lambda_hyper=0.0)
                opts[key].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(models[key].parameters(), 1.0)
                opts[key].step(); scheds[key].step()
            step += 1
            if (step <= 40 and step % 2 == 0) or step % EVERY == 0:      # dense early, where it learns fastest
                a, s_ = predict(); steps.append(step); pdl.append(a); psl.append(s_)
        print(f"epoch {ep + 1}/{EPOCHS}  {time.time() - t0:.0f}s  "
              f"val MAE direct {np.abs(pdl[-1] - val['td']).mean():.2f}  seabed {np.abs(psl[-1] - val['ts']).mean():.2f}")
    train_s = time.time() - t0
    np.savez(cache, step=np.array(steps), pred_d=np.stack(pdl), pred_s=np.stack(psl), train_s=train_s, spe=spe)
    return val, np.array(steps), np.stack(pdl), np.stack(psl), train_s, spe


# ---- render ---------------------------------------------------------------------------------
def render():
    val, steps, pred_d, pred_s, train_s, spe = train()
    td, ts = val["td"], val["ts"]
    mae_d = np.abs(pred_d - td[None]).mean(axis=(1, 2)); mae_s = np.abs(pred_s - ts[None]).mean(axis=(1, 2))
    g_d, g_s = np.abs(td).mean(), np.abs(ts).mean()                 # error of the physics guess alone
    print(f"guess MAE direct {g_d:.2f} seabed {g_s:.2f} | model start {mae_d[0]:.2f}/{mae_s[0]:.2f}"
          f" end {mae_d[-1]:.2f}/{mae_s[-1]:.2f} samples | trained {train_s:.0f}s")
    # shown gather: held-out shot whose final error is closest to the median (not the best one)
    per = np.abs(pred_s[-1] - ts).mean(1) + np.abs(pred_d[-1] - td).mean(1)
    i = int(np.argsort(per)[len(per) // 2])
    g = val["gather"][i]; cd, cs = val["cd"][i], val["cs"][i]
    ch = np.arange(1, NCH + 1)

    fig, status = fig_base(("UHRSeismicAI", "learning per-channel picks"),
                           "SeabedMultiChannelRefiner (models/refiner.py, base 12 ch, 96 x 64 window) · picking_loss_direct ·"
                           " picking_loss_seabed\n(hyperbolic term off) · guess: core/geometry.py · trained"
                           f" {took(train_s)} on CPU on synthetic gathers - not production weights")
    clip = np.percentile(np.abs(g), 98)
    gkw = dict(aspect="auto", cmap="Greys", vmin=-clip, vmax=clip, interpolation="bilinear", alpha=0.85)
    # A: the whole held-out shot gather, model picks on top
    a1 = fig.add_axes([0.055, 0.17, 0.27, 0.69]); style(a1)
    tmax = int(min(NS, cs.max() + 70))
    ms = DT_US / 1000
    a1.imshow(g[:, :tmax].T, extent=(0.5, NCH + 0.5, tmax * ms, 0), **gkw)
    a1.set_xticks([1, 32, 64, 96])
    a1.set_xlabel("channel", fontsize=6, labelpad=1); a1.set_ylabel("time (ms)", fontsize=6, labelpad=1)
    label(a1, "held-out synthetic shot")
    dd = a1.scatter([], [], s=3, color=DCOL, lw=0, zorder=5)
    ds = a1.scatter([], [], s=3, color=SCOL, lw=0, zorder=5)
    # B: the two model inputs - one window per channel, centred on the physics guess
    axd = fig.add_axes([0.40, 0.545, 0.245, 0.315]); axs = fig.add_axes([0.40, 0.17, 0.245, 0.315])
    tops = []
    for ax, patch, tr_, col, nm in ((axd, val["pd"][i], td[i], DCOL, "direct arrival input"),
                                     (axs, val["ps"][i], ts[i], SCOL, "seabed input")):
        style(ax); ax.set_xticks([]); ax.set_yticks([-8, 0, 8]); ax.tick_params(labelsize=5.5)
        ax.imshow(patch.T, extent=(0.5, NCH + 0.5, PW / 2, -PW / 2), **gkw)
        ax.axhline(0, color=C["mute"], lw=0.8, ls=(0, (3, 2)))
        ax.plot(ch, tr_, color=C["ink"], lw=0.8)
        tops.append(ax.scatter([], [], s=4, color=col, lw=0, zorder=5))
        ax.set_ylim(12, -12); ax.set_xlim(0.5, NCH + 0.5)
        label(ax, nm)
    axs.set_xlabel("channel 1-96   line truth · dots model", fontsize=6, labelpad=1)
    fig.text(0.358, 0.515, "samples from physics guess (dashed)", rotation=90, ha="center", va="center",
             color=C["mute"], fontsize=6)
    # C: held-out error, log epoch axis (most of the learning is in the first epoch)
    a3 = fig.add_axes([0.725, 0.225, 0.26, 0.635]); style(a3, grid=True)
    label(a3, f"held-out error ({N_VAL} shots)")
    X0 = 0.025
    ep = np.maximum(steps / spe, X0)
    a3.set_xscale("log"); a3.set_xlim(X0 * 0.85, EPOCHS * 1.15); a3.minorticks_off()
    a3.set_xticks([X0, 0.1, 1, EPOCHS]); a3.set_xticklabels(["0", "0.1", "1", f"{EPOCHS}"])
    a3.set_ylim(0, max(g_d, g_s, mae_d.max(), mae_s.max()) * 1.18)
    a3.set_xlabel("epoch (log)", fontsize=6, labelpad=1); a3.set_ylabel("samples (0.25 ms)", fontsize=6, labelpad=1)
    for gv, col, nm in ((g_d, DCOL, "direct"), (g_s, SCOL, "seabed")):
        a3.axhline(gv, color=col, lw=0.7, ls=(0, (3, 2)), alpha=0.8)
        a3.text(EPOCHS * 1.05, gv, f"{nm} guess", ha="right", va="bottom", color=col, fontsize=5.5)
    ld, = a3.plot([], [], color=DCOL, lw=1.3); ls_, = a3.plot([], [], color=SCOL, lw=1.3)
    td_txt = a3.text(0.97, 0.93, "", transform=a3.transAxes, ha="right", va="top", color=DCOL, fontsize=6)
    ts_txt = a3.text(0.97, 0.85, "", transform=a3.transAxes, ha="right", va="top", color=SCOL, fontsize=6)

    frames = []
    for k in range(len(steps)):
        rd, rs = pred_d[k, i], pred_s[k, i]
        dd.set_offsets(np.c_[ch, (cd + rd) * ms]); ds.set_offsets(np.c_[ch, (cs + rs) * ms])
        tops[0].set_offsets(np.c_[ch, rd]); tops[1].set_offsets(np.c_[ch, rs])
        ld.set_data(ep[:k + 1], mae_d[:k + 1]); ls_.set_data(ep[:k + 1], mae_s[:k + 1])
        td_txt.set_text(f"direct  {mae_d[k]:.2f}"); ts_txt.set_text(f"seabed  {mae_s[k]:.2f}")
        status.set_text(f"epoch {steps[k] / spe:4.1f}/{EPOCHS}")
        frames.append(grab(fig))
        if k == 0:
            frames += [frames[-1]] * 8
    write_gif(frames, "uhr-picking.gif", hold=48)
    return dict(train_s=train_s, guess=(g_d, g_s), start=(mae_d[0], mae_s[0]), end=(mae_d[-1], mae_s[-1]))


if __name__ == "__main__":
    print(render())
