"""VelocityPicking profile animation. Trains the real repo model on synthetic CMP gathers, on CPU.

  velocity-picking.gif  VelocityPicking  models/velocity_model.py::VelocityPickerModel
                                          (Swin-T encoder, ImageNet init from the local timm cache,
                                           + 6-layer Transformer decoder -> P(v|t) over 261 bins)
                                          training/losses.py::VelocityPickingLoss (Gaussian soft target CE)
                                          core/semblance.py::compute_semblance, velocity_axis
                                          core/preprocessing.py::semblance_to_rgb, velocity_to_bin_index
                                          core/velocity_function.py::interval_to_rms
                                          core/viterbi.py::viterbi_decode_vectorized
                                          core/topk_decoder.py::topk_decode
                                          evaluation/metrics.py::velocity_mae, top1_mae, oracle_mae

Shown: a held-out synthetic gather, its semblance (primaries + water-bottom and peg-leg multiples, so the
strongest semblance is not always the primary trend), the model's P(v|t) on that semblance sharpening step by
step with its Viterbi pick, and the held-out pick error (below the seabed) against a no-ML baseline
(Viterbi run straight on the semblance). At the end topk_decode turns the last P(v|t) into 5 candidates.

Weights are trained here, for a few minutes, on synthetic data - they are not the production weights.
Training settings differ from training/train.py (224 time samples instead of 512, 420 steps, lr 2e-4,
no AMP) so it fits in about 10 minutes on CPU. Style (palette, write_gif, title/footer) is copied from
make_visuals_ml.py.

usage:  python make_velocity_gif.py <out_dir> [repo_src] [png_dir]
needs:  numpy, scipy, matplotlib, pillow, torch (CPU), timm with the swin_tiny_patch4_window7_224 weights
        cached (HF_HUB_OFFLINE=1 is set, nothing is downloaded)
"""
import os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image
import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos/AI/velocity_picking/src")
LAST_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if LAST_DIR:
    LAST_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = (LAST_DIR or OUT) / "_ckpt"; SCRATCH.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)

C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
SEIS = LinearSegmentedColormap.from_list("seis", ["#0A3069", "#54AEFF", "#FFFFFF", "#FF8182", "#82071E"])
PROB = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#FFF8C5", "#D4A72C", "#BC4C00", "#82071E"])
GREY = LinearSegmentedColormap.from_list("grey", ["#FFFFFF", "#D0D7DE", "#6E7781", "#1F2328"])


# ---- figure helpers (as in make_visuals_ml.py) ----------------------------------------------
def fig_base(title, foot):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=C["bg"])
    prog, what = title
    fig.text(0.015, 0.935, prog, color=C["ink"], fontsize=9, fontweight="bold")
    fig.text(0.015 + 0.0128 * len(prog) + 0.008, 0.935, "·  " + what, color=C["mute"], fontsize=9)
    fig.text(0.015, 0.012, foot, color=C["mute"], fontsize=6, va="bottom", linespacing=1.4)   # 1-2 lines
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
    for hx in C.values():                                      # theme colours kept exact
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


# =========================================================================================
# synthetic CMP gathers: primaries on a smooth V_rms(t), plus water-bottom and peg-leg multiples
# =========================================================================================
sys.path.insert(0, str(SRC))
from velocity_picking.core.semblance import compute_semblance, velocity_axis
from velocity_picking.core.preprocessing import semblance_to_rgb, velocity_to_bin_index
from velocity_picking.core.velocity_function import interval_to_rms
from velocity_picking.core.viterbi import viterbi_decode_vectorized
from velocity_picking.core.topk_decoder import topk_decode, compute_candidate_diversity
from velocity_picking.evaluation.metrics import velocity_mae, top1_mae, oracle_mae

NT, DT, NTR = 224, 0.008, 48
OFFS = np.linspace(100.0, 2500.0, NTR).astype(np.float32)
TWT = (np.arange(NT) * DT).astype(np.float32)
VEL = velocity_axis()                                          # 1200..2500 m/s, 5 m/s, 261 bins
NV = len(VEL)
VW = 1500.0                                                    # water velocity


def ricker_at(t, tev, f=25.0):
    a = (np.pi * f * (t - tev)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_gather(rng):
    t_wb = rng.uniform(0.12, 0.30)
    v0, v1 = rng.uniform(1550, 1720), rng.uniform(2300, 3000)
    tt = np.clip((TWT - t_wb) / (TWT[-1] - t_wb), 0, 1)
    vint = v0 + (v1 - v0) * tt ** rng.uniform(0.7, 1.4)
    for _ in range(rng.integers(1, 4)):                        # a few layers that step the interval velocity
        tc = rng.uniform(t_wb + 0.1, 1.7)
        vint += rng.uniform(-120, 160) / (1 + np.exp(-(TWT - tc) / 0.02))
    vint = np.where(TWT < t_wb, VW, vint).astype(np.float32)
    vint[0] = VW
    vrms = interval_to_rms(vint, np.maximum(TWT, 1e-4))
    vrms = np.clip(vrms, VEL[0], VEL[-1] - 30).astype(np.float32)

    events = [(t_wb, 1.0, VW)]                                 # seabed
    for t0 in np.sort(rng.uniform(t_wb + 0.04, TWT[-1] - 0.03, rng.integers(8, 15))):
        events.append((t0, rng.choice([-1, 1]) * rng.uniform(0.3, 0.9), float(np.interp(t0, TWT, vrms))))
    mult = rng.uniform(0.4, 1.0)                               # multiple strength varies by gather
    prim = list(events)
    for k, a in ((2, -0.75), (3, 0.5), (4, -0.3)):            # water-bottom multiples, V_rms = water velocity
        events.append((k * t_wb, mult * a, VW))
    for t0, a, v in prim[1:]:                                  # first-order peg-legs: one extra water bounce
        tp = t0 + t_wb
        vp = np.sqrt((v ** 2 * t0 + VW ** 2 * t_wb) / tp)
        events.append((tp, -0.55 * mult * a, vp))

    g = np.zeros((NTR, NT), np.float32)
    for t0, a, v in events:
        if t0 >= TWT[-1] + 0.05:
            continue
        tx = np.sqrt(t0 ** 2 + (OFFS / v) ** 2)
        keep = tx / t0 < 1.7                                   # stretch mute
        for i in np.where(keep)[0]:
            g[i] += a * ricker_at(TWT, tx[i])
    g += rng.normal(0, rng.uniform(0.04, 0.12), g.shape).astype(np.float32)
    return g, vrms, t_wb


def sample(rng):
    g, vrms, t_wb = make_gather(rng)
    sem = compute_semblance(g, OFFS, DT, VEL, half_window=3)
    return dict(g=g, sem=sem, x=semblance_to_rgb(sem).astype(np.float32),
                y=velocity_to_bin_index(vrms, v_min=float(VEL[0]), v_step=5.0, n_bins=NV), vrms=vrms, t_wb=t_wb)


def viterbi_pick(prob):
    path, _ = viterbi_decode_vectorized(np.log(np.clip(prob, 1e-10, None)))
    return VEL[path]


def mae_below_seabed(v, s):
    m = TWT >= s["t_wb"]
    return velocity_mae(v[m], s["vrms"][m])


# =========================================================================================
# training (cached, so the figure can be re-rendered without re-training)
# =========================================================================================
STEPS, BATCH, EVERY, LR, N_TRAIN, N_HOLD = 420, 4, 14, 2e-4, 480, 16
cache = SCRATCH / "vp_snaps.npz"
t0 = time.time()
hold = [sample(np.random.default_rng(90000 + i)) for i in range(N_HOLD)]
SHOW = 0
base_mae = np.array([mae_below_seabed(viterbi_pick(s["sem"] / (s["sem"].sum(1, keepdims=True) + 1e-9)), s)
                     for s in hold])
if cache.exists():
    ck = np.load(cache)
    steps_, maes, losses, probs, train_s = ck["steps"], ck["maes"], ck["losses"], ck["probs"], float(ck["train_s"])
    last_prob = ck["last_prob"]
else:
    from velocity_picking.models.velocity_model import VelocityPickerModel
    from velocity_picking.training.losses import VelocityPickingLoss
    torch.manual_seed(42)
    pool = [sample(np.random.default_rng(i)) for i in range(N_TRAIN)]
    print(f"data {time.time() - t0:.0f}s")
    model = VelocityPickerModel(pretrained=True, n_velocity_bins=NV, input_height=NT, input_width=NV)
    crit = VelocityPickingLoss(n_bins=NV)                          # sigma 2 bins, label smoothing 0.1
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.08)
    xh = torch.from_numpy(np.stack([s["x"] for s in hold]))

    def evaluate():
        model.eval()
        with torch.no_grad():
            p = torch.cat([model.predict_proba(xh[i:i + 8]) for i in range(0, N_HOLD, 8)]).numpy()
        model.train()
        return p, np.array([mae_below_seabed(viterbi_pick(p[i]), hold[i]) for i in range(N_HOLD)])

    rng = np.random.default_rng(1)
    steps_, maes, losses, probs = [], [], [], []
    p, m = evaluate(); steps_.append(0); maes.append(m); losses.append(np.nan); probs.append(p[SHOW])
    t1 = time.time(); run = []
    model.train()
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, N_TRAIN, BATCH)
        xb = torch.from_numpy(np.stack([pool[i]["x"] for i in idx]))
        yb = torch.from_numpy(np.stack([pool[i]["y"] for i in idx]).astype(np.int64))
        opt.zero_grad()
        loss = crit(model(xb), yb)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)       # as train_one_epoch
        opt.step(); sched.step()
        run.append(float(loss))
        if step % EVERY == 0:
            p, m = evaluate()
            steps_.append(step); maes.append(m); losses.append(np.mean(run[-EVERY:])); probs.append(p[SHOW])
            print(f"step {step:3d}  loss {losses[-1]:.3f}  held-out MAE {m.mean():6.1f} m/s"
                  f"  ({time.time() - t1:.0f}s)", flush=True)
    train_s = time.time() - t1
    last_prob = p
    steps_, maes, losses, probs = np.array(steps_), np.array(maes), np.array(losses), np.array(probs, np.float32)
    np.savez_compressed(cache, steps=steps_, maes=maes, losses=losses, probs=probs, train_s=train_s,
                        last_prob=last_prob)

mm = maes.mean(1)
print(f"trained {train_s:.0f}s  held-out MAE {mm[0]:.1f} -> {mm[-1]:.1f} m/s, baseline {base_mae.mean():.1f}")

# top-k on the shown gather with the last weights
s = hold[SHOW]
cands = topk_decode(last_prob[SHOW], semblance=s["sem"], k=5, velocity_bins=VEL)
mask = TWT >= s["t_wb"]
t1m = velocity_mae(cands[0].velocity[mask], s["vrms"][mask])
orm = min(velocity_mae(c.velocity[mask], s["vrms"][mask]) for c in cands)
print(f"shown gather: top-1 MAE {t1m:.1f}, oracle {orm:.1f} (whole trace: {top1_mae(cands, s['vrms']):.1f}"
      f" / {oracle_mae(cands, s['vrms']):.1f})")
base_show = viterbi_pick(s["sem"] / (s["sem"].sum(1, keepdims=True) + 1e-9))

# =========================================================================================
# figure
# =========================================================================================
print(f"candidate diversity {compute_candidate_diversity(cands):.1f} m/s")
fig, status = fig_base(("VelocityPicking", "Swin-T learns P(v|t)"),
                       "VelocityPickerModel (Swin-T + 6-layer Transformer) · VelocityPickingLoss · compute_semblance"
                       " · Viterbi · topk_decode\n"
                       f"synthetic CMP gathers with multiples · trained {took(train_s)} here on CPU ({STEPS} steps)"
                       " - not production weights")
ext = [VEL[0], VEL[-1], TWT[-1], 0]
seis = ListedColormap(SEIS(np.linspace(0, 1, 21)))
grey = ListedColormap(GREY(np.linspace(0, 1, 12)))
prob_cm = ListedColormap(PROB(np.linspace(0, 1, 14)))

a0 = fig.add_axes([0.058, 0.17, 0.104, 0.69]); style(a0); a0.set_xticks([])
a0.imshow(s["g"].T, cmap=seis, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
          extent=[OFFS[0], OFFS[-1], TWT[-1], 0])
a0.set_ylabel("TWT (s)", fontsize=6, labelpad=1)
label(a0, "CMP gather")

a1 = fig.add_axes([0.175, 0.17, 0.19, 0.69]); style(a1); a1.set_yticklabels([])
a1.imshow(s["sem"], cmap=grey, vmin=0, vmax=np.percentile(s["sem"], 99.5), aspect="auto",
          interpolation="nearest", extent=ext)
a1.plot(s["vrms"], TWT, color=C["ok"], lw=1.0, ls=(0, (3, 1.5)))
a1.plot(base_show, TWT, color=C["bad"], lw=0.9)
a1.set_xticks([1500, 2000, 2500]); a1.set_xlim(VEL[0], VEL[-1])
label(a1, "semblance")

a2 = fig.add_axes([0.38, 0.17, 0.19, 0.69]); style(a2); a2.set_yticklabels([])


def show_prob(p):
    return p / (p.max(1, keepdims=True) + 1e-12)               # display only: each time row scaled to its max


im2 = a2.imshow(show_prob(probs[0]), cmap=prob_cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                extent=ext)
a2.plot(s["vrms"], TWT, color=C["ok"], lw=1.0, ls=(0, (3, 1.5)))
pk, = a2.plot(viterbi_pick(probs[0]), TWT, color=C["acc"], lw=1.2)
a2.set_xticks([1500, 2000, 2500]); a2.set_xlim(VEL[0], VEL[-1])
label(a2, "model P(v|t) + Viterbi")
for ax in (a1, a2):
    ax.set_xlabel("V_rms (m/s)", fontsize=6, labelpad=1)

a3 = fig.add_axes([0.64, 0.225, 0.345, 0.635]); style(a3, grid=True)
label(a3, f"held-out pick error, {N_HOLD} gathers (m/s)")
a3.set_xlim(0, STEPS); a3.set_yscale("log"); a3.set_ylim(12, 520)
a3.set_yticks([20, 50, 100, 200, 400]); a3.set_yticklabels(["20", "50", "100", "200", "400"])
a3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a3.axhline(base_mae.mean(), color=C["bad"], lw=0.8, ls=(0, (3, 2)))
a3.text(STEPS * 0.98, base_mae.mean() * 1.06, f"Viterbi on semblance {base_mae.mean():.0f}", ha="right",
        va="bottom", color=C["bad"], fontsize=6)
a3.set_xlabel("training step (batch of 4)", fontsize=6, labelpad=1)
ln, = a3.plot([], [], color=C["acc"], lw=1.3)
dot, = a3.plot([], [], "o", color=C["acc"], ms=3)
mt = a3.text(0.98, 0.92, "", transform=a3.transAxes, ha="right", color=C["acc"], fontsize=6)
a3.text(0.98, 0.83, "-- true V_rms (panels)", transform=a3.transAxes, ha="right", color=C["ok"], fontsize=6)

frames = []
status.set_text("step   0  untrained decoder")
ln.set_data(steps_[:1], mm[:1]); dot.set_data([steps_[0]], [mm[0]])
frames += [grab(fig)] * 8
sub = 2
for k in range(1, len(steps_)):
    for j in range(sub):
        f = (j + 1) / sub
        p = probs[k - 1] * (1 - f) + probs[k] * f
        im2.set_data(show_prob(p))
        if j == sub - 1:
            pk.set_data(viterbi_pick(probs[k]), TWT)
            ln.set_data(steps_[:k + 1], mm[:k + 1]); dot.set_data([steps_[k]], [mm[k]])
            mt.set_text(f"model + Viterbi {mm[k]:.0f}")
        status.set_text(f"step {steps_[k]:3d}/{STEPS}   MAE {mm[k]:5.1f} m/s")
        frames.append(grab(fig))
frames += [frames[-1]] * 10

# top-k decoder: 5 diverse candidates from the last P(v|t)
pk.set_alpha(0.0)
im2.set_alpha(0.35)                                            # fade the heat so the 5 candidates show
arts = []
for r, c in enumerate(sorted(cands, key=lambda c: c.rank, reverse=True)):
    arts.append(a2.plot(c.velocity, TWT, color=C["ink"] if c.rank == 1 else C["mute"],
                        lw=1.3 if c.rank == 1 else 0.6, alpha=1.0 if c.rank == 1 else 0.8)[0])
    status.set_text(f"topk_decode   candidate {r + 1}/5")
    frames += [grab(fig)] * 4
status.set_text(f"top-1 {t1m:.0f} m/s · best of 5 {orm:.0f}")
frames += [grab(fig)] * 6
write_gif(frames, "velocity-picking.gif", hold=48)
print(f"total {time.time() - t0:.0f}s")
