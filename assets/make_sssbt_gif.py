"""sss-bottomtrack.gif - SSS-BT: the 1-D CNN learning to pick the side-scan bottom track, epoch by epoch.

Uses the real sss_bt_picker code (/home/user/repos/AI/sss_bt_picker/src):
  training/cnn1d.py::BT1DCNN          the model (default base_channels=32), unchanged
  training/trainer.py::Trainer        the real training loop (AdamW, cosine LR, L1 loss, early stop)
  training/dataset_loader.py::HDF5BTDataset   reads the synthetic HDF5 written here
  training/evaluator.py::evaluate_checkpoint  final MAE / RMSE / >1 m rate on the held-out line
  dataset/features.py::normalize_log  CNN input normalisation
  picker/gradient_picker.py::pick_bt_gradient + smooth_picks   the classic picker for contrast

Pings are synthetic (built here, no sonar files are read). Weights are trained here on CPU -
they are not the production weights. Per-epoch snapshots come from a subclass of BT1DCNN that only
copies its state_dict during training-mode forward calls (same architecture, same forward).

usage:  python make_sssbt_gif.py [out_dir]
"""
import copy, math, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image
import torch

REPO = Path("/home/user/repos/AI/sss_bt_picker/src")
sys.path.insert(0, str(REPO))
import sss_bt.training.cnn1d as cnnmod
from sss_bt.training.trainer import Trainer, TrainerConfig
from sss_bt.training.evaluator import evaluate_checkpoint
from sss_bt.dataset.features import normalize_log
from sss_bt.picker.gradient_picker import pick_bt_gradient, smooth_picks, PickResult

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent); OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = OUT / "_ckpt"; SCRATCH.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)

# ---- style (copied from make_visuals_ml.py) ------------------------------------------------
C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})


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
    return ax.text(0.0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom",
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
    frames[-1].save(OUT / (Path(name).stem + "-last.png"))
    frames[len(frames) // 2].save(OUT / (Path(name).stem + "-mid.png"))
    print(f"{name}: {len(pal)} frames, {(OUT / name).stat().st_size / 1024:.0f} KB")
    plt.close("all")


def took(sec):
    return f"{sec / 60:.0f} min" if sec >= 90 else f"{sec:.0f} s"


# ---- synthetic side-scan pings ---------------------------------------------------------------
L, RMAX = 1024, 60.0                  # samples per side, slant range per side (m)
DR = RMAX / L
R_ = (np.arange(L) + 0.5) * DR
SCALE = 65535 / 1.6                   # amplitude 1.6 -> uint16 full scale


def smooth1d(rng, n, k):
    z = rng.normal(0, 1, n + 4 * k)
    w = np.exp(-0.5 * (np.arange(-2 * k, 2 * k + 1) / k) ** 2)
    z = np.convolve(z, w / w.sum(), mode="same")[2 * k:2 * k + n]
    return (z - z.mean()) / (z.std() + 1e-9)


def smooth2d(rng, shape, k):
    z = rng.normal(0, 1, shape)
    f = np.fft.rfft2(z); ky = np.fft.fftfreq(shape[0])[:, None]; kx = np.fft.rfftfreq(shape[1])[None]
    f *= np.exp(-(kx ** 2 + ky ** 2) * k)
    z = np.fft.irfft2(f, s=shape); return (z - z.mean()) / z.std()


def render_line(rng, alt, fish, surf_amp, noise, edge_w, intf):
    """alt (npg, 2) first-return slant range per side; others (npg,) per ping. Returns raw uint16 (npg, 2, L)."""
    npg = len(fish)
    raw = np.zeros((npg, 2, L), np.uint16)
    for s in range(2):
        patch = smooth2d(rng, (npg, L), 3000)
        for i in range(npg):
            a = alt[i, s]
            grain = rng.gamma(4.0, 0.25, L)
            amp = noise[i] * np.abs(rng.normal(0, 1, L))                              # receiver noise floor
            # surface return at slant range = fish depth (sea-state dependent), with a short tail
            f = fish[i]
            sr = np.where(R_ < f, np.exp(-((R_ - f) / 0.35) ** 2), np.exp(-(R_ - f) / 1.2))
            amp += surf_amp[i] * sr * rng.gamma(3.0, 1 / 3.0, L)
            # seabed: soft or sharp rise at the first return, grazing-angle and range decay, texture
            rise = 1 / (1 + np.exp(-(R_ - a) / max(edge_w[i], 1e-3)))
            graze = np.clip(a / np.maximum(R_, a), 0, 1)
            bs = (0.55 + 0.18 * patch[i]) * grain * (0.25 + 0.75 * graze ** 0.7) * np.exp(-R_ / 140)
            bs += 0.9 * np.exp(-((R_ - a) / 0.5) ** 2) * grain                        # near-nadir return
            amp += np.clip(bs, 0, None) * rise
            # first seabed-surface multiple, weak
            m0 = 2 * a + 0.5 * f
            if m0 < RMAX:
                amp += 0.18 * grain * np.exp(-np.clip(R_ - m0, 0, None) / 6) / (1 + np.exp(-(R_ - m0) / 0.3))
            # cross-talk / interference: a comb of narrow spikes on this ping
            if intf[i] > 0:
                ph = rng.integers(0, 60); idx = np.arange(ph, L, rng.integers(45, 80))
                amp[idx] += intf[i] * rng.uniform(0.5, 1.0, idx.size)
                amp[np.clip(idx + 1, 0, L - 1)] += 0.6 * intf[i]
            raw[i, s] = np.clip(amp * SCALE, 0, 65535).astype(np.uint16)
    return raw


def random_line(rng, npg):
    """Training line: same artifact families as the display line, random parameters."""
    p = np.arange(npg)
    depth = rng.uniform(10, 40) + rng.uniform(0, 5) * smooth1d(rng, npg, 12)
    fish = np.clip(rng.uniform(1, 12) + rng.uniform(0, 1.5) * smooth1d(rng, npg, 10), 0.8, None)
    if rng.random() < 0.5:                                                         # fish-depth change
        c = rng.uniform(0, npg); fish = fish + rng.uniform(-6, 6) / (1 + np.exp(-(p - c) / rng.uniform(2, 8)))
        fish = np.clip(fish, 0.8, None)
    alt1 = np.clip(depth - fish, 3.5, 34)
    slope = rng.uniform(-0.04, 0.04)
    alt = np.c_[alt1 * (1 + slope), alt1 * (1 - slope)]
    surf = np.clip(rng.uniform(0, 0.7) * (1 + 0.6 * smooth1d(rng, npg, 8)), 0, None)
    noise = np.clip(rng.choice([0.006, 0.015, 0.03, 0.06]) * (1 + 0.5 * smooth1d(rng, npg, 15)), 0.002, None)
    edge = np.full(npg, rng.choice([0.03, 0.1, 0.3, 0.6]))
    intf = np.where(rng.random(npg) < rng.choice([0, 0.05, 0.3]), rng.uniform(0.2, 1.0), 0.0)
    return alt, render_line(rng, alt, fish, surf, noise, edge, intf)


def display_line():
    """Held-out line (own seed, never in training): fish-depth step, sea-state patch, interference, noisy soft end."""
    rng = np.random.default_rng(20261106)
    npg = 200; p = np.arange(npg)
    depth = 27 + 3.0 * np.sin(p / 26) + 4.0 * np.exp(-((p - 150) / 22) ** 2) + 0.15 * smooth1d(rng, npg, 4)
    fish = 2.4 + 6.0 / (1 + np.exp(-(p - 88) / 4.0)) + 0.2 * smooth1d(rng, npg, 6)   # winch out: 2.4 m -> 8.4 m
    alt1 = depth - fish
    alt = np.c_[alt1 * 1.015, alt1 * 0.985]
    surf = 0.08 + 1.4 * np.exp(-((p - 125) / 18) ** 2)                               # rough sea patch
    noise = 0.008 + 0.05 * (1 / (1 + np.exp(-(p - 168) / 3.0)))                       # noise floor rises at the end
    edge = np.where(p > 165, 0.45, 0.06)                                                 # soft seabed at the end
    intf = np.where((p > 35) & (p < 62) & (rng.random(npg) < 0.6), rng.uniform(0.4, 1.0, npg), 0.0)
    return dict(alt=alt, fish=fish, raw=render_line(rng, alt, fish, surf, noise, edge, intf))


def write_h5(path, prof, alts):
    import h5py
    n = len(prof)
    with h5py.File(path, "w") as f:
        f.attrs["n_samples"] = n; f.attrs["sample_length"] = L
        f["samples"] = prof.astype(np.float32); f["altitudes"] = alts.astype(np.float32)
        f["ping_numbers"] = np.arange(n, dtype=np.int32)
        f["sides"] = np.array([b"port", b"stbd"] * (n // 2))
        f["source_files"] = np.array([b"synthetic"] * n)


# ---- training through the real Trainer, with per-half-epoch snapshots --------------------------
EPOCHS, N_LINES, NPG_TRAIN, BATCH, LR = 12, 48, 60, 64, 3e-4
Base = cnnmod.BT1DCNN


class SnapCNN(Base):
    """BT1DCNN unchanged; copies its weights at the training-mode forward calls listed in `when`."""
    snaps, calls, when, last = [], 0, set(), None

    def __init__(self, *a, **k):
        super().__init__(*a, **k); SnapCNN.last = self

    def forward(self, x):
        if self.training:
            if SnapCNN.calls in SnapCNN.when:
                SnapCNN.snaps.append((SnapCNN.calls, copy.deepcopy(self.state_dict())))
            SnapCNN.calls += 1
        return super().forward(x)


def train():
    cache = SCRATCH / "sssbt_run.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    rng = np.random.default_rng(7)
    prof, alts = [], []
    t0 = time.time()
    for _ in range(N_LINES):
        alt, raw = random_line(rng, NPG_TRAIN)
        prof.append(normalize_log(raw).reshape(-1, L)); alts.append(alt.reshape(-1))
    prof = np.concatenate(prof); alts = np.concatenate(alts)
    h5 = SCRATCH / "synthetic_train.h5"; write_h5(h5, prof, alts)
    print(f"train set {len(prof)} profiles built in {time.time() - t0:.0f}s")

    cfg = TrainerConfig(dataset_path=h5, output_dir=SCRATCH, model_name="cnn1d_synth", batch_size=BATCH,
                        epochs=EPOCHS, learning_rate=LR, device="cpu", num_workers=0, seed=42)
    n_train = len(prof) - int(len(prof) * cfg.val_frac)
    nb = math.ceil(n_train / BATCH)
    SnapCNN.when = set(range(0, nb, 4)) | set(range(nb, nb * EPOCHS + 1, max(1, nb // 2)))   # dense in epoch 1
    cnnmod.BT1DCNN = SnapCNN
    t0 = time.time()
    try:
        res = Trainer(cfg).train()
    finally:
        cnnmod.BT1DCNN = Base
    train_s = time.time() - t0
    snaps = [(c / nb, sd) for c, sd in SnapCNN.snaps]
    snaps.append((SnapCNN.calls / nb, copy.deepcopy(SnapCNN.last.state_dict())))
    run = dict(snaps=snaps, history=res["history"], best=res["best_val_mae"], ckpt=res["checkpoint"],
               train_s=train_s, n_profiles=len(prof), n_train=n_train)
    torch.save(run, cache)
    return run


def main():
    run = train()
    disp = display_line()
    raw, alt = disp["raw"], disp["alt"]
    npg = len(raw)
    x = torch.from_numpy(normalize_log(raw).reshape(-1, 1, L))
    truth = alt.reshape(-1)

    # held-out line as HDF5 so the repo's evaluator can score the best checkpoint
    h5h = SCRATCH / "synthetic_heldout.h5"; write_h5(h5h, x[:, 0].numpy(), truth)
    ev = evaluate_checkpoint(Path(run["ckpt"]), h5h, split="heldout", outlier_threshold_m=1.0, device="cpu")
    best = torch.load(run["ckpt"], weights_only=False)

    model = Base(); model.eval()

    def predict(sd):
        model.load_state_dict(sd)
        with torch.no_grad():
            return model(x).numpy().reshape(npg, 2)

    preds = [(ep, predict(sd)) for ep, sd in run["snaps"]]
    pred_best = predict(best["model_state"])
    mae_line = np.array([np.abs(pp - alt).mean() for _, pp in preds])
    eps = np.array([ep for ep, _ in preds])

    # classic gradient picker on the linear amplitudes, then the repo's temporal smoothing (as pick_jsf_pings)
    grad = np.zeros((npg, 2))
    for s, ch in enumerate(("port", "starboard")):
        picks = []
        for i in range(npg):
            bi, a, cf = pick_bt_gradient(raw[i, s].astype(np.float32), DR)
            picks.append(PickResult(ping_number=i, altitude_m=a, bt_sample_index=bi, confidence=cf, channel=ch))
        grad[:, s] = [pk.altitude_m for pk in smooth_picks(picks, window=11)]
    g_err = np.abs(grad - alt)
    g_mae = g_err.mean(); g_out = (g_err > 1.0).mean()
    g_bias = float(np.median((grad - alt)[g_err < 1.0]))     # constant offset on clean pings
    print(f"trained {run['train_s']:.0f}s on {run['n_profiles']} profiles; best val MAE {run['best']:.3f} m")
    print(f"held-out line: CNN best MAE {ev.mae:.3f} m, >1 m {100 * ev.outlier_rate:.1f}%  |  "
          f"gradient MAE {g_mae:.3f} m, >1 m {100 * g_out:.1f}%")
    for a, b in ((0, 35), (35, 62), (62, 88), (88, 105), (105, 150), (150, 168), (168, 200)):
        print(f"  pings {a:3d}-{b:3d}: gradient MAE {g_err[a:b].mean():6.2f}   CNN-best {np.abs(pred_best - alt)[a:b].mean():6.2f}")

    # ---- figure ----
    hist = run["history"]
    fig, status = fig_base(("SSS-BT", "1-D CNN learning the bottom track"),
                           f"synthetic pings · BT1DCNN + Trainer (L1, AdamW), trained {took(run['train_s'])} here on"
                           f" CPU, {run['n_profiles']:,} profiles · not production weights\n"
                           "pick_bt_gradient + smooth_picks · normalize_log · evaluate_checkpoint"
                           " · the line shown is held out from training")
    cmap = LinearSegmentedColormap.from_list("sss", ["#FFFFFF", "#F1E6CC", "#C9A35A", "#7A5418", "#2B1D07"])
    cmap.set_bad(C["bg"])
    a1 = fig.add_axes([0.06, 0.205, 0.43, 0.655]); style(a1)
    a2 = fig.add_axes([0.515, 0.205, 0.12, 0.655]); style(a2, grid=True)
    a3 = fig.add_axes([0.70, 0.205, 0.285, 0.655]); style(a3, grid=True)

    ds = 4
    disp_amp = raw.astype(np.float32).reshape(npg, 2, L // ds, ds).mean(-1) / SCALE
    full = np.hstack([disp_amp[:, 0, ::-1], disp_amp[:, 1]])
    vmax = np.percentile(full, 99.5)
    img = np.full_like(full, np.nan)
    im = a1.imshow(img, origin="lower", aspect="auto", cmap=cmap, norm=PowerNorm(0.6, 0, vmax),
                   extent=[-RMAX, RMAX, 0, npg], interpolation="nearest")
    a1.set_xticks([-60, -30, 0, 30, 60]); a1.set_xticklabels(["60", "30", "0", "30", "60"])
    a1.set_xlim(-RMAX, RMAX)
    a1.set_xlabel("port     slant range (m)     stbd", fontsize=6, labelpad=1)
    a1.set_ylabel("ping", fontsize=6, labelpad=1)
    label(a1, "waterfall, held out")
    y = np.arange(npg) + 0.5
    gp = [a1.scatter([], [], s=1.6, color=C["bad"], lw=0, zorder=5) for _ in range(2)]
    cl = [a1.plot([], [], color=C["acc"], lw=1.0, zorder=6)[0] for _ in range(2)]
    a1.text(0.45, 1.015, "gradient", transform=a1.transAxes, color=C["bad"], fontsize=6.5, va="bottom")
    cnn_lab = a1.text(1.0, 1.015, "", transform=a1.transAxes, color=C["acc"], fontsize=6.5, va="bottom",
                      ha="right")

    EMAX = 6.0
    a2.set_ylim(0, npg); a2.set_yticklabels([]); a2.set_xlim(0, EMAX)
    a2.set_xlabel("|error| (m)", fontsize=6, labelpad=1)
    label(a2, "per ping")
    a2.axvline(1.0, color=C["mute"], lw=0.6, ls=(0, (3, 2)))
    a2.text(1.12, npg - 3, "1 m", color=C["mute"], fontsize=5.5, va="top")
    e_g, = a2.plot([], [], color=C["bad"], lw=0.8)
    e_c, = a2.plot([], [], color=C["acc"], lw=0.8)

    a3.set_yscale("log"); EP = len(hist); a3.set_xlim(0, EP); a3.set_ylim(0.05, 30)
    a3.set_xlabel("epoch", fontsize=6, labelpad=1)
    label(a3, "MAE on the held-out line (m)")
    a3.axhline(g_mae, color=C["bad"], lw=0.8, ls=(0, (3, 2)))
    a3.text(EP * 0.98, 22, f"- - gradient picker {g_mae:.2f}", ha="right", va="top",
            color=C["bad"], fontsize=6)
    a3.text(EP * 0.98, 12, f"({abs(g_bias):.1f} m early on clean pings)", ha="right",
            va="top", color=C["bad"], fontsize=5.5)
    tr, = a3.plot([], [], color=C["mute"], lw=0.8)
    ln, = a3.plot([], [], color=C["acc"], lw=1.3)
    dot, = a3.plot([], [], "o", color=C["acc"], ms=3)
    trl = a3.text(0.02, 0.03, "", transform=a3.transAxes, color=C["mute"], fontsize=6, va="bottom")

    def clipe(e):
        return np.minimum(e, EMAX - 0.05)

    frames = []
    # phase 1: waterfall arrives ping by ping, the classic picker runs on it
    for k in list(range(10, npg + 1, 10)):
        img[:k] = full[:k]; im.set_data(img)
        for s, sg in enumerate((-1, 1)):
            gp[s].set_offsets(np.c_[sg * grad[:k, s], y[:k]])
        e_g.set_data(clipe(g_err[:k].mean(1)), y[:k])
        status.set_text(f"ping {k:3d}/{npg}   gradient picker")
        status.set_color(C["mute"])
        frames.append(grab(fig))
    frames += [frames[-1]] * 4

    # phase 2: CNN snapshots (every 4 batches in epoch 1, then every half epoch)
    for k, (ep, pp) in enumerate(preds):
        for s, sg in enumerate((-1, 1)):
            cl[s].set_data(np.clip(sg * pp[:, s], -RMAX, RMAX), y)
        e_c.set_data(clipe(np.abs(pp - alt).mean(1)), y)
        ln.set_data(eps[:k + 1], mae_line[:k + 1]); dot.set_data([ep], [mae_line[k]])
        done = [h for h in hist if h["epoch"] <= ep + 1e-6]
        tr.set_data([0] + [h["epoch"] for h in done], [np.nan] + [h["val_mae"] for h in done])
        if done:
            trl.set_text(f"grey: val split (synthetic) {done[-1]['val_mae']:.2f}")
        cnn_lab.set_text(f"1-D CNN ep {ep:5.2f}")
        status.set_text(f"epoch {ep:5.2f}/{EP}   CNN MAE {mae_line[k]:5.2f} m")
        status.set_color(C["acc"])
        frames.append(grab(fig))

    # final: best checkpoint, scored by the repo's evaluator
    for s, sg in enumerate((-1, 1)):
        cl[s].set_data(np.clip(sg * pred_best[:, s], -RMAX, RMAX), y)
    e_c.set_data(clipe(np.abs(pred_best - alt).mean(1)), y)
    cnn_lab.set_text(f"1-D CNN best ep {best['epoch']}")
    trl.set_text(f"best ckpt: >1 m on {100 * ev.outlier_rate:.0f}% of pings\n"
                 f"gradient: >1 m on {100 * g_out:.0f}% of pings")
    status.set_text(f"MAE  CNN {ev.mae:.2f} m  vs  gradient {g_mae:.2f} m")
    status.set_color(C["ok"] if ev.mae < g_mae else C["bad"])
    frames.append(grab(fig))
    write_gif(frames, "sss-bottomtrack.gif", hold=48)


if __name__ == "__main__":
    main()
