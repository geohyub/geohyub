"""Three machine-learning profile animations. Each trains a real GeoView model on synthetic data, on CPU.

  denoise-dncnn.gif     SeismicDenoising  models/dncnn.py::DnCNN, core/noise_generator.py::generate_mixed_noise
                                          training/losses.py::DenoisingLoss, evaluation/metrics.py::psnr
  maguxo-cnn.gif        MagUXO            core/cnn.py::CnnClassifier, core/ensemble.py::EnsembleClassifier
                                          (held-out class probabilities on a simplex, epoch by epoch)
                                          core/synthetic.py::generate_profiles
  magembed-knn.gif      MagEmbed          magembed/extractor.py::MagEmbedder, core.py::build_index, MagIndex.query
                                          (embeds with the MagUXO CnnClassifier trained for maguxo-cnn.gif)

Weights are trained here, for a few minutes, on synthetic data - they are not the production weights.
Style (palette, write_gif, title/footer) is copied from make_visuals.py.

usage:  python make_visuals_ml.py <out_dir> [repos_root] [png_dir] [denoise,uxo,embed]
needs:  numpy, scipy, matplotlib, pillow, scikit-learn, torch (CPU is fine)
"""
import copy, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image
import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "."); OUT.mkdir(parents=True, exist_ok=True)
R = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/repos")
LAST_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None   # optional: where to drop mid/last-frame PNGs
if LAST_DIR:
    LAST_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = (LAST_DIR or OUT) / "_ckpt"; SCRATCH.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

C = dict(bg="#FFFFFF", ink="#1F2328", mute="#6E7781", hair="#E6E9ED", acc="#0969DA",
         bad="#CF222E", warn="#9A6700", ok="#1A7F37")
W, H, DPI, FPS = 6.4, 3.0, 100, 12
plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 7,
                     "axes.edgecolor": C["hair"], "axes.labelcolor": C["mute"],
                     "xtick.color": C["mute"], "ytick.color": C["mute"]})
SEIS = LinearSegmentedColormap.from_list("seis", ["#0A3069", "#54AEFF", "#FFFFFF", "#FF8182", "#82071E"])
PROB = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#FFF8C5", "#D4A72C", "#BC4C00", "#82071E"])
CLS = {0: C["bad"], 1: C["acc"], 2: "#8C959F"}            # uxo, geologic, noise


# ---- figure helpers (as in make_visuals.py) -------------------------------------------------
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


def ricker(f, n=41):
    t = np.arange(n) - n // 2
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


# =========================================================================================
# 1. SeismicDenoising - DnCNN learns to strip mixed marine noise from a section
# =========================================================================================
def make_section(rng, nt=128, nx=128):
    """Layered reflectivity with dip, a fold and one normal fault, convolved with a Ricker."""
    x = np.arange(nx)
    refl = np.zeros((nt, nx))
    fold_a, fold_c = rng.uniform(-8, 8), rng.uniform(0.2, 0.8) * nx
    fold_w = rng.uniform(0.15, 0.4) * nx
    dip = rng.uniform(-0.12, 0.12)
    fx, throw = rng.uniform(0.25, 0.75) * nx, rng.uniform(4, 12)
    fdip = rng.uniform(0.3, 0.8)
    for t0 in np.sort(rng.uniform(4, nt + 20, rng.integers(10, 18))):
        amp = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
        z = t0 + dip * x + fold_a * np.exp(-((x - fold_c) / fold_w) ** 2)
        zf = z + throw * (x > fx + (z - nt / 2) * fdip)
        for j in range(nx):
            k = int(round(zf[j]))
            if 0 <= k < nt:
                refl[k, j] += amp
    w = ricker(rng.uniform(0.06, 0.11))
    sec = np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0, refl)
    return (sec / (np.abs(sec).max() + 1e-9)).astype(np.float32)


def denoise():
    sys.path.insert(0, str(R / "seismicdenoising/src"))
    from seismic_denoising.models.dncnn import DnCNN
    from seismic_denoising.core.noise_generator import generate_mixed_noise
    from seismic_denoising.training.losses import DenoisingLoss
    from seismic_denoising.evaluation.metrics import psnr

    NL, NF, PATCH, BATCH, STEPS, EVERY = 12, 48, 64, 16, 420, 10
    model = DnCNN(n_layers=NL, features=NF)
    cache = SCRATCH / "dncnn_snaps.pt"                 # re-render without re-training
    if cache.exists():
        ck = torch.load(cache); wsnaps, train_s = ck["snaps"], ck["train_s"]
    else:
        torch.manual_seed(0)
        rng = np.random.default_rng(3); nrng = np.random.RandomState(3)
        pool = [make_section(rng) for _ in range(48)]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        crit = DenoisingLoss()

        def batch():
            cl, no = [], []
            for _ in range(BATCH):
                s = pool[rng.integers(len(pool))]
                i, j = rng.integers(0, s.shape[0] - PATCH + 1), rng.integers(0, s.shape[1] - PATCH + 1)
                c = s[i:i + PATCH, j:j + PATCH]
                if rng.random() < 0.5:
                    c = c[:, ::-1]
                c = np.ascontiguousarray(c)
                cl.append(c); no.append(c + generate_mixed_noise(c, snr_range=(2.0, 15.0), rng=nrng))
            return torch.from_numpy(np.stack(no))[:, None], torch.from_numpy(np.stack(cl))[:, None]

        wsnaps = [(0, copy.deepcopy(model.state_dict()))]
        t0 = time.time()
        for step in range(1, STEPS + 1):
            noisy, clean = batch()
            opt.zero_grad()
            loss, _ = crit(noisy - model(noisy), clean)        # residual learning, as training/train.py
            loss.backward(); opt.step(); sched.step()
            if step % EVERY == 0:
                wsnaps.append((step, copy.deepcopy(model.state_dict())))
        train_s = time.time() - t0
        torch.save({"snaps": wsnaps, "train_s": train_s}, cache)

    # held-out section (never in the training pool), all three noise types at 4 dB SNR
    test_clean = make_section(np.random.default_rng(2026), nt=128, nx=192)
    test_noisy = test_clean + generate_mixed_noise(test_clean, noise_types=["gaussian", "coherent", "swell"],
                                                   snr_range=(4.0, 4.0), rng=np.random.RandomState(5))
    xt = torch.from_numpy(test_noisy.astype(np.float32))[None, None]
    snaps = []
    model.eval()
    for st_, sd in wsnaps:
        model.load_state_dict(sd)
        with torch.no_grad():
            out = (xt - model(xt))[0, 0].numpy()
        snaps.append((st_, out, psnr(test_clean, out)))
    p_in = psnr(test_clean, test_noisy)
    print(f"denoise: trained {train_s:.0f}s, PSNR in {p_in:.1f} -> out {snaps[-1][2]:.1f} dB")

    fig, status = fig_base(("SeismicDenoising", "DnCNN learns the noise"),
                           f"DnCNN({NL} layers, {NF} ch) · DenoisingLoss · psnr · generate_mixed_noise"
                           f" (gaussian + linear coherent + swell, 4 dB)\ntrained {took(train_s)} here on"
                           f" synthetic data - not production weights   ·   seismic_denoising/{{models,training,evaluation,core}}")
    lim = 1.0
    seis = ListedColormap(SEIS(np.linspace(0, 1, 21)))      # banded: crisper on a GIF palette, smaller file
    axes = [fig.add_axes([0.035 + k * 0.205, 0.17, 0.19, 0.69]) for k in range(3)]
    titles = ["noisy input (held out)", "DnCNN output", "removed = input - output"]
    ims = []
    for ax, ttl, img in zip(axes, titles, [test_noisy, snaps[0][1], test_noisy - snaps[0][1]]):
        style(ax); ax.set_xticks([]); ax.set_yticks([])
        ims.append(ax.imshow(img, cmap=seis, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest"))
        label(ax, ttl)
    a4 = fig.add_axes([0.685, 0.225, 0.30, 0.635]); style(a4, grid=True)
    label(a4, "PSNR on this held-out section (dB)")
    st = np.array([c[0] for c in snaps]); pv = np.array([c[2] for c in snaps])
    a4.set_xlim(0, STEPS); a4.set_ylim(p_in - 4, pv.max() + 2)
    a4.axhline(p_in, color=C["mute"], lw=0.8, ls=(0, (3, 2)))
    a4.text(STEPS * 0.98, p_in + 0.3, f"input {p_in:.1f}", ha="right", va="bottom", color=C["mute"], fontsize=6)
    a4.set_xlabel("training step (batch of 16, 64x64 patches)", fontsize=6, labelpad=1)
    ln, = a4.plot([], [], color=C["acc"], lw=1.3)
    dot, = a4.plot([], [], "o", color=C["acc"], ms=3)
    frames = [grab(fig)] * 6
    for k, (s_, out, _) in enumerate(snaps):
        ims[1].set_data(out); ims[2].set_data(test_noisy - out)
        ln.set_data(st[:k + 1], pv[:k + 1]); dot.set_data([st[k]], [pv[k]])
        status.set_text(f"step {s_:3d}/{STEPS}   PSNR {pv[k]:4.1f} dB")
        frames.append(grab(fig))
    write_gif(frames, "denoise-dncnn.gif", hold=48)
    return dict(train_s=train_s, psnr_in=p_in, psnr_out=float(pv[-1]))


# =========================================================================================
# 2. MagUXO - CNN target-probability map over a synthetic survey sharpens as it trains
# =========================================================================================
_CNN = {}


def maguxo_setup():
    """Train MagUXO's CnnClassifier (with per-epoch snapshots) and EnsembleClassifier once."""
    if _CNN:
        return _CNN
    sys.path.insert(0, str(R / "maguxo"))
    from core import synthetic as syn
    from core.cnn import CnnClassifier, MagCnn
    from core.ensemble import EnsembleClassifier

    EPOCHS = 40                                                       # CnnClassifier default
    prof, lab = syn.generate_profiles(n_per_class=400, seed=20260520, remanence=True)
    vprof, vlab = syn.generate_profiles(n_per_class=100, seed=7, remanence=True)
    clf = CnnClassifier(epochs=EPOCHS, device="cpu", random_state=42)
    torch.manual_seed(42)
    snaps = [copy.deepcopy(MagCnn(dropout=clf.dropout).state_dict())]   # = fit()'s seeded init
    t0 = time.time()
    clf.fit(prof, lab, progress_cb=lambda e, n, l: snaps.append(copy.deepcopy(clf.model.state_dict())))
    cnn_s = time.time() - t0
    t0 = time.time()
    ens = EnsembleClassifier(cnn_epochs=EPOCHS, cnn_device="cpu", random_state=42).fit(prof, lab)
    ens_s = time.time() - t0
    same = np.allclose(ens.cnn.predict_proba(vprof[:50]), clf.predict_proba(vprof[:50]), atol=1e-5)
    print(f"maguxo: cnn {cnn_s:.0f}s, ensemble {ens_s:.0f}s, ensemble-cnn == snapshot-cnn: {same}")
    _CNN.update(syn=syn, clf=clf, ens=ens, snaps=snaps, loss=list(clf.loss_history),
                vprof=vprof, vlab=vlab, cnn_s=cnn_s, ens_s=ens_s, CnnClassifier=CnnClassifier)
    return _CNN


def maguxo_cnn():
    d = maguxo_setup()
    clf, ens, snaps, syn = d["clf"], d["ens"], d["snaps"], d["syn"]
    vprof, vlab = d["vprof"], d["vlab"]
    work = d["CnnClassifier"](epochs=1, device="cpu"); work.model = copy.deepcopy(clf.model); work.fitted = True
    probs, acc = [], []
    for s in snaps:
        work.model.load_state_dict(s)
        pr = work.predict_proba(vprof); probs.append(pr)
        acc.append(float((pr.argmax(1) == vlab).mean()))
    comp = ens._component_proba(vprof)
    p_ens = ens.predict_proba(vprof)
    spread = np.stack([comp[k] for k in ("baseline", "cnn", "physics")]).std(0).max(1)
    ens_acc = float((p_ens.argmax(1) == vlab).mean())
    n_dis = int((spread > 0.25).sum())
    print(f"  held-out acc cnn {acc[-1]:.3f}  ensemble {ens_acc:.3f}  ({len(vlab)} profiles, {n_dis} with spread>0.25)")

    # simplex corners: uxo top, geologic bottom-left, noise bottom-right
    Cn = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    jit = np.random.default_rng(1).normal(0, 0.012, (len(vlab), 2))         # display only
    pos = lambda p: p @ Cn + jit

    fig, status = fig_base(("MagUXO", "1D-CNN converging, then the ensemble"),
                           "core/cnn.py::CnnClassifier · core/ensemble.py::EnsembleClassifier · core/synthetic.py::"
                           "generate_profiles\n"
                           f"trained {took(d['cnn_s'] + d['ens_s'])} here on synthetic data - not production weights")
    a1 = fig.add_axes([0.015, 0.15, 0.47, 0.70]); a1.set_axis_off()
    a1.set_xlim(-0.2, 1.2); a1.set_ylim(-0.13, 0.94); a1.set_aspect("equal", adjustable="box")
    a1.plot(*np.r_[Cn, Cn[:1]].T, color=C["hair"], lw=0.9)
    cen = Cn.mean(0)
    for i, j in ((0, 1), (1, 2), (2, 0)):                                   # argmax decision boundaries
        mid = (Cn[i] + Cn[j]) / 2; a1.plot([cen[0], mid[0]], [cen[1], mid[1]], color=C["hair"], lw=0.7, ls=(0, (2, 2)))
    for k, (nm, (px, py), ha) in enumerate(zip(syn.CLASS_NAMES, Cn, ["left", "left", "right"])):
        a1.text(px + (0.05, 0, 0)[k], py + (-0.02, -0.035, -0.035)[k], f"{nm} = 1", color=CLS[k],
                fontsize=6.5, ha=ha, va="top")
    t1 = a1.text(0.03, 1.0, "", transform=a1.transAxes, ha="left", va="bottom", color=C["mute"], fontsize=6.5)
    sc = a1.scatter(*pos(probs[0]).T, s=6, c=[CLS[int(v)] for v in probs[0].argmax(1)], lw=0, alpha=0.6)

    a2 = fig.add_axes([0.575, 0.60, 0.41, 0.25]); style(a2, grid=True)
    E = len(snaps) - 1
    a2.set_xlim(0, E); a2.set_ylim(0, 1.05); a2.set_xticklabels([])
    label(a2, "loss (scaled) / held-out accuracy, by epoch")
    loss = np.r_[np.nan, d["loss"]]; lmax = np.nanmax(loss)
    l1, = a2.plot([], [], color=C["mute"], lw=1.1)
    l2, = a2.plot([], [], color=C["acc"], lw=1.3)
    lt = a2.text(0.98, 0.5, "", transform=a2.transAxes, ha="right", color=C["ink"], fontsize=6)
    # three held-out profiles (one per class) with live class-probability bars
    ex = [int(np.where((vlab == c) & (p_ens.argmax(1) == c) & (probs[-1].argmax(1) == c))[0][0]) for c in (0, 1, 2)]
    a3 = fig.add_axes([0.575, 0.15, 0.41, 0.33]); style(a3); a3.set_xticks([]); a3.set_yticks([])
    a3.set_xlim(0, 3); a3.set_ylim(-0.88, 1.05)
    label(a3, "held-out profiles -> P(uxo / geologic / noise)")
    xs = np.linspace(0.06, 0.94, syn.PROFILE_SAMPLES)
    bars = []
    for k, i in enumerate(ex):
        p = vprof[i] - np.median(vprof[i]); p = 0.45 * p / np.abs(p).max()
        a3.plot(k + xs, p + 0.5, color=C["ink"], lw=0.7)
        row = []
        for c in range(3):
            row.append(a3.add_patch(plt.Rectangle((k + 0.06, -0.25 - 0.17 * c), 0, 0.12, color=CLS[c], lw=0)))
        bars.append(row)
    btx = [a3.text(k + 0.94, -0.77, "", ha="right", va="center", color=C["mute"], fontsize=5.5) for k in range(3)]

    def show(p, e_text, ex_p):
        sc.set_offsets(pos(p)); sc.set_color([CLS[int(v)] for v in p.argmax(1)])
        t1.set_text(e_text)
        for k, i in enumerate(ex):
            for c in range(3):
                bars[k][c].set_width(0.88 * ex_p[i, c])
            btx[k].set_text(f"{syn.CLASS_NAMES[int(ex_p[i].argmax())]} {ex_p[i].max():.2f}")

    frames, sub = [], 3
    ep = np.arange(E + 1)
    for e in range(E + 1):
        for k in range(sub if e else 1):
            f = (k + 1) / sub if e else 1.0
            p = probs[e - 1] * (1 - f) + probs[e] * f if e else probs[0]
            show(p, f"predict_proba, {len(vlab)} held-out profiles", p)
            l1.set_data(ep[:e + 1], loss[:e + 1] / lmax); l2.set_data(ep[:e + 1], acc[:e + 1])
            lt.set_text(f"loss {loss[e]:.3f}   acc {acc[e]:.2f}" if e else f"acc {acc[e]:.2f}")
            status.set_text(f"epoch {e:2d}/{E}")
            frames.append(grab(fig))
    frames += [frames[-1]] * 12
    for f in np.linspace(0, 1, 10) ** 0.7:
        p = probs[-1] * (1 - f) + p_ens * f
        show(p, "ensemble soft vote 0.45 / 0.40 / 0.15", p)
        status.set_text("ensemble")
        frames.append(grab(fig))
    a2.axhline(ens_acc, color=C["ok"], lw=0.9, ls=(0, (3, 2)))
    lt.set_text(f"cnn acc {acc[-1]:.2f}   ensemble acc {ens_acc:.2f}")
    status.set_text(f"ensemble   held-out acc {ens_acc:.2f}")
    frames += [grab(fig)] * 20
    m = spread > 0.25
    a1.scatter(*pos(p_ens)[m].T, s=22, facecolors="none", edgecolors=C["ink"], lw=0.6, zorder=5)
    t1.set_text(f"ringed: {m.sum()} where components disagree")
    frames.append(grab(fig))
    write_gif(frames, "maguxo-cnn.gif", hold=48)
    return dict(cnn_acc=acc[-1], ens_acc=ens_acc, n_disagree=int(m.sum()))


# =========================================================================================
# 3. MagEmbed - CNN embedding space organises itself; a query retrieves its neighbours
# =========================================================================================
def magembed():
    d = maguxo_setup()
    sys.path.insert(0, str(R / "magembed"))
    from magembed import MagEmbedder, build_index, retrieval_precision
    from sklearn.decomposition import PCA
    from scipy.linalg import orthogonal_procrustes

    syn = d["syn"]
    db, dbl = syn.generate_profiles(n_per_class=90, seed=99, remanence=True)
    qp, ql = syn.generate_profiles(n_per_class=30, seed=1234, remanence=True)
    work = d["CnnClassifier"](epochs=1, device="cpu"); work.model = copy.deepcopy(d["clf"].model); work.fitted = True
    paths = []
    for e, s in enumerate(d["snaps"]):                                 # checkpoint per epoch, CnnClassifier.save format
        work.model.load_state_dict(s); p = SCRATCH / f"cnn_e{e:02d}.pt"; work.save(p); paths.append(p)
    embs = [MagEmbedder(str(p)).embed(db) for p in paths]

    def unit(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    pcas = [PCA(2).fit(unit(e)) for e in embs]
    xy = [p.transform(unit(e)) for p, e in zip(pcas, embs)]
    ref = xy[-1] / np.abs(xy[-1]).max()
    al = []
    for a in xy:                                                       # display only: align each epoch's PCA to the last
        a = a / (np.abs(a).max() + 1e-12)
        Rm, _ = orthogonal_procrustes(a, ref); al.append(a @ Rm)

    final = MagEmbedder(str(paths[-1]))
    idx = build_index(db, labels=dbl, embedder=final)
    prec = retrieval_precision(np.r_[db, qp], np.r_[dbl, ql], k=5, embedder=final)
    qe = final.embed(qp)
    hits_all = [idx.query(e, k=5) for e in qe]
    agree_all = np.array([sum(m["label"] == ql[i] for m, _ in h) for i, h in enumerate(hits_all)])
    for c in (0, 1, 2):
        a = agree_all[ql == c]
        print(f"  held-out {syn.CLASS_NAMES[c]:8s} queries: 5/5 same-class neighbours {np.mean(a == 5):.2f},"
              f" majority {np.mean(a >= 3):.2f}")
    # shown queries: the first held-out profile of each class whose 5 neighbours all share its class
    # (fraction of such queries is printed above and reported)
    order = [int(np.where((ql == c) & (agree_all == 5))[0][0]) for c in (0, 1, 2)]
    queries = []
    for i in order:
        hits = hits_all[i]
        q2 = (pcas[-1].transform(unit(qe[i:i + 1])) / np.abs(xy[-1]).max())
        queries.append((i, [m["id"] for m, _ in hits], [x for _, x in hits], q2[0], int(agree_all[i])))
    print(f"  retrieval_precision@5 = {prec:.2f}")

    fig, status = fig_base(("MagEmbed", "similar-target search"),
                           "magembed: MagEmbedder.embed (MagUXO MagCnn.features, 64-d) · build_index · MagIndex.query"
                           " (cosine k-NN) · retrieval_precision\n"
                           f"MagUXO CnnClassifier trained {took(d['cnn_s'])} here on synthetic data"
                           " - not production weights")
    a1 = fig.add_axes([0.035, 0.17, 0.43, 0.68]); style(a1); a1.set_xticks([]); a1.set_yticks([])
    a1.set_xlim(-1.15, 1.15); a1.set_ylim(-1.15, 1.15)
    t1 = a1.text(0.0, 1.015, "", transform=a1.transAxes, ha="left", va="bottom", color=C["mute"], fontsize=6.5)
    cols = [CLS[int(v)] for v in dbl]
    sc = a1.scatter(al[0][:, 0], al[0][:, 1], s=7, c=cols, lw=0, alpha=0.75)
    for k, nm in enumerate(syn.CLASS_NAMES):
        a1.text(0.02, 0.05 + 0.065 * (2 - k), "● " + nm, transform=a1.transAxes, color=CLS[k], fontsize=6)
    # right: query profile + its 5 neighbours
    a2 = fig.add_axes([0.53, 0.17, 0.455, 0.68]); style(a2); a2.set_xticks([]); a2.set_yticks([])
    a2.set_xlim(0, 2); a2.set_ylim(-0.6, 5.6)
    label(a2, "query profile and its 5 nearest neighbours")
    xs = np.linspace(0, 1, syn.PROFILE_SAMPLES)

    def norm(p):
        p = p - np.median(p); return 0.42 * p / (np.abs(p).max() + 1e-9)
    frames = []
    E = len(al) - 1
    sub = 2
    for e in range(E + 1):
        for k in range(sub if e else 1):
            f = (k + 1) / sub if e else 1.0
            p = al[e - 1] * (1 - f) + al[e] * f if e else al[0]
            sc.set_offsets(p)
            t1.set_text(f"PCA of the 64-d embeddings, CNN epoch {e}")
            status.set_text(f"epoch {e:2d}/{E}   {len(db)} profiles")
            frames.append(grab(fig))
    frames += [frames[-1]] * 8
    t1.set_text(f"PCA of the 64-d embeddings · held-out precision@5 {prec:.2f}")
    arts = []
    for qi, (i, nn_ids, dists, q2, agree) in enumerate(queries):
        for a in arts:
            a.remove()
        arts = []
        sc.set_alpha(0.35)
        arts.append(a1.scatter([q2[0]], [q2[1]], s=90, marker="*", c=[CLS[int(ql[i])]], edgecolors=C["ink"],
                               lw=0.7, zorder=6))
        arts.append(a2.plot(xs, norm(qp[i]) + 5.0, color=CLS[int(ql[i])], lw=1.3)[0])
        arts.append(a2.text(1.05, 5.0, f"query  ({syn.CLASS_NAMES[ql[i]]}, held out)", color=C["ink"], fontsize=6,
                            va="center"))
        status.set_text(f"query {qi + 1}/3   embed -> k-NN")
        frames += [grab(fig)] * 6
        for r, (j, dist) in enumerate(zip(nn_ids, dists)):
            p = al[-1][j]
            arts.append(a1.plot([q2[0], p[0]], [q2[1], p[1]], color=C["ink"], lw=0.6, alpha=0.7, zorder=5)[0])
            arts.append(a1.scatter([p[0]], [p[1]], s=26, facecolors="none", edgecolors=C["ink"], lw=0.8, zorder=6))
            yb = 3.9 - r * 0.98
            arts.append(a2.plot(xs, norm(db[j]) + yb, color=CLS[int(dbl[j])], lw=0.9)[0])
            arts.append(a2.text(1.05, yb, f"#{j:<3d} {syn.CLASS_NAMES[dbl[j]]:<8s} d={dist:.3f}",
                                color=C["mute"], fontsize=6, va="center"))
            status.set_text(f"query {qi + 1}/3   neighbour {r + 1}/5")
            frames += [grab(fig)] * 3
        status.set_text(f"query {qi + 1}/3   {agree}/5 same class")
        frames += [grab(fig)] * 16
    write_gif(frames, "magembed-knn.gif", hold=36)
    return dict(prec=prec, agree=[q[4] for q in queries])


if __name__ == "__main__":
    which = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"denoise", "uxo", "embed"}
    if "denoise" in which: print(denoise())
    if "uxo" in which: print(maguxo_cnn())
    if "embed" in which: print(magembed())
