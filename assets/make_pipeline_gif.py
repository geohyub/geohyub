"""Terminal-style GIF of a real run: SeismicQC headless CLI -> .qcresult.json -> QCHub import.

Every number and verdict below is read from the files the real run produced
(demo/out/*.json and the QCHub sqlite DB); nothing is typed in by hand.
"""
import json, sqlite3, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
C = dict(bg="#FFFFFF", bar="#F6F8FA", edge="#D0D7DE", ink="#1F2328", mute="#6E7781",
         blue="#0969DA", green="#1A7F37", amber="#9A6700", red="#CF222E")
ST = {"meet": ("meet", "green"), "caution": ("caution", "amber"), "fail": ("fail", "red"),
      "unmeasured": ("n/a", "mute")}
VERD = {"pass": ("PASS", "green"), "conditional": ("CONDITIONAL", "amber"),
        "fail": ("FAIL", "red"), "unmeasured": ("NOT MEASURED", "mute")}
NAME = {"seafloor": "seafloor consistency", "navigation": "navigation coverage",
        "signal": "signal-to-noise", "spectrum": "bandwidth", "integrity": "dead traces"}

# ---- build the transcript from real outputs --------------------------------
lines = []  # each line: list of (text, colour, bold)
def say(*parts): lines.append(list(parts))
def cmd(text): lines.append([("$ ", "blue", True), (text, "ink", False), ("__TYPE__", None, None)])

for L in ("L0412", "L0413", "L0414"):
    r = json.load(open(D / "out" / f"{L}.json"))
    cmd(f"python -m seismicqc.analysis.cli run {L}.sgy --project DEMO-2026")
    for h in r.get("header_failures") or []:
        say(("  header ", "mute", False), (h["code"], "red", True),
            ("  file is truncated (not a whole number of traces)", "red", False))
    for d in r["dimensions"]:
        v = "—" if d["value"] is None else f"{d['value']:.1f} {d['unit']}"
        lab, col = ST.get(d["status"], (d["status"], "mute"))
        say((f"  {NAME.get(d['key'], d['key']):<22}", "ink", False), (f"{v:>11}  ", "mute", False), (lab, col, True))
    vl, vc = VERD[r["verdict"]]
    say(("  verdict ", "mute", False), (vl, vc, True), (f"   -> {L}.qcresult.json   {r['elapsed_sec']:.1f}s", "mute", False))
    say(("", "ink", False))

db = sqlite3.connect(D / "qchub-state" / "qchub.db")
rows = list(db.execute("select q.item_name, q.overall_status from qc_items q order by q.item_name"))
api = json.load(open(D / "qchub_responses.json")); pid = api["project_id"]
nf = sum(r["imported_files"] for r in api["imports"]); nc = sum(f["items_created"] for r in api["imports"] for f in r["files"])
proj = api["imports"][0]["files"][0]["declared_project"]
cmd(f"POST /api/project/{pid}/import-qc-result   (QCHub, {nf} files)")
say((f"  imported {nf} files  ·  {nc} line items created  ·  project {proj}", "mute", False))
for name, status in rows:
    col = {"Pass": "green", "Warning": "amber", "Fail": "red"}.get(status, "mute")
    say(("  ", "ink", False), ("■ ", col, True), (f"{name}   ", "ink", True), (status, col, True))
ag = api["reimport"]["files"][0]
cmd(f"POST /api/project/{pid}/import-qc-result   (L0413 again)")
say((f"  inserted {ag['inserted']}  ·  updated {ag['updated']}   re-import is idempotent", "green", False))

# ---- render frames ---------------------------------------------------------
W, LH, TOP, PAD = 760, 18, 40, 16
VIS = 22  # visible lines (scrolls)
H = TOP + VIS * LH + 14

def render(shown, partial=None):
    im = Image.new("RGB", (W, H), C["bg"]); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W - 1, H - 1], outline=C["edge"])
    dr.rectangle([1, 1, W - 2, 28], fill=C["bar"]); dr.line([1, 28, W - 2, 28], fill=C["edge"])
    for i, col in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        dr.ellipse([12 + i * 18, 9, 23 + i * 18, 20], fill=col)
    dr.text((W // 2, 15), "geoview · seismicqc -> qchub", font=F, fill=C["mute"], anchor="mm")
    body = shown + ([partial] if partial else [])
    body = body[-VIS:]
    y = TOP
    for ln in body:
        x = PAD
        for text, col, bold in ln:
            if text == "__TYPE__":
                continue
            dr.text((x, y), text, font=FB if bold else F, fill=C[col])
            x += dr.textlength(text, font=FB if bold else F)
        y += LH
    return im

frames, durs = [], []
shown = []
for ln in lines:
    if ln[-1][0] == "__TYPE__":                       # typed command
        prefix, (text, col, b) = ln[0], ln[1]
        for k in range(0, len(text) + 1, 3):
            frames.append(render(shown, [prefix, (text[:k], col, b), ("▌", "blue", False)])); durs.append(30)
        shown.append([prefix, (text, col, b)]); frames.append(render(shown)); durs.append(350)
    else:
        shown.append(ln); frames.append(render(shown)); durs.append(70)
durs[-1] = 3500
pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
frames[-1].save(str(OUT).replace(".gif", "-last.png"))
print(len(frames), "frames")
