"""Label the ball AND who is holding it, on the same frame, in one pass.

WHY BOTH. The ball is invisible in a large share of broadcast frames -- behind
the dribbler's own body, lost in a crowd of arms, smeared by motion. The
detector's proposals contain it 6 to 8 times in 13, and that is the wall the
last round hit: ranking candidates cannot find a ball that was never proposed.
But a person watching knows who has it anyway, from the way he is turned and
what everyone else is doing. So the handler is labellable on frames where the
ball is not, and it is the more useful of the two for almost every question
this project answers -- "who did what" needs possession, not a rectangle.

So each frame gets two verdicts, and they are INDEPENDENT:

    ball      located / not in this picture / in here but I can't find it
    handler   this player / nobody has it (in flight, loose) / he has no box

"I cannot find the ball, and player 4 has it" is the most valuable label in the
set, not a failure to label. "unknown" on either side is excluded from both
sides of that side's accuracy, so marking one keeps the number honest.

"he has no box" is a third handler verdict and it is a measurement of the
DETECTOR, not of the labeller: the handler is on screen and the detector missed
him. Those frames are exactly the training examples the player class needs.

WHICH FRAMES. Half are drawn from the frames where the ball model's best
candidate is under HARD_CONF -- where it is failing, and where a label is worth
most -- and half uniformly at random, so the set can also be used to measure
without the measurement being taken only on hard cases. Frames are spread over
all three broadcasts, and the split is recorded per frame.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: A frame whose best ball candidate is under this is one the model is failing.
HARD_CONF = 0.35
#: Player boxes offered for the handler choice must be at least this confident.
PLAYER_CONF = 0.35
#: Which broadcasts have labelling artefacts, from `data/games.json`.
#:
#: This was a hardcoded dict of (video, clip index, label) in THREE scripts, and
#: `rebuild_detector_dataset.py` kept two more keyed the same way. Five copies of
#: the same five facts, and a fourth broadcast meant editing all of them and
#: getting every one to agree -- which is another way of saying a fourth
#: broadcast could not be added. See `courtvision.games`.
def _games():
    from courtvision.games import registry
    return {g.clip_detections.stem: (str(g.video), str(g.clip_index), g.label)
            for g in registry().values()}


GAMES = _games()

PAGE = """<meta charset="utf-8"><title>Ball + handler labeller</title>
<style>
:root{--bg:#0f1319;--fg:#e6edf3;--dim:#8b98a5;--acc:#5cc0da;--ok:#3fb950;--no:#f85149;
      --line:#2b3440;--sel:#f0b400;--hand:#ff8a3d}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar{flex:1;height:6px;background:#1c2330;border-radius:3px;overflow:hidden;min-width:120px}
#fill{height:100%;background:var(--acc);width:0}
main{display:flex;gap:14px;padding:10px 14px;align-items:flex-start}
#wrap{position:relative;cursor:crosshair;flex:none}
#img{display:block} #marks{position:absolute;inset:0;pointer-events:none}
aside{display:flex;flex-direction:column;gap:9px;min-width:min(440px,100%)}
#loupe{border:1px solid var(--line);background:#000;image-rendering:pixelated}
kbd{background:#1c2330;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px monospace}
button{background:#1c2330;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)} ul{margin:0;padding-left:18px;color:var(--dim);font-size:13px}
li{margin:3px 0}
.row{display:flex;gap:10px;align-items:baseline}
.tag{font:11px/1 monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);min-width:62px}
.val{font-weight:600} .found{color:var(--ok)} .none{color:var(--no)} .unk{color:var(--sel)}
.hand{color:var(--hand)} .counts b{color:var(--fg)}
</style>
<header><b>Ball &amp; handler</b>
<div id="bar"><div id="fill"></div></div>
<span><span id="idx">0</span>/<span id="tot">0</span></span>
<span class="counts"><b id="ndone">0</b> done &middot; <b id="nball">0</b> balls &middot;
<b id="nhand">0</b> handlers</span>
<span id="meta" style="color:var(--dim)"></span>
<button onclick="exportLabels()">Export possession_labels.json</button></header>
<main><div id="wrap"><img id="img"><svg id="marks"></svg></div>
<aside>
<div class="row"><span class="tag">ball</span><span class="val" id="bstate">&mdash;</span>
<span id="rad" style="color:var(--dim)"></span></div>
<div class="row"><span class="tag">handler</span><span class="val" id="hstate">&mdash;</span></div>
<canvas id="loupe" width="440" height="330"></canvas>
<ul>
<li><b>Click the centre of the ball.</b> The loupe is 8&times; around the cursor.
<kbd>[</kbd>/<kbd>]</kbd> size the circle to it.</li>
<li><kbd>x</kbd> no ball in this picture &middot; <kbd>?</kbd> it is in here and you cannot find it.
Both are real labels; <kbd>?</kbd> is excluded from the score rather than counted wrong.</li>
<li><b>Shift-click the player holding it</b> &mdash; on his box to pick that box, or
anywhere on him if the detector drew no box (that is recorded as a miss, and it is
a measurement of the detector).</li>
<li><kbd>n</kbd> nobody has it &mdash; in flight, loose, or between players.</li>
<li><b>The two are independent.</b> "Cannot find the ball, and that player has it" is
the most useful label here, not a half-finished one.</li>
<li><kbd>&rarr;</kbd> next &middot; <kbd>&larr;</kbd> previous &middot; <kbd>d</kbd> clear this frame</li>
<li>Saved in this browser as you go. Export when you stop &mdash; part-way is fine.</li>
</ul></aside></main>
<script>
const MANIFEST = __MANIFEST__;
const KEY = '__KEY__';
let i = 0, radius = 9, L = {}, SCALE = 1, lastXY = null;
try { L = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) { L = {} }
const $ = id => document.getElementById(id);
const img = $('img'), marks = $('marks'), loupe = $('loupe').getContext('2d');
const ns = 'http://www.w3.org/2000/svg';
function frame() { return MANIFEST.frames[i] }
function rec() {
  const f = frame().file;
  return L[f] || (L[f] = { t: frame().t, game: frame().game, pick: frame().pick,
                           ball: null, ballVerdict: null, radius: radius,
                           handler: null, handlerVerdict: null, handlerAt: null });
}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(L)) } catch (e) {} }

function box(b, colour, width, dash) {
  const r = document.createElementNS(ns, 'rect');
  r.setAttribute('x', b[0] * SCALE); r.setAttribute('y', b[1] * SCALE);
  r.setAttribute('width', (b[2] - b[0]) * SCALE); r.setAttribute('height', (b[3] - b[1]) * SCALE);
  r.setAttribute('fill', 'none'); r.setAttribute('stroke', colour);
  r.setAttribute('stroke-width', width); if (dash) r.setAttribute('stroke-dasharray', dash);
  r.setAttribute('rx', 3); return r;
}
function draw() {
  const r = rec(); marks.innerHTML = '';
  frame().boxes.forEach((b, n) => {
    const chosen = r.handlerVerdict === 'box' && r.handler === n;
    marks.appendChild(box(b, chosen ? 'var(--hand)' : 'rgba(120,150,180,.55)', chosen ? 3 : 1));
  });
  if (r.handlerVerdict === 'missing' && r.handlerAt) {
    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', r.handlerAt[0] * SCALE); c.setAttribute('cy', r.handlerAt[1] * SCALE);
    c.setAttribute('r', 22 * SCALE); c.setAttribute('fill', 'none');
    c.setAttribute('stroke', 'var(--hand)'); c.setAttribute('stroke-width', 2);
    c.setAttribute('stroke-dasharray', '5 4'); marks.appendChild(c);
  }
  if (r.ball) {
    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', r.ball[0] * SCALE); c.setAttribute('cy', r.ball[1] * SCALE);
    c.setAttribute('r', Math.max(r.radius * SCALE, 3)); c.setAttribute('fill', 'none');
    c.setAttribute('stroke', 'var(--ok)'); c.setAttribute('stroke-width', 2);
    marks.appendChild(c);
  }
  const bs = $('bstate');
  bs.textContent = r.ballVerdict === 'ball' ? 'located'
                 : r.ballVerdict === 'none' ? 'not in this picture'
                 : r.ballVerdict === 'unknown' ? "in here, can't find it" : '—';
  bs.className = 'val ' + (r.ballVerdict === 'ball' ? 'found' : r.ballVerdict === 'none' ? 'none'
                           : r.ballVerdict === 'unknown' ? 'unk' : '');
  $('rad').textContent = r.ballVerdict === 'ball' ? `radius ${r.radius} px` : '';
  const hs = $('hstate');
  hs.textContent = r.handlerVerdict === 'box' ? `player ${r.handler + 1}`
                 : r.handlerVerdict === 'missing' ? 'a player with no box'
                 : r.handlerVerdict === 'nobody' ? 'nobody has it' : '—';
  hs.className = 'val ' + (r.handlerVerdict ? 'hand' : '');
  let done = 0, b = 0, h = 0;
  for (const k in L) {
    const v = L[k];
    if (v.ballVerdict || v.handlerVerdict) done++;
    if (v.ballVerdict === 'ball') b++;
    if (v.handlerVerdict === 'box' || v.handlerVerdict === 'missing') h++;
  }
  $('ndone').textContent = done; $('nball').textContent = b; $('nhand').textContent = h;
  $('fill').style.width = (100 * done / MANIFEST.frames.length) + '%';
  $('idx').textContent = i + 1; $('tot').textContent = MANIFEST.frames.length;
  $('meta').textContent = `${frame().game}  t=${frame().t.toFixed(1)}s  ${frame().pick}`;
}
function fit() {
  // A narrow window stacks the panel under the picture instead of beside it,
  // otherwise the frame is squeezed to nothing and there is nothing to label.
  const beside = window.innerWidth > 1180;
  document.querySelector('main').style.flexDirection = beside ? 'row' : 'column';
  const room = Math.min(beside ? window.innerWidth - 480 : window.innerWidth - 28, 1400);
  SCALE = Math.min(1, room / (img.naturalWidth || 1280));
  img.width = img.naturalWidth * SCALE; img.height = img.naturalHeight * SCALE;
  marks.setAttribute('width', img.width); marks.setAttribute('height', img.height);
  draw();
}
window.addEventListener('resize', () => { if (img.naturalWidth) fit() });
function show() {
  img.onload = () => {
    const beside = window.innerWidth > 1180;
    const room = Math.min(beside ? window.innerWidth - 480 : window.innerWidth - 28, 1400);
    SCALE = Math.min(1, room / img.naturalWidth);
    img.width = img.naturalWidth * SCALE; img.height = img.naturalHeight * SCALE;
    marks.setAttribute('width', img.width); marks.setAttribute('height', img.height);
    draw();
  };
  img.src = 'images/' + frame().file;
}
function at(ev) { const b = img.getBoundingClientRect();
  return [(ev.clientX - b.left) / SCALE, (ev.clientY - b.top) / SCALE] }
function paintLoupe() {
  if (!lastXY) return;
  const [x, y] = lastXY, z = 8, w = 440, h = 330;
  loupe.imageSmoothingEnabled = false; loupe.clearRect(0, 0, w, h);
  loupe.drawImage(img, x - w/(2*z), y - h/(2*z), w/z, h/z, 0, 0, w, h);
  loupe.strokeStyle = 'rgba(240,180,0,.9)'; loupe.lineWidth = 1;
  loupe.beginPath(); loupe.moveTo(w/2, 0); loupe.lineTo(w/2, h);
  loupe.moveTo(0, h/2); loupe.lineTo(w, h/2); loupe.stroke();
  loupe.strokeStyle = 'rgba(60,200,90,.9)';
  loupe.beginPath(); loupe.arc(w/2, h/2, radius*z, 0, 6.2832); loupe.stroke();
}
$('wrap').addEventListener('mousemove', ev => { lastXY = at(ev); paintLoupe() });
$('wrap').addEventListener('click', ev => {
  const [x, y] = at(ev), r = rec();
  if (ev.shiftKey) {
    let found = -1;
    frame().boxes.forEach((b, n) => {
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]
          && (found < 0 || (b[2]-b[0])*(b[3]-b[1])
              < (frame().boxes[found][2]-frame().boxes[found][0])
                * (frame().boxes[found][3]-frame().boxes[found][1]))) found = n;
    });
    if (found >= 0) { r.handlerVerdict = 'box'; r.handler = found; r.handlerAt = null }
    else { r.handlerVerdict = 'missing'; r.handler = null; r.handlerAt = [x, y] }
  } else {
    r.ballVerdict = 'ball'; r.ball = [x, y]; r.radius = radius;
  }
  persist(); draw();
});
document.addEventListener('keydown', ev => {
  const r = rec();
  if (ev.key === 'ArrowRight') { i = Math.min(i+1, MANIFEST.frames.length-1); show() }
  else if (ev.key === 'ArrowLeft') { i = Math.max(i-1, 0); show() }
  else if (ev.key === '[') { radius = Math.max(3, radius-1); if (r.ballVerdict==='ball') r.radius=radius; persist(); draw(); paintLoupe() }
  else if (ev.key === ']') { radius = Math.min(60, radius+1); if (r.ballVerdict==='ball') r.radius=radius; persist(); draw(); paintLoupe() }
  else if (ev.key === 'x') { r.ballVerdict = 'none'; r.ball = null; persist(); draw() }
  else if (ev.key === '?' || ev.key === '/') { r.ballVerdict = 'unknown'; r.ball = null; persist(); draw() }
  else if (ev.key === 'n') { r.handlerVerdict = 'nobody'; r.handler = null; r.handlerAt = null; persist(); draw() }
  else if (ev.key === 'd') { r.ballVerdict = null; r.ball = null; r.handlerVerdict = null;
                             r.handler = null; r.handlerAt = null; persist(); draw() }
});
function exportLabels() {
  const frames = [];
  for (const file in L) {
    const r = L[file];
    if (!r.ballVerdict && !r.handlerVerdict) continue;
    const src = MANIFEST.frames.find(f => f.file === file) || {};
    frames.push({ file: file, t: r.t, game: r.game, pick: r.pick,
                  ball_verdict: r.ballVerdict,
                  ball: r.ball ? [Math.round(r.ball[0]), Math.round(r.ball[1])] : null,
                  radius: r.ball ? r.radius : 0,
                  handler_verdict: r.handlerVerdict,
                  handler_box: (r.handlerVerdict === 'box' && src.boxes)
                    ? src.boxes[r.handler].map(v => Math.round(v)) : null,
                  handler_at: r.handlerAt ? r.handlerAt.map(v => Math.round(v)) : null });
  }
  frames.sort((a, b) => (a.game + a.t) < (b.game + b.t) ? -1 : 1);
  const blob = new Blob([JSON.stringify({
    note: "Ball and ball-handler located by hand on the same frames, independently. "
        + "ball_verdict: 'ball' (located), 'none' (not in the picture), 'unknown' (present, "
        + "not findable by eye -- excluded from both sides of the ball accuracy). "
        + "handler_verdict: 'box' (that detector box has it), 'missing' (a player has it and "
        + "the detector drew no box for him -- a detector miss, with his position in "
        + "handler_at), 'nobody' (in flight, loose, contested). pick is 'hard' (the ball "
        + "model's best candidate was under 0.35 here) or 'random'.",
    frame_size: MANIFEST.size, games: MANIFEST.games, frames: frames }, null, 1)],
    { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'possession_labels.json'; a.click();
}
show();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/labeling/possession")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--hard-share", type=float, default=0.5)
    parser.add_argument("--key", default="possessionlabels")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import cv2

    random.seed(args.seed)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # ---- pick the frames, half where the ball model is failing --------------
    hard, easy = [], []
    for stem, (video, index, label) in GAMES.items():
        cache_path = Path("outputs") / f"{stem}.json"
        if not cache_path.exists() or not Path(video).exists() or not Path(index).exists():
            print(f"  skipping {label}: no cache or video")
            continue
        cached = json.load(open(cache_path))
        starts = {c["clip"]: float(c.get("start_s", float(c["video_s"]) - 3.0))
                  for c in json.load(open(index))["clips"] if c.get("clip")}
        for clip, rows in cached["clips"].items():
            if clip not in starts:
                continue
            for row in rows:
                balls = [b[1] for b in row["d"] if b[0] == "b"]
                on = row.get("on") or []
                people = [b for b in row["d"] if b[0] in ("p", "h")]
                boxes = [b[2:] for n, b in enumerate(people)
                         if b[1] >= PLAYER_CONF and (n >= len(on) or on[n])]
                if len(boxes) < 4:
                    continue          # a replay or a close-up, not a possession
                item = (video, label, starts[clip] + row["f"] / 30.0, boxes)
                (hard if (not balls or max(balls) < HARD_CONF) else easy).append(item)

    want_hard = int(args.frames * args.hard_share)
    random.shuffle(hard)
    random.shuffle(easy)
    picked = ([(i, "hard") for i in hard[:want_hard]]
              + [(i, "random") for i in easy[:args.frames - want_hard]])
    random.shuffle(picked)
    print(f"  {len(hard)} frames where the ball model is failing, {len(easy)} where it is not")
    print(f"  labelling {sum(1 for _, p in picked if p == 'hard')} hard "
          f"and {sum(1 for _, p in picked if p == 'random')} random")

    # ---- cut them out of the broadcasts ------------------------------------
    by_video = defaultdict(list)
    for (video, label, when, boxes), pick in picked:
        by_video[video].append((when, label, boxes, pick))
    manifest, size = [], None
    for video, spots in by_video.items():
        capture = cv2.VideoCapture(video)
        for when, label, boxes, pick in sorted(spots):
            capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            size = [frame.shape[1], frame.shape[0]]
            name = f"{Path(video).stem}_{int(round(when * 10)):07d}.jpg"
            cv2.imwrite(str(out / "images" / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            manifest.append({"file": name, "t": round(when, 1), "game": label,
                             "pick": pick,
                             "boxes": [[round(v, 1) for v in b] for b in boxes]})
        capture.release()
    manifest.sort(key=lambda m: (m["game"], m["t"]))

    page = (PAGE.replace("__MANIFEST__", json.dumps(
        {"size": size, "games": sorted({m["game"] for m in manifest}),
         "frames": manifest}))
        .replace("__KEY__", args.key))
    (out / "label.html").write_text(page)
    json.dump({"size": size, "frames": manifest}, open(out / "manifest.json", "w"))
    print(f"  {len(manifest)} frames written")
    print(f"  -> open {out / 'label.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
