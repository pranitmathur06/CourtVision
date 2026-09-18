"""Who has the ball. One keystroke a frame, and the model's guess is not shown.

WHY A SECOND ROUND, AND WHY ONLY THE HANDLER. Measured on 92 hand-labelled
frames where somebody had the ball, the detector names the right player 44.6%
of the time. It is not sloppy boxes -- scoring by IoU 0.3, or by one box's
centre landing inside the other, gives the identical count. It names the wrong
man. And no rule downstream rescues it: nearest box to the ball, a box
containing it, nearest in body-heights, any of them voted over a 1.2 s window,
all land between 42% and 49%, and even with the ball position supplied BY HAND
the ceiling is 58.7%.

The reason is that the handler class is trained on labels a proximity rule
produced: `harvest_handler_labels.py` calls whoever is nearest the most
confident ball the handler. So the model is being taught a rule whose own
ceiling is 59%, on a game where a defender's hands are often nearer the ball
than the holder's. It cannot beat its teacher. Human labels are the only way
past that, and this collects them in volume.

THE MODEL'S GUESS IS DELIBERATELY NOT SHOWN. Highlighting it and asking for a
yes would be three times faster and would produce labels that agree with the
thing being corrected. Every frame here is judged cold.

SPEED INSTEAD COMES FROM the shape of the question: the boxes are numbered
left to right, one digit answers, and the page advances by itself. No ball, no
loupe, no zoom.

WHICH FRAMES. Two thirds are frames where the detector's handler class and the
nearest-player-to-the-ball rule DISAGREE -- one of them is wrong there, so the
label settles something -- and one third uniformly at random so the set can
still measure. Frames with no ball candidate and no handler box at all are
skipped: they are dead balls and teach nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Ball candidates below this are not evidence of anything.
BALL_CONF = 0.25
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

PAGE = """<meta charset="utf-8"><title>Who has the ball</title>
<style>
:root{--bg:#0f1319;--fg:#e6edf3;--dim:#8b98a5;--acc:#5cc0da;--line:#2b3440;--hand:#ff8a3d;--no:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar{flex:1;height:6px;background:#1c2330;border-radius:3px;overflow:hidden;min-width:120px}
#fill{height:100%;background:var(--acc);width:0}
main{padding:10px 14px}
#wrap{position:relative;cursor:pointer;display:inline-block}
#img{display:block}#marks{position:absolute;inset:0;pointer-events:none}
button{background:#1c2330;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}
kbd{background:#1c2330;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px monospace}
#help{color:var(--dim);padding:6px 14px 12px;font-size:13px}
#said{font-weight:600;color:var(--hand)}
</style>
<header><b>Who has the ball?</b>
<div id="bar"><div id="fill"></div></div>
<span><span id="idx">0</span>/<span id="tot">0</span></span>
<span style="color:var(--dim)"><b id="ndone">0</b> answered &middot; <span id="rate"></span></span>
<span id="said"></span>
<span id="meta" style="color:var(--dim)"></span>
<button onclick="exportLabels()">Export handler_labels.json</button></header>
<main><div id="wrap"><img id="img"><svg id="marks"></svg></div></main>
<div id="help"><b>Click the player holding the ball</b>, or press his number.
<kbd>n</kbd> nobody has it &mdash; in flight, loose, contested, dead ball.
<kbd>?</kbd> can't tell. <kbd>m</kbd> somebody has it and he has <b>no box</b> &mdash;
then click him. It moves on by itself; <kbd>&larr;</kbd> goes back to fix one.
Saved as you go; export whenever you stop.</div>
<script>
const MANIFEST = __MANIFEST__;
const KEY = '__KEY__';
let i = 0, L = {}, SCALE = 1, missing = false, started = Date.now(), answered0 = 0;
try { L = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) { L = {} }
const $ = id => document.getElementById(id);
const img = $('img'), marks = $('marks');
const ns = 'http://www.w3.org/2000/svg';
function frame() { return MANIFEST.frames[i] }
function rec() { const f = frame().file;
  return L[f] || (L[f] = { t: frame().t, game: frame().game, pick: frame().pick,
                           verdict: null, handler: null, at: null }) }
function persist() { try { localStorage.setItem(KEY, JSON.stringify(L)) } catch (e) {} }
function draw() {
  const r = rec(); marks.innerHTML = '';
  frame().boxes.forEach((b, n) => {
    const on = r.verdict === 'box' && r.handler === n;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', b[0]*SCALE); rect.setAttribute('y', b[1]*SCALE);
    rect.setAttribute('width', (b[2]-b[0])*SCALE); rect.setAttribute('height', (b[3]-b[1])*SCALE);
    rect.setAttribute('fill', 'none'); rect.setAttribute('rx', 3);
    rect.setAttribute('stroke', on ? 'var(--hand)' : 'rgba(130,160,190,.65)');
    rect.setAttribute('stroke-width', on ? 4 : 1.5);
    marks.appendChild(rect);
    const tag = document.createElementNS(ns, 'text');
    tag.setAttribute('x', b[0]*SCALE + 3); tag.setAttribute('y', b[1]*SCALE - 4);
    tag.setAttribute('fill', on ? 'var(--hand)' : 'rgba(150,180,210,.9)');
    tag.setAttribute('font-size', 15); tag.setAttribute('font-family', 'monospace');
    tag.textContent = n + 1; marks.appendChild(tag);
  });
  if (r.verdict === 'missing' && r.at) {
    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', r.at[0]*SCALE); c.setAttribute('cy', r.at[1]*SCALE);
    c.setAttribute('r', 24*SCALE); c.setAttribute('fill','none');
    c.setAttribute('stroke','var(--hand)'); c.setAttribute('stroke-width',3);
    c.setAttribute('stroke-dasharray','6 4'); marks.appendChild(c);
  }
  let done = 0;
  for (const k in L) if (L[k].verdict) done++;
  $('ndone').textContent = done;
  $('fill').style.width = (100*done/MANIFEST.frames.length) + '%';
  $('idx').textContent = i+1; $('tot').textContent = MANIFEST.frames.length;
  $('meta').textContent = `${frame().game} t=${frame().t.toFixed(1)}s`;
  $('said').textContent = missing ? 'click the player with no box'
    : r.verdict === 'box' ? `player ${r.handler+1}`
    : r.verdict === 'nobody' ? 'nobody' : r.verdict === 'unknown' ? "can't tell"
    : r.verdict === 'missing' ? 'no box for him' : '';
  const secs = (Date.now()-started)/1000, n = done - answered0;
  $('rate').textContent = n > 4 ? `${(secs/n).toFixed(1)}s a frame` : '';
}
function show() {
  img.onload = () => {
    SCALE = Math.min(1, (window.innerWidth - 40) / img.naturalWidth);
    img.width = img.naturalWidth*SCALE; img.height = img.naturalHeight*SCALE;
    marks.setAttribute('width', img.width); marks.setAttribute('height', img.height);
    draw();
  };
  missing = false;
  img.src = 'images/' + frame().file;
}
function next() { if (i < MANIFEST.frames.length-1) { i++; show() } else draw() }
function choose(n) {
  const r = rec();
  if (n < 0 || n >= frame().boxes.length) return;
  r.verdict = 'box'; r.handler = n; r.at = null; persist(); draw(); setTimeout(next, 90);
}
$('wrap').addEventListener('click', ev => {
  const b = img.getBoundingClientRect();
  const x = (ev.clientX-b.left)/SCALE, y = (ev.clientY-b.top)/SCALE, r = rec();
  if (missing) { r.verdict = 'missing'; r.handler = null; r.at = [x,y];
                 missing = false; persist(); draw(); setTimeout(next, 90); return }
  let found = -1, area = Infinity;
  frame().boxes.forEach((bx, n) => {
    const a = (bx[2]-bx[0])*(bx[3]-bx[1]);
    if (x>=bx[0] && x<=bx[2] && y>=bx[1] && y<=bx[3] && a < area) { found = n; area = a }
  });
  if (found >= 0) choose(found);
});
document.addEventListener('keydown', ev => {
  const r = rec();
  if (ev.key >= '1' && ev.key <= '9') choose(parseInt(ev.key,10)-1);
  else if (ev.key === '0') choose(9);
  else if (ev.key === 'n') { r.verdict='nobody'; r.handler=null; r.at=null; persist(); draw(); setTimeout(next,90) }
  else if (ev.key === '?' || ev.key === '/') { r.verdict='unknown'; r.handler=null; r.at=null; persist(); draw(); setTimeout(next,90) }
  else if (ev.key === 'm') { missing = true; draw() }
  else if (ev.key === 'ArrowLeft') { i = Math.max(i-1,0); show() }
  else if (ev.key === 'ArrowRight') next();
  else if (ev.key === 'd') { r.verdict=null; r.handler=null; r.at=null; persist(); draw() }
});
function exportLabels() {
  const frames = [];
  for (const file in L) {
    const r = L[file]; if (!r.verdict) continue;
    const src = MANIFEST.frames.find(f => f.file === file) || {};
    frames.push({ file, t: r.t, game: r.game, pick: r.pick, verdict: r.verdict,
      handler_box: (r.verdict === 'box' && src.boxes) ? src.boxes[r.handler].map(Math.round) : null,
      handler_at: r.at ? r.at.map(Math.round) : null,
      boxes: src.boxes || [] });
  }
  frames.sort((a,b) => (a.game+a.t) < (b.game+b.t) ? -1 : 1);
  const blob = new Blob([JSON.stringify({
    note: "Ball-handler by hand, judged without seeing any model's guess. verdict: 'box' "
        + "(that box has it), 'missing' (somebody has it and no box was drawn for him -- his "
        + "position is handler_at, and it is a player-detector miss), 'nobody' (in flight, "
        + "loose, dead), 'unknown' (excluded from both sides of the score). pick is "
        + "'disagree' (the handler class and the nearest-to-ball rule chose different players "
        + "here) or 'random'. boxes are the detector's, in the order shown.",
    frame_size: MANIFEST.size, frames }, null, 1)], { type:'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'handler_labels.json'; a.click();
}
show();
</script>
"""


def near(box, point):
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/labeling/handler")
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--disagree-share", type=float, default=0.66)
    parser.add_argument("--key", default="handlerlabels")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    import cv2

    random.seed(args.seed)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    already = set()
    seen = Path("data/labels/possession_labels.json")
    if seen.exists():
        already = {(r["game"], round(r["t"], 1))
                   for r in json.load(open(seen))["frames"]}

    disagree, plain = [], []
    for stem, (video, index, label) in GAMES.items():
        cache = Path("outputs") / f"{stem}.json"
        if not cache.exists() or not Path(video).exists() or not Path(index).exists():
            continue
        cached = json.load(open(cache))
        starts = {c["clip"]: float(c.get("start_s", float(c["video_s"]) - 3.0))
                  for c in json.load(open(index))["clips"] if c.get("clip")}
        for clip, rows in cached["clips"].items():
            if clip not in starts:
                continue
            for row in rows:
                on = row.get("on") or []
                people = [b for b in row["d"] if b[0] in ("p", "h")]
                boxes = [b[2:] for n, b in enumerate(people)
                         if b[1] >= PLAYER_CONF and (n >= len(on) or on[n])]
                if len(boxes) < 4 or len(boxes) > 10:
                    continue
                balls = [b for b in row["d"] if b[0] == "b" and b[1] >= BALL_CONF]
                handlers = [b for b in people if b[0] == "h" and b[1] >= PLAYER_CONF]
                if not balls and not handlers:
                    continue                      # a dead ball; teaches nothing
                when = round(starts[clip] + row["f"] / 30.0, 1)
                if (label, when) in already:
                    continue
                item = (video, label, when, boxes)
                split = "random"
                if balls and handlers:
                    top = max(balls, key=lambda b: b[1])
                    point = ((top[2] + top[4]) / 2, (top[3] + top[5]) / 2)
                    by_ball = min(boxes, key=lambda b: near(b, point))
                    claimed = max(handlers, key=lambda b: b[1])[2:]
                    if near(by_ball, ((claimed[0] + claimed[2]) / 2,
                                      (claimed[1] + claimed[3]) / 2)) > 5:
                        split = "disagree"
                (disagree if split == "disagree" else plain).append(item)

    want = int(args.frames * args.disagree_share)
    random.shuffle(disagree)
    random.shuffle(plain)
    picked = ([(x, "disagree") for x in disagree[:want]]
              + [(x, "random") for x in plain[:args.frames - want]])
    random.shuffle(picked)
    print(f"  {len(disagree)} frames where the two rules disagree, {len(plain)} where they do not")
    print(f"  taking {sum(1 for _, p in picked if p == 'disagree')} disagreements "
          f"and {sum(1 for _, p in picked if p == 'random')} random")

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
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            manifest.append({"file": name, "t": when, "game": label, "pick": pick,
                             "boxes": [[round(v, 1) for v in b]
                                       for b in sorted(boxes, key=lambda b: b[0])]})
        capture.release()
    manifest.sort(key=lambda m: (m["game"], m["t"]))
    page = (PAGE.replace("__MANIFEST__",
                         json.dumps({"size": size, "frames": manifest}))
            .replace("__KEY__", args.key))
    (out / "label.html").write_text(page)
    json.dump({"size": size, "frames": manifest}, open(out / "manifest.json", "w"))
    print(f"  {len(manifest)} frames written")
    print(f"  -> open {out / 'label.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
