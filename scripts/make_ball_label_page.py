"""A browser page for locating the BALL by hand, on frames the gate is scored on.

Two separate things are missing and this supplies both.

THE GATE IS NOT MEASURABLE. There are 13 hand-located balls. At n=13 a system
whose true accuracy is 0.95 measures as 0.667-0.986, so the gate cannot tell a
passing system from a 0.80 one. It needs about 200 located balls before the
question "is this 95%?" has an answer at all:

    n= 13   a true 0.95 reads 0.667-0.986
    n=100   a true 0.95 reads 0.888-0.978
    n=200   a true 0.95 reads 0.910-0.973

TRAINING LABELS for the measured failures. On five of the thirteen truth
frames `ball_clean` scores the true ball 0.00 even on a crop centred on it:
extreme close-ups, a loose-ball scramble, a ball held between two players'
legs, one in flight against a dark arena, one on the floor among feet.

THE BALL NEEDS A DIFFERENT TOOL FROM THE RIM. A rim is 400 px and two clicks
bound it. A ball is 15-25 px -- smaller than the mouse cursor -- so it is ONE
click for the centre, with the radius set by key and drawn live, and a loupe at
8x that is the only way the thing is visible at all. Getting a 20 px box from
two clicks would put more error in the box than the box has pixels.

THREE VERDICTS, NOT TWO, and the distinction is load-bearing:

    a click   the ball is here
    x         there is no ball in this picture (a negative, and a real label)
    ?         the ball is in here somewhere and I cannot find it

"?" is not a skip. `eval_rim_and_ball.py` excludes unknowns from BOTH sides of
the accuracy, so marking one is a statement that keeps the metric honest rather
than quietly inflating it -- and of 41 frames sampled this way before, 28 came
back "?", which is itself the most important measurement about this video.

Frames come from the EXISTING evaluation grid, so labelling them extends the
metric the gate is already defined on rather than inventing a second one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE = """<meta charset="utf-8"><title>Ball labeller</title>
<style>
:root{--bg:#12161c;--fg:#e6edf3;--dim:#8b98a5;--acc:#5cc0da;--ok:#3fb950;--no:#f85149;--line:#2b3440;--sel:#f0b400}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar{flex:1;height:6px;background:#1c2330;border-radius:3px;overflow:hidden;min-width:120px}#fill{height:100%;background:var(--acc);width:0}
main{display:flex;gap:14px;padding:10px 14px;align-items:flex-start}
#wrap{position:relative;cursor:crosshair;flex:none}
#img{display:block} #marks{position:absolute;inset:0;pointer-events:none}
aside{display:flex;flex-direction:column;gap:8px;min-width:430px}
#loupe{border:1px solid var(--line);background:#000;image-rendering:pixelated}
kbd{background:#1c2330;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px monospace}
button{background:#1c2330;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)} ul{margin:0;padding-left:18px;color:var(--dim);font-size:13px}
#state{font-weight:600} .found{color:var(--ok)} .none{color:var(--no)} .unk{color:var(--sel)}
.counts b{color:var(--fg)}
</style>
<header><b>Ball labeller &mdash; ONE click on the ball's centre</b>
<div id="bar"><div id="fill"></div></div>
<span><span id="idx">0</span>/<span id="tot">0</span></span>
<span class="counts"><b id="nball">0</b> located &middot; <b id="nnone">0</b> no ball &middot; <b id="nunk">0</b> can't find</span>
<span class="k" id="meta"></span><button onclick="exportLabels()">Export ball_labels.json</button></header>
<main><div id="wrap"><img id="img"><svg id="marks"></svg></div>
<aside><div>This frame: <span id="state">&mdash;</span> <span id="rad" style="color:var(--dim)"></span></div>
<canvas id="loupe" width="420" height="320"></canvas>
<ul>
<li><b>Click the centre of the ball.</b> The loupe shows 8&times; around the cursor &mdash; at this size it is the only way to see it.</li>
<li><kbd>[</kbd> / <kbd>]</kbd> shrink / grow the circle to match the ball. It is remembered for the next frame.</li>
<li><kbd>x</kbd> &mdash; <b>no ball in this picture</b> (out of frame, or a graphic). That is a real label, keep it.</li>
<li><kbd>?</kbd> &mdash; <b>the ball is in here somewhere and you cannot find it.</b> Not a skip: it is excluded from both sides of the score, so it keeps the number honest instead of flattering it.</li>
<li>If you are not sure whether the blob is the ball or someone's head, it is <kbd>?</kbd>.</li>
<li><kbd>&rarr;</kbd>/<kbd>n</kbd> next &middot; <kbd>&larr;</kbd> previous &middot; <kbd>u</kbd> undo &middot; <kbd>d</kbd> clear this frame</li>
<li>Saved in this browser as you go. Export when done and put it at <code>__OUT__</code>.</li>
</ul></aside></main>
<script>
const MANIFEST = __MANIFEST__;
const KEY = '__KEY__';
let i = 0, radius = 9, L = {}, SCALE = 1, lastXY = null;
try { L = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) { L = {} }
const $ = id => document.getElementById(id);
const img = $('img'), marks = $('marks'), loupe = $('loupe').getContext('2d');
function frame() { return MANIFEST.frames[i] }
function rec() { const f = frame().file; return L[f] || (L[f] = { t: frame().t, verdict: null, ball: null, radius: radius }) }
function persist() { try { localStorage.setItem(KEY, JSON.stringify(L)) } catch (e) {} }

function draw() {
  const r = rec(); marks.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  if (r.ball) {
    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', r.ball[0] * SCALE); c.setAttribute('cy', r.ball[1] * SCALE);
    c.setAttribute('r', Math.max(r.radius * SCALE, 3));
    c.setAttribute('fill', 'none'); c.setAttribute('stroke', '#3fb950');
    c.setAttribute('stroke-width', 2); marks.appendChild(c);
    const x = document.createElementNS(ns, 'path');
    x.setAttribute('d', `M${r.ball[0]*SCALE-14} ${r.ball[1]*SCALE}h9M${r.ball[0]*SCALE+5} ${r.ball[1]*SCALE}h9`
                      + `M${r.ball[0]*SCALE} ${r.ball[1]*SCALE-14}v9M${r.ball[0]*SCALE} ${r.ball[1]*SCALE+5}v9`);
    x.setAttribute('stroke', '#3fb950'); x.setAttribute('stroke-width', 1); marks.appendChild(x);
  }
  const s = $('state');
  s.textContent = r.verdict === 'ball' ? 'ball located' : r.verdict === 'none' ? 'no ball in this picture'
                : r.verdict === 'unknown' ? "in here somewhere, can't find it" : '—';
  s.className = r.verdict === 'ball' ? 'found' : r.verdict === 'none' ? 'none' : r.verdict === 'unknown' ? 'unk' : '';
  $('rad').textContent = r.verdict === 'ball' ? `radius ${r.radius} px` : '';
  let b = 0, n = 0, u = 0;
  for (const k in L) { const v = L[k].verdict; if (v === 'ball') b++; else if (v === 'none') n++; else if (v === 'unknown') u++; }
  $('nball').textContent = b; $('nnone').textContent = n; $('nunk').textContent = u;
  $('fill').style.width = (100 * (b + n + u) / MANIFEST.frames.length) + '%';
  $('idx').textContent = i + 1; $('tot').textContent = MANIFEST.frames.length;
  $('meta').textContent = `${frame().file}  t=${frame().t.toFixed(1)}s`;
}
function show() {
  img.onload = () => {
    const room = Math.min(window.innerWidth - 470, 1240);
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
  const [x, y] = lastXY, z = 8, w = 420, h = 320;
  loupe.imageSmoothingEnabled = false;
  loupe.clearRect(0, 0, w, h);
  loupe.drawImage(img, x - w/(2*z), y - h/(2*z), w/z, h/z, 0, 0, w, h);
  loupe.strokeStyle = 'rgba(240,180,0,0.9)'; loupe.lineWidth = 1;
  loupe.beginPath(); loupe.moveTo(w/2, 0); loupe.lineTo(w/2, h);
  loupe.moveTo(0, h/2); loupe.lineTo(w, h/2); loupe.stroke();
  loupe.strokeStyle = 'rgba(60,200,90,0.9)';
  loupe.beginPath(); loupe.arc(w/2, h/2, radius*z, 0, 6.2832); loupe.stroke();
}
$('wrap').addEventListener('mousemove', ev => { lastXY = at(ev); paintLoupe() });
$('wrap').addEventListener('click', ev => {
  const [x, y] = at(ev), r = rec();
  r.verdict = 'ball'; r.ball = [x, y]; r.radius = radius;
  persist(); draw();
});
document.addEventListener('keydown', ev => {
  const r = rec();
  if (ev.key === 'ArrowRight' || ev.key === 'n') { i = Math.min(i+1, MANIFEST.frames.length-1); show() }
  else if (ev.key === 'ArrowLeft') { i = Math.max(i-1, 0); show() }
  else if (ev.key === '[') { radius = Math.max(3, radius-1); if (r.verdict==='ball') r.radius=radius; persist(); draw(); paintLoupe() }
  else if (ev.key === ']') { radius = Math.min(60, radius+1); if (r.verdict==='ball') r.radius=radius; persist(); draw(); paintLoupe() }
  else if (ev.key === 'x') { r.verdict = 'none'; r.ball = null; persist(); draw() }
  else if (ev.key === '?' || ev.key === '/') { r.verdict = 'unknown'; r.ball = null; persist(); draw() }
  else if (ev.key === 'u' || ev.key === 'd') { r.verdict = null; r.ball = null; persist(); draw() }
});
function exportLabels() {
  const frames = [];
  for (const file in L) {
    const r = L[file];
    if (!r.verdict) continue;
    frames.push({ t: r.t, verdict: r.verdict,
                  ball: r.ball ? [Math.round(r.ball[0]), Math.round(r.ball[1])] : null,
                  radius: r.ball ? r.radius : 0, file: file });
  }
  frames.sort((a, b) => a.t - b.t);
  const blob = new Blob([JSON.stringify({
    note: "Ball located by hand, one click per centre, radius set by key. verdict is "
        + "'ball' (located), 'none' (no ball in the picture -- a negative), or 'unknown' "
        + "(present but not findable by eye, EXCLUDED from both sides of the accuracy).",
    video: MANIFEST.video, frame_size: MANIFEST.size, frames: frames }, null, 1)],
    { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'ball_labels.json'; a.click();
}
show();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", required=True,
                        help="a file with frames:[{t}] -- the evaluation grid, so that "
                             "labelling extends the metric the gate already uses")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--key", default="balllabels")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import cv2

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = json.load(open(args.frames))["frames"]
    if args.limit:
        rows = rows[:args.limit]
    capture = cv2.VideoCapture(args.video)
    size, written = None, []
    for row in rows:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        size = [frame.shape[1], frame.shape[0]]
        name = f"t{row['t']:08.1f}.jpg"
        # Quality 98: a 15 px ball is a handful of pixels and JPEG ringing at 94
        # can be the difference between findable and not.
        cv2.imwrite(str(out / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
        written.append({"file": name, "t": round(float(row["t"]), 2)})
    capture.release()

    manifest = {"video": args.video, "size": size, "frames": written}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "label.html").write_text(
        PAGE.replace("__MANIFEST__", json.dumps(manifest))
            .replace("__KEY__", args.key)
            .replace("__OUT__", str(out / "ball_labels.json")))
    print(f"{len(written)} frames written")
    print(f"  open {out / 'label.html'} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
