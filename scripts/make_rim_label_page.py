"""A browser page for locating rims by hand, in the shape the trainer already reads.

The rim's remaining misses are close-up and alternate-camera frames, and the
measured reason is that there is no training data for them: 49 hand anchors
multiplied by ORB to 173 labels, of which only 53 come from the under-basket
family. More labels have to come from a person, and a person needs a tool
better than reading coordinates off a printed grid.

This is the court labeller's workflow pointed at rims. A rim takes TWO CLICKS --
the left edge of the ring, then the right edge -- which is the whole label:
the centre is the midpoint and the width is the separation, exactly the
{"t", "rim": [cx, cy], "width"} that `add_rim_hand_labels.py` consumes. Nothing
has to be converted afterwards, so nothing can be converted wrongly.

Two clicks rather than a dragged box because the ring is an ELLIPSE seen at an
angle and its left and right extremes are unambiguous from any viewpoint, while
the top and bottom of a box around it are not: under the basket the near rim
edge and the far one are at different depths and a labeller has to guess where
the box ends. The stored height follows the same 0.45 ratio the existing hand
labels use, so old and new labels stay one population.

Everything is saved in the browser as it goes and exported as one JSON, so a
closed tab costs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE = """<meta charset="utf-8"><title>Rim labeller</title>
<style>
:root{--bg:#12161c;--fg:#e6edf3;--dim:#8b98a5;--acc:#5cc0da;--ok:#3fb950;--no:#f85149;--line:#2b3440;--sel:#f0b400}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar{flex:1;height:6px;background:#1c2330;border-radius:3px;overflow:hidden;min-width:140px}#fill{height:100%;background:var(--acc);width:0}
main{display:flex;gap:14px;padding:10px 14px;align-items:flex-start}
#wrap{position:relative;cursor:crosshair;flex:none}
#img{display:block} #marks{position:absolute;inset:0;pointer-events:none}
aside{display:flex;flex-direction:column;gap:10px;min-width:330px}
#loupe{border:1px solid var(--line);background:#000}
.k{color:var(--dim)} kbd{background:#1c2330;border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px monospace}
button{background:#1c2330;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)} ul{margin:0;padding-left:18px;color:var(--dim)}
#state{font-weight:600;color:var(--sel)} .done{color:var(--ok)} .none{color:var(--no)}
#list{font:12px monospace;color:var(--dim);white-space:pre}
</style>
<header><b>Rim labeller &mdash; click the ring's LEFT edge, then its RIGHT edge</b>
<div id="bar"><div id="fill"></div></div>
<span><span id="idx">0</span>/<span id="tot">0</span></span>
<span><b id="ndone">0</b> rims on <b id="nframes">0</b> frames &middot; <b id="nnone">0</b> marked no&nbsp;rim</span>
<span class="k" id="meta"></span><button onclick="exportLabels()">Export rim_labels.json</button></header>
<main><div id="wrap"><img id="img"><svg id="marks"></svg></div>
<aside><div>This frame: <span id="state">&mdash;</span></div>
<canvas id="loupe" width="320" height="220"></canvas>
<div id="list"></div>
<ul>
<li>Click the <b>far-left OUTER edge of the orange metal</b>, then the <b>far-right OUTER edge</b> &mdash; the widest points of the ring itself, <b>not</b> the hole it encircles. The loupe shows 6&times; around the cursor.</li>
<li>Two labellers read an earlier wording differently: one clicked the ring, the other the opening inside it, and the widths came out 30% apart on close-ups and 100% apart on distant rims. Outer edge of the metal, every time.</li>
<li>Label <b>every</b> rim you can see, including one that is cut off by the frame edge &mdash; put the click where the ring's edge would be.</li>
<li>Mark the ring itself, <b>not the backboard and not the net</b>. If a player hides the ring, skip the frame.</li>
<li><kbd>x</kbd> no rim in this picture (that is useful too &mdash; it is a negative)</li>
<li><kbd>&rarr;</kbd>/<kbd>n</kbd> next &middot; <kbd>&larr;</kbd> previous &middot; <kbd>u</kbd> undo last click &middot; <kbd>d</kbd> clear this frame</li>
<li>Saved in this browser as you go. Export when done and put the file at <code>__OUT__</code>.</li>
</ul></aside></main>
<script>
const MANIFEST = __MANIFEST__;
const KEY = '__KEY__';
let i = 0, pending = null, L = {};
try { L = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) { L = {} }
const $ = id => document.getElementById(id);
const img = $('img'), marks = $('marks'), loupe = $('loupe').getContext('2d');
let SCALE = 1;
function frame() { return MANIFEST.frames[i] }
function rec() { const f = frame().file; return L[f] || (L[f] = { t: frame().t, none: false, rims: [] }) }
function persist() { try { localStorage.setItem(KEY, JSON.stringify(L)) } catch (e) {} }

function draw() {
  const r = rec(); marks.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';
  const dot = (x, y, c) => { const e = document.createElementNS(ns, 'circle');
    e.setAttribute('cx', x * SCALE); e.setAttribute('cy', y * SCALE); e.setAttribute('r', 4);
    e.setAttribute('fill', 'none'); e.setAttribute('stroke', c); e.setAttribute('stroke-width', 2); marks.appendChild(e) };
  const ell = (cx, cy, w, c) => { const e = document.createElementNS(ns, 'ellipse');
    e.setAttribute('cx', cx * SCALE); e.setAttribute('cy', cy * SCALE);
    e.setAttribute('rx', (w / 2) * SCALE); e.setAttribute('ry', (w * 0.225) * SCALE);
    e.setAttribute('fill', 'none'); e.setAttribute('stroke', c); e.setAttribute('stroke-width', 2); marks.appendChild(e) };
  r.rims.forEach(m => { ell(m.rim[0], m.rim[1], m.width, '#3fb950');
                        dot(m.rim[0] - m.width / 2, m.rim[1], '#3fb950');
                        dot(m.rim[0] + m.width / 2, m.rim[1], '#3fb950') });
  if (pending) dot(pending[0], pending[1], '#f0b400');
  $('state').textContent = r.none ? 'no rim' : (r.rims.length ? r.rims.length + ' rim(s)' : (pending ? 'left edge set — now the right' : '—'));
  $('state').className = r.none ? 'none' : (r.rims.length ? 'done' : '');
  $('list').textContent = r.rims.map((m, k) =>
    `${k}: centre ${m.rim[0].toFixed(0)},${m.rim[1].toFixed(0)}  width ${m.width.toFixed(0)}`).join('\\n');
  let rims = 0, frames = 0, none = 0;
  for (const k in L) { if (L[k].none) none++; if (L[k].rims.length) { frames++; rims += L[k].rims.length } }
  $('ndone').textContent = rims; $('nframes').textContent = frames; $('nnone').textContent = none;
  const seen = Object.keys(L).filter(k => L[k].none || L[k].rims.length).length;
  $('fill').style.width = (100 * seen / MANIFEST.frames.length) + '%';
  $('idx').textContent = i + 1; $('tot').textContent = MANIFEST.frames.length;
  $('meta').textContent = `${frame().file}  t=${frame().t.toFixed(1)}s`;
}

function show() {
  pending = null;
  img.onload = () => {
    const room = Math.min(window.innerWidth - 400, 1180);
    SCALE = Math.min(1, room / img.naturalWidth);
    img.width = img.naturalWidth * SCALE; img.height = img.naturalHeight * SCALE;
    marks.setAttribute('width', img.width); marks.setAttribute('height', img.height);
    draw();
  };
  img.src = 'images/' + frame().file;
}

function at(ev) {
  const b = img.getBoundingClientRect();
  return [(ev.clientX - b.left) / SCALE, (ev.clientY - b.top) / SCALE];
}
$('wrap').addEventListener('mousemove', ev => {
  const [x, y] = at(ev), z = 6, w = 320, h = 220;
  loupe.imageSmoothingEnabled = false;
  loupe.clearRect(0, 0, w, h);
  loupe.drawImage(img, x - w / (2 * z), y - h / (2 * z), w / z, h / z, 0, 0, w, h);
  loupe.strokeStyle = '#f0b400'; loupe.lineWidth = 1;
  loupe.beginPath(); loupe.moveTo(w / 2, 0); loupe.lineTo(w / 2, h);
  loupe.moveTo(0, h / 2); loupe.lineTo(w, h / 2); loupe.stroke();
});
$('wrap').addEventListener('click', ev => {
  const [x, y] = at(ev), r = rec();
  if (!pending) { pending = [x, y]; r.none = false; }
  else {
    const cx = (pending[0] + x) / 2, cy = (pending[1] + y) / 2;
    const width = Math.hypot(x - pending[0], y - pending[1]);
    if (width >= 6) r.rims.push({ rim: [cx, cy], width: width });
    pending = null;
  }
  persist(); draw();
});
document.addEventListener('keydown', ev => {
  const r = rec();
  if (ev.key === 'ArrowRight' || ev.key === 'n') { i = Math.min(i + 1, MANIFEST.frames.length - 1); show() }
  else if (ev.key === 'ArrowLeft') { i = Math.max(i - 1, 0); show() }
  else if (ev.key === 'u') { if (pending) pending = null; else r.rims.pop(); persist(); draw() }
  else if (ev.key === 'd') { r.rims = []; r.none = false; pending = null; persist(); draw() }
  else if (ev.key === 'x') { r.none = !r.none; r.rims = []; pending = null; persist(); draw() }
});
function exportLabels() {
  const frames = [];
  for (const file in L) {
    const r = L[file];
    for (const m of r.rims)
      frames.push({ t: r.t, rim: [Math.round(m.rim[0]), Math.round(m.rim[1])],
                    width: Math.round(m.width), source: 'hand', file: file });
    if (r.none && !r.rims.length) frames.push({ t: r.t, rim: null, width: 0, source: 'hand', file: file });
  }
  frames.sort((a, b) => a.t - b.t);
  const blob = new Blob([JSON.stringify(
    { note: 'Rims located by hand on close-up and alternate-camera frames, two clicks per ring (left edge, right edge). rim:null means no rim in the picture, which is a negative.',
      video: MANIFEST.video, frame_size: MANIFEST.size, frames: frames }, null, 1)],
    { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'rim_labels.json'; a.click();
}
show();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", required=True, help="mine_rim_frames.py frames.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--key", default="rimlabels_closeups",
                        help="localStorage key; change it to start a fresh pass")
    args = parser.parse_args()

    import cv2

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = json.load(open(args.frames))["frames"]
    capture = cv2.VideoCapture(args.video)
    size = None
    written = []
    for row in rows:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        size = [frame.shape[1], frame.shape[0]]
        name = f"t{row['t']:08.1f}.jpg"
        cv2.imwrite(str(out / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append({"file": name, "t": round(float(row["t"]), 2)})
    capture.release()

    manifest = {"video": args.video, "size": size, "frames": written}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "label.html").write_text(
        PAGE.replace("__MANIFEST__", json.dumps(manifest))
            .replace("__KEY__", args.key)
            .replace("__OUT__", str(out / "rim_labels.json")))
    print(f"{len(written)} frames written")
    print(f"  open {out / 'label.html'} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
