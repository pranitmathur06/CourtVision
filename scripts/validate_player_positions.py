"""Do the registrations put PLAYERS somewhere a basketball player can be?

MEASURED, AND IT CAUGHT A REAL FAILURE. On a run whose frames all passed the
line-evidence gate (median 0.346-0.373, above the 0.30 accept threshold):

    on court (+-15 ft margin)   82.8%
    x range                     -27..76 ft   (the court is 0..50)
    displacements above 25 ft/s 29.5%        (a sprint peaks near 20-22)

So a third of the accepted registrations place players impossibly, and the line
score cannot see it. Reporting coverage and line evidence alone would have
looked like progress while producing unusable court coordinates.

Caveat on the speed figure: players are matched to the NEAREST player in the
previous frame with no identity tracking, so identity swaps inflate it. That
does not explain a p99 of 225 ft/s or players 27 ft off the side.

Line evidence says the model's lines land on the image's lines. It does not say
the homography is usable, because a court that has slid sideways still explains
the lines -- court_lines records a line-only fit that put the rim 174 px out
while still scoring well. Two checks the line score cannot make:

  1. players must be ON the court (94 x 50 ft, with a margin for the bench);
  2. between frames a third of a second apart they must move at human speed.

Neither uses court lines, so both are independent of what the search optimised.
"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0,"src")
from ultralytics import YOLO

SP=os.path.dirname(os.path.abspath(__file__))
path=sys.argv[1] if len(sys.argv)>1 else f"{SP}/reg6.json"
data=json.load(open(path))
mats={int(k):np.array(v) for k,v in data["matrices"].items()}
stride=data["stride"]; start_s=data["start_s"]; fps=data["fps"]
print(f"  {len(mats)} registered frames from {os.path.basename(path)}")

cap=cv2.VideoCapture("data/raw_clips/fullgame.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_s*fps))
n=max(mats)+1; images=[]
while len(images)<n:
    ok,f=cap.read()
    if not ok: break
    for _ in range(stride-1): cap.grab()
    images.append(f)
det=YOLO("yolo11n.pt")

MARGIN=15.0           # bench and photographers sit off the floor
positions={}
for i in sorted(mats):
    if i>=len(images): continue
    r=det(images[i],verbose=False,conf=0.35,classes=[0])[0]
    if not len(r.boxes): continue
    b=r.boxes.xyxy.cpu().numpy()
    feet=np.stack([(b[:,0]+b[:,2])/2, b[:,3]],axis=1)
    h=np.hstack([feet,np.ones((len(feet),1))]) @ mats[i].T
    w=h[:,2]; ok=np.abs(w)>1e-9
    pos=h[ok,:2]/w[ok,None]
    positions[i]=pos

allpos=np.vstack([p for p in positions.values() if len(p)])
inside=((allpos[:,0]>=-MARGIN)&(allpos[:,0]<=50+MARGIN)
        &(allpos[:,1]>=-MARGIN)&(allpos[:,1]<=94+MARGIN))
print(f"\n  {len(allpos)} player positions")
print(f"  on court (+-{MARGIN:.0f} ft margin): {inside.mean():.1%}")
print(f"  x range {np.percentile(allpos[:,0],2):.0f}..{np.percentile(allpos[:,0],98):.0f} ft"
      f"   (court is 0..50)")
print(f"  y range {np.percentile(allpos[:,1],2):.0f}..{np.percentile(allpos[:,1],98):.0f} ft"
      f"   (court is 0..94)")

# speed: nearest-neighbour displacement between consecutive registered frames
dt=stride/fps
speeds=[]
keys=sorted(positions)
for a,b in zip(keys,keys[1:]):
    if b-a!=1 or not len(positions[a]) or not len(positions[b]): continue
    for p in positions[b]:
        d=np.hypot(*(positions[a]-p).T)
        speeds.append(d.min()/dt)
speeds=np.array(speeds)
if len(speeds):
    print(f"\n  {len(speeds)} frame-to-frame displacements, dt={dt:.2f}s")
    print(f"    median {np.median(speeds):.1f} ft/s   p90 {np.percentile(speeds,90):.1f}"
          f"   p99 {np.percentile(speeds,99):.1f}")
    print(f"    above 25 ft/s (impossible): {(speeds>25).mean():.1%}")
    print("    a sprinting NBA player peaks near 20-22 ft/s")
