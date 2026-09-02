# Training data: what exists, what is reachable, what to use

Every claim here was verified by fetching the resource on 2026-09-02, not by
reading a paper. Where something is dead, the failure is recorded with the
error the server actually returned.

## The problem this is trying to solve

The action classifier trains on 1.6-second clips cut around play-by-play
events:

```
data/labeled/actions/{block,dribble,other,pass,rebound,shot,steal}
  400 / 400 / 795 / 400 / 436 / 800 / 436     = 3,667 clips
```

Every positive is an event cut from `nba.com/stats/events` video via BARD.
The only negatives are 795 `other` clips. At serving time the model sees
continuous broadcast that is roughly 90% ordinary play. That is the
train/serve mismatch, and it is a property of the *corpus*, not of the model
— no amount of retraining on this corpus fixes it.

So the requirement is not "more clips". It is **untrimmed footage whose
un-annotated time is usable as in-domain negatives**, ideally with boxes so
the crop lands on the right player.

## Tier 1 — reachable right now, no account

### BARD, in full (already the training source, ~10% used)

`GabrieleGiudici/BARD` on HuggingFace. Ungated, CC BY 4.0, 25,119 files,
last updated 2026-08-13. Organised per game (`bkn-vs-det-0022400861/2.mp4`),
and `dataset.csv` carries the NBA `GameID` **and** `GameEventID`, so every
clip joins to official play-by-play.

The finding that matters is local and measurable:

```
data/labeled/actions/*/*.mp4   1.60 s   <- what we train on
data/raw_clips/sample.mp4     10.42 s   <- the BARD clip it was cut from
data/raw_clips/holdout.mp4     8.85 s
```

**~80% of every BARD clip is unlabeled ordinary play that is currently
discarded.** Same broadcast, same game, same camera, seconds away from the
positive — which is exactly the distribution the negatives need to come from.
14,677 labeled events × ~8 s of surrounding context is on the order of 30
hours of in-domain negative footage, against the 795 `other` clips in use
today. This needs no new dataset and no permission.

Caveat: BARD has no `dribble` or `pass` label and no bounding boxes, which is
why it cannot carry the whole load on its own.

### SportsMOT — tracking, and continuous negatives

`MCG-NJU/SportsMOT` on HuggingFace. Ungated: the resolve URL redirects to a
public CDN (`user_id=public`), no token needed. Verified sizes:

| file | bytes |
|---|---|
| `dataset/train.tar` | 6,703,191,552 |
| `dataset/val.tar` | 6,592,876,544 |
| `dataset/test.tar` | 23,208,035,328 |

`splits_txt/basketball.txt` lists **80 basketball sequences** — continuous
720p/25fps broadcast with per-frame player boxes and persistent track ids.
CC BY-NC 4.0.

This is the direct answer to the 14,640 spurious track ids seen in one game:
it is supervision for detection and tracking specifically, which is the layer
possession is built on, and possession is what steal and rebound are derived
from.

### NCAA annotations — alive; NCAA video — effectively dead

The Stanford host still serves the annotations. Confirmed 200:

```
http://vision.stanford.edu/vigneshr_release_data/train_test_val_merged_detections_v2_ts_fixed.tgz
  58,828,421 bytes, Last-Modified Mon, 24 Oct 2016
```

Downloaded and characterised:

- **14,548 events across 257 games, 11 classes**, official train/test/val
  splits (11,436 / 2,256 / 856)
- clips are **median 32.8 s** with the event **median 1.25 s** inside them —
  i.e. the negatives-within-clips structure, already annotated, with
  `NOEVENT` an explicit label
- **4,045,698 player boxes** with persistent track ids
  (`person_0_00511477633`), stored normalised 0–1 so they survive
  re-downloading at a higher resolution than the 490×360 they were drawn on
- **2,379 `steal success`** events — 5.5× the 436 steal clips in use now

The video is the problem. `basketballattention.appspot.com` returns 404 and
the YouTube ids have been swept:

```
256 ids checked with yt-dlp 2026.08.19 -> 6 alive
error on the rest: "ERROR: [youtube] <id>: Private video"
```

Not rate-limiting — verified individually. The 6 survivors are real, distinct
games (1990–2010, 480p, ~75 min each) carrying **303 events including 48
steals**, with detections present for all 6. That is ~7.5 hours of untrimmed
broadcast with ground-truth event times — worth having as a *continuous-footage
evaluation set*, since the project currently has one. It is not a training set,
and 1990s 480p college footage is a real domain shift from modern NBA broadcast.

There is a 2020 mirror of all 112 GB on a ZJUT SharePoint and on Baidu.
The SharePoint link returns 200 but only an SPA shell; whether the files are
still behind it could not be settled programmatically.

## Tier 2 — one click, free HuggingFace account

### MultiSports — the best match for the classes that fail

`MCG-NJU/SportsAction` (the repo was renamed from `MultiSports`; the old path
307-redirects). `gated: auto`, so access is granted immediately on
acknowledging the CC BY-NC 4.0 licence — it needs a free HF account, which is
the only reason this is not Tier 1. The two ungated mirrors
(`lmwang/MultiSports`, `shreyansh-sh/MultiSports`) are empty shells.

`data/trainval/basketball.tar` + `multisports_GT.pkl`. The basketball
vocabulary is 18 classes:

> pass, drive, dribble, 3-point shot, 2-point shot, free throw, **block**,
> offensive rebound, defensive rebound, **pass steal**, **dribble steal**,
> interfere shot, pick-and-roll defensive, sag, screen, pass-inbound, save,
> jump ball

Annotation is 25 fps **action tubes** — per-frame boxes with precise temporal
boundaries inside longer clips, so background frames come for free.

Why this one matters more than its size suggests, against measurements already
in this repo:

- **block** measured **−0.170 separability from pixels, below chance.** The
  crop was being taken from a player box with no guarantee the shot-contest
  was in it. MultiSports gives block as a *tube*, and gives `interfere shot`
  as the labelled confusable next to it.
- **steal** measured +0.054 and hit 0.24 precision even with perfect
  perception. MultiSports splits it into `pass steal` and `dribble steal` —
  which is a plausible explanation for why one flat `steal` class was hard.
- `screen`, `sag`, `pick-and-roll defensive`, `drive` are labels for exactly
  the ordinary play that currently gets classified as an event.
- **rebound** splits offensive/defensive, matching how `derived_events`
  already reasons about it.

## Tier 3 — request form

**BASKET** (CVPR 2025, UNC). 4,477 hours, 32,232 players, 21 leagues, 20
skills. Access via a form on `yulupan/BASKET`; the authors reply with a link.
**~1.8 TB** — larger than the 131 GB free on this machine, and the task is
per-player skill *rating* from 8–10 minute highlight reels, not event
detection. Not a fit for this pipeline.

## Tier 4 — announced, not yet released

**BasketEvent** (arXiv 2607.21267, July 2026) is conceptually the closest
thing to what this project needs and does not exist yet in downloadable form:
**226 NBA games, 35,000 broadcast clips, 90.6 hours**, events derived from
official play-by-play and grounded to the responsible player, 10 classes
**plus an explicit Background class**, with 1,000 test samples carrying
manually annotated event intervals. The paper says data, code and models
"will be made publicly available"; no repository has appeared. Worth watching.

**Not found, still:** any public dataset pairing NBA broadcast video with
SportVU tracking for the same possession. The nearest equivalent is
**SoccerNet-GAR** (arXiv 2511.12606, `drishyakarki/pixels_vs_positions`),
which does exactly that synchronisation — 87,939 group activities, video and
tracking aligned — but for football. It is a template for how such a pairing
is built, not a basketball resource.

## Recommendation

1. **Mine negatives out of BARD.** Free, no permission, no download, and it
   attacks the measured failure directly. This should happen first because it
   is the only item here that costs nothing and is testable against the
   existing scorers.
2. **MultiSports basketball**, once someone clicks through the HF licence. It
   is the only source found with `block` as a spatio-temporal tube, and block
   is the one class currently below chance.
3. **SportsMOT** (13.3 GB train+val) to improve detection and tracking, since
   possession — and therefore steal and rebound — is built on track quality.
4. **The 6 surviving NCAA games** as a second continuous-footage evaluation
   set, kept clearly separate from training and reported as out-of-domain.

## Provenance and licensing

- BARD — CC BY 4.0, redistributable with attribution.
- MultiSports, SportsMOT — CC BY-NC 4.0. **Non-commercial.**
- NCAA — annotations released by Stanford; the videos are third-party
  broadcast. Use for research, do not redistribute.
- SportVU tracking already in this repo is unlicensed re-hosting of
  proprietary NBA data: build and validate against it, never ship it.
