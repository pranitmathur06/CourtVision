# Reproducing the live-labelled pipeline

The three games of live-labelled windows were generated on a rented pod and
**lost when it was destroyed** — the archive was written but the copy back
failed silently and only the checkpoint was retried. The trained models
survived; their training data did not. Everything below regenerates it from
scratch, and the order matters because each step needs the one before it.

## What survived

    outputs/final-state/ckpt_final.tgz   8-class action classifier, live-trained
    outputs/live-fix/ckpt.tgz            earlier 8-class checkpoint
    outputs/twoview/rim_ckpt.tgz         full-frame rim detector
    outputs/final-state/commentary.json  the scored full-game run, 3,538 events

The classifier loads and runs: classes are dribble, pass, shot, rebound, block,
steal, other, background.

## What it takes to regenerate

1. **Game video.** Any continuous broadcast file. `label_live_game.py` takes a
   path, so a purchased archive, a League Pass recording or an institutional
   dataset all work. Public download is currently gated behind bot detection,
   which is not something to route around.

2. **The game's id**, so the official play-by-play can be fetched. Ids are
   discoverable through `nba_api`; 2025 playoff games run `0042400RSG`.

3. **A scoreboard profile — or none at all.** `BROADCASTS` in
   `label_live_game.py` holds hand-built profiles for TNT and ESPN.
   `autoscoreboard.py` derives one automatically and should make that table
   unnecessary; it is unit-tested but has never run against a real broadcast,
   so verify it before trusting it on a new network.

```bash
python -m scripts.label_live_game --video GAME.mp4 --game-id 00424003XX \
    --crop-out data/live/actions --full-out data/live/rim --backgrounds 400
```

Expect 39-59% of official plays to align. **Copy `data/live` off the machine
before destroying it.**

4. **Merge and retrain.** Match real game frequencies rather than balancing —
   a steal happens 13 times a game, and giving it 451 clips taught the model
   steals are as common as shots.

5. **Score against the official record**, not against held-out clips:

```bash
python -m scripts.run_pipeline GAME.mp4 --out outputs/run --no-narrate \
    --no-render --prior-strength 1.0
python -m scripts.evaluate_live_game --events outputs/run/commentary.json \
    --game-id 00424003XX
```

## What to expect

Accuracy per class tracks live-labelled examples and little else:

    shot     254 examples -> 0.94x official
    rebound  149          -> 4.25x
    block     15          -> 0.36x
    steal     15          -> 33.8x

Roughly 250 examples per class is what "within 6%" cost for shot. At 13 steals a
game that is about twenty labelled games. Three is not enough, and no
reweighting, calibration or architecture change substituted for it — that was
tested repeatedly and is the single clearest result in this project.
