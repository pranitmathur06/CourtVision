"""Build a pick / not-pick dataset from SpaceJam's own annotations.

SpaceJam labels 712 clips as `pick` -- a human watched each one and said a
screen was being set. This project mapped that class to "other" and threw it
away, and then spent a long time concluding that no screen labels existed.
They existed.

Two things matter in the construction.

SpaceJam ships each clip twice, as `<id>` and `<id>_flipped`, which are mirror
images of the same footage. Splitting by clip would put a near-duplicate of a
training example in the test set, so the split is by BASE id.

And the negatives are chosen to be hard. A classifier that separates picks from
`walk` has learned nothing useful; the confusable classes are the ones a pick
looks like -- defence, standing with the ball, and no_action -- so the negative
pool is drawn from those in proportion.
"""
import zipfile, ast, collections, random, json, sys
from pathlib import Path

ARCHIVE = Path("data/labeled/spacejam/spacejam-action-recognition.zip")
OUT = Path("data/labeled/picks")
INDEX = Path("data/labeled/picks/index.json")
PICK = 7
# Everything a set screen might be confused with: a player standing still, a
# player defending, a player holding the ball. `walk` and `run` are included
# but capped, because a detector that only beats those is not doing anything.
NEGATIVE_MIX = {6: 0.30, 8: 0.30, 5: 0.15, 2: 0.10, 9: 0.10, 3: 0.05}


def base_of(clip_id: str) -> str:
    return clip_id.replace("_flipped", "")


def main() -> int:
    archive = zipfile.ZipFile(ARCHIVE)
    annotations = ast.literal_eval(archive.read("annotation_dict.json").decode())
    by_class = collections.defaultdict(list)
    for clip, cls in annotations.items():
        by_class[cls].append(clip)

    picks = sorted(by_class[PICK])
    rng = random.Random(0)
    wanted = len(picks)
    negatives = []
    for cls, share in NEGATIVE_MIX.items():
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        negatives += pool[:int(round(wanted * share))]
    rng.shuffle(negatives)
    negatives = negatives[:wanted]

    # Split by BASE id so a clip and its mirror never straddle the split.
    bases = sorted({base_of(c) for c in picks + negatives})
    rng.shuffle(bases)
    cut_a = int(0.70 * len(bases))
    cut_b = int(0.85 * len(bases))
    split = {}
    for i, base in enumerate(bases):
        split[base] = "train" if i < cut_a else ("val" if i < cut_b else "test")

    rows = []
    for clip in picks:
        rows.append(dict(clip=clip, label=1, split=split[base_of(clip)]))
    for clip in negatives:
        rows.append(dict(clip=clip, label=0, split=split[base_of(clip)]))

    OUT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = f"examples/{row['clip']}.mp4"
        target = OUT / row["split"] / ("pick" if row["label"] else "not_pick")
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"{row['clip']}.mp4"
        if not out.exists():
            out.write_bytes(archive.read(name))
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(INDEX, "w"))

    tally = collections.Counter((r["split"], r["label"]) for r in rows)
    print(f"  {len(picks)} picks, {len(negatives)} negatives, "
          f"{len(bases)} distinct source clips")
    for s in ("train", "val", "test"):
        print(f"    {s:<6} pick {tally[(s,1)]:>4}   not pick {tally[(s,0)]:>4}")
    overlap = {base_of(r["clip"]) for r in rows if r["split"] == "train"} & \
              {base_of(r["clip"]) for r in rows if r["split"] == "test"}
    print(f"  source clips appearing in both train and test: {len(overlap)}")
    return 0


sys.exit(main())
