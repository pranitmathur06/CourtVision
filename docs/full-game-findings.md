# What 84 minutes of continuous footage found

Everything before this was validated on clips of eight to ten seconds. This run
processed **50,304 frames** — 570x the reference clip — and the gap it exposed is
the most important result in the project.

Footage: 268 broadcast clips from one game (chi-vs-tor-0022401223), concatenated.
Real NBA video, but with hard cuts between plays rather than one continuous feed.
That matters for the tracking finding below and not for the others.

## The pipeline scales. The model does not transfer.

    1-3 extract/detect/track   1095.6s  38.3%
    4   team assignment         419.1s  14.6%
    5   possession               12.7s   0.4%
    6   action classification   512.3s  17.9%
    9   render                  822.5s  28.7%
    TOTAL                      2862.2s          47.7 min for 84 min of video

**1.76x real time on one RTX 4090**, with memory flat in clip length. That half of
the test passed cleanly.

## Finding 1: the classifier is calibrated for clips, not for games

3,383 events over 84 minutes — about 40 a minute, where real basketball produces
four to eight.

    action     emitted/min   realistic/min   ratio
    rebound           26.9             1.8   15x too many
    steal              3.4             0.3   11x too many
    shot               1.6             3.5   half as many
    pass               0.2             9.6   48x too FEW

Rebound alone is 2,253 of 3,383 events — 67% of everything the system reported.

The cause is a distribution mismatch, not a bad model. Training clips are a
roughly balanced mix of seven action classes, because they were cut to contain an
action. A real game is mostly ordinary play with no notable action at all. The
classifier has never seen that prior, so every window is forced into an action
class and rebound — the class whose visual signature is "several players near a
loose ball" — absorbs the slack.

0.78 held-out accuracy on rebound clips and 15x over-prediction on a game are
both true at once. That is what a distribution shift looks like.

The fix is not more training on the same data. It needs either a genuine
background class sampled from ordinary play, or a confidence floor below which a
window emits nothing, calibrated against a game rather than against clips.

## Finding 2: cuts fragment tracking, and events inherit the damage

**8,602 track ids**, against 19 on a short clip. ByteTrack has no re-identification
by design, so every one of the 268 cuts restarts identity for all ten players.

That propagates: `build_events` collapses consecutive windows sharing an action
and a holder, but consecutive windows rarely share a holder id when ids churn, so
the de-duplication does almost nothing. The median gap between reported events is
0.80 s — exactly the window stride, meaning nearly every window emitted an event.
2,007 distinct track ids appear in the event log.

A continuous feed would fragment far less than this concatenation does, so treat
8,602 as a worst case. The mechanism is real regardless: possession-level identity
needs to survive a camera cut, and today it does not.

## Finding 3: the render is impractical at length

4.8 GB for 84 minutes, written frame by frame with `mp4v`. Fine for a ten-second
clip, unusable as a deliverable. Segment extraction or a modern codec is needed
before anyone would watch this.

## What this changes

The honest claim is now narrower and better evidenced: **the pipeline runs on
game-length video, and the action model is not yet usable on it.** Everything
measured on clips remains true on clips.
