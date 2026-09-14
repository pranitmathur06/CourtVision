"""A detection pass that spends its time on the GPU instead of on seeking.

MEASURED, on this video and this detector, before any of it was written:

    seek to each wanted frame   97.9 ms   <- what the pipeline does today
    sequential read + grab()     6.5 ms      15x faster
    inference (imgsz 1280)      44.1 ms

    today, seek + infer        142.0 ms/frame
    sequential + infer          50.6 ms/frame
    decode overlapped with GPU  44.1 ms/frame

So 69% of a detection pass is spent SEEKING, and the fix is not a faster model.
`cv2.VideoCapture.set(CAP_PROP_POS_MSEC)` re-seeks the container and decodes
forward to the requested instant every time; reading sequentially and throwing
away the frames between with `grab()` -- which demuxes without decoding -- gets
the same frames 15x cheaper.

Batching was tried first and REJECTED with numbers: through ultralytics on MPS,
batches of 8, 16 and 32 ran at 0.72x, 0.74x and 0.81x of one-at-a-time, because
its preprocessing and NMS are per-image Python either way and the batch only
adds latency. That is worth stating, because "just batch it" is the obvious
advice and here it is wrong.

What does help is overlapping: the decoder is a producer thread filling a
bounded queue while the GPU works, so decode time disappears behind inference
rather than adding to it. The queue is bounded because a 1280x720 frame is
2.6 MB and an unbounded one reads the whole game into memory.

HALF PRECISION is used where the device supports it -- CUDA does, MPS is left
alone because its fp16 path is not uniformly faster and this machine is where
the numbers above were taken. On a rented CUDA box `half=True` roughly halves
the inference term, at which point decode becomes the limit and the thread is
what keeps it hidden.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

#: A 1280x720 BGR frame is 2.6 MB. Sixty of them is 160 MB, which buys the GPU
#: plenty of runway without reading the game into memory.
QUEUE_DEPTH = 60


@dataclass(frozen=True)
class Decoded:
    """One frame and the instant it came from."""

    time_s: float
    frame: object


def frame_positions(start_s, end_s, step_s):
    """The instants a pass wants, as a plain sequence."""
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    # Counted, not accumulated. Adding step_s repeatedly drifts -- ten
    # additions of 0.2 land on 1.9999999999999998, which is still < 2.0 and
    # yields an eleventh frame nobody asked for.
    import math
    span = float(end_s) - float(start_s)
    count = max(int(math.ceil(span / float(step_s) - 1e-9)), 0)
    return [round(float(start_s) + i * float(step_s), 3) for i in range(count)]


def skip_count(fps, step_s):
    """How many frames to demux and discard between two wanted ones.

    Returns at least 1: a step shorter than the frame interval still advances,
    it just cannot give more frames than the video has.
    """
    return max(int(round(float(fps) * float(step_s))), 1)


def sequential_frames(video, start_s, end_s, step_s, capture=None):
    """Yield `Decoded` by reading FORWARD, never seeking per frame.

    Seeks once to `start_s`, then advances with grab(), which demuxes a frame
    without decoding it -- the whole reason this is 15x cheaper than asking for
    each instant by timestamp.
    """
    import cv2

    own = capture is None
    cap = capture if capture is not None else cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = skip_count(fps, step_s)
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
        for t in frame_positions(start_s, end_s, step_s):
            ok, frame = cap.read()
            if not ok:
                return
            yield Decoded(t, frame)
            for _ in range(step - 1):
                if not cap.grab():
                    return
    finally:
        if own:
            cap.release()


def decode_ahead(video, start_s, end_s, step_s, depth=QUEUE_DEPTH):
    """`sequential_frames` on a producer thread, so decode hides behind the GPU.

    Yields the same `Decoded` values in the same order. The thread is a daemon
    and the queue is bounded, so an abandoned consumer cannot leave it reading
    the rest of the game into memory.
    """
    box: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    failure = []

    def produce():
        try:
            for item in sequential_frames(video, start_s, end_s, step_s):
                box.put(item)
        except Exception as error:          # surfaced on the consumer's thread
            failure.append(error)
        finally:
            box.put(done)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    while True:
        item = box.get()
        if item is done:
            break
        yield item
    if failure:
        raise failure[0]


def half_precision_ok(device: str) -> bool:
    """Whether to ask for fp16 on this device.

    CUDA yes. MPS no -- its fp16 path is not uniformly faster and every number
    in this module's docstring was measured on MPS in fp32, so turning it on
    would make those numbers describe something else.
    """
    return str(device).startswith("cuda")
