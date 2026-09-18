# CI

One workflow, `tests.yml`, running `pytest` on a clean Ubuntu machine.

**What it cannot check**, and why that is stated rather than worked around:

- **The CUDA kernels.** `verify_possession_numerics.py` needs a GPU. The host
  path is exercised by the suite; the device path is verified by hand on a
  rented A6000 and the result lives in `docs/v8-possession-kernels.md`.
- **Anything that decodes a broadcast.** The videos are 1 to 5 GB and are not in
  the repository.
- **Anything that reaches `stats.nba.com`.** `data/pbp_cache/` exists precisely
  so the aligner does not need it, and the tests use the cache.

The job prints the collected count beside the passed count on every run, because
the failure mode of a suite full of `importorskip` is that it goes green by
running almost nothing.
