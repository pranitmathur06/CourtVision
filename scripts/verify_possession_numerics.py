"""Check the possession kernel's arithmetic, and its BACKWARD, with no GPU.

`nvcc` does not exist for Apple Silicon, so this kernel cannot be compiled here.
Finding a swapped index or a dropped chain-rule term on rented hardware is the
expensive way to find it, and a wrong backward is worse than a wrong forward: it
does not crash, it trains to something slightly wrong and looks fine.

So the real `possession.cu` is compiled by a host C++ compiler behind CUDA
stubs — `threadIdx` thread-local, `__syncthreads()` a genuine
condition_variable barrier, so the block runs with real concurrency and the
shared-memory tree reductions are exercised as written rather than as a serial
paraphrase. This is the same trick `verify_kernel_numerics.py` uses for
torso_color.cu, extended to cover gradients.

Five things are checked, three for each kernel:

  KERNEL 1, the per-frame operator
    FORWARD at 1 thread against the NumPy oracle        -- the arithmetic
    FORWARD at 64 threads against 1 thread              -- the reductions
    BACKWARD at 64 threads against torch autograd       -- the chain rule

  KERNEL 2, the forward-backward over time
    POSTERIOR AND LOSS at 1 and 64 threads vs the oracle -- the scan
    GRADIENTS at 64 threads against torch autograd       -- free minus clamped

The last is the point. The backward is hand-derived — through a softmax over
players, a tanh head, a smooth maximum over 64 samples, and a bilinear read of
the image whose spatial derivative moves the sampling region — and autograd
differentiating the portable torch path is an independent witness to it.

What this does NOT prove: launch configuration, occupancy, or anything about
memory bandwidth. Those still need real hardware.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

KERNEL = Path("src/courtvision/kernels/possession.cu")

STUBS = r"""
// ---- host-side CUDA stubs -------------------------------------------------
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
#define __shared__
struct Dim3 { int x, y, z; };
static Dim3 blockIdx{0, 0, 0};
static thread_local Dim3 threadIdx{0, 0, 0};
static Dim3 blockDim{1, 1, 1};

static int g_block_threads = 1;
static std::mutex g_mutex;
static std::condition_variable g_cv;
static int g_waiting = 0, g_generation = 0;
static inline void __syncthreads() {
  std::unique_lock<std::mutex> lock(g_mutex);
  const int gen = g_generation;
  if (++g_waiting == g_block_threads) {
    g_waiting = 0; ++g_generation; g_cv.notify_all();
  } else {
    g_cv.wait(lock, [gen] { return g_generation != gen; });
  }
}
// `min` and `max` on ints are CUDA built-ins with no host equivalent; without
// these the build fails with "use of undeclared identifier 'min'", which is
// this harness doing its job before a pod is ever rented.
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
// MACROS, NOT FUNCTIONS, AND AFTER EVERY SYSTEM HEADER. glibc's <cmath>
// already DECLARES `__expf` and `__logf` -- they are its own internal symbols --
// so `static inline float __expf(float)` is a redeclaration with different
// linkage and the build fails on Linux while succeeding on macOS, where those
// names are free. This harness exists to check the numerics without renting a
// GPU, and it could only do that on one operating system until CI said so.
// Function-like macros expand only at call sites, and every system header is
// already parsed above, so nothing but the kernel body is rewritten.
#ifdef __expf
#undef __expf
#endif
#define __expf(x) expf(x)
#ifdef __logf
#undef __logf
#endif
#define __logf(x) logf(x)
static std::vector<float> shared_storage;
#define extern_shared_decl float* shared = shared_storage.data();
"""

MAIN = r"""
// ---- driver ---------------------------------------------------------------
// stdin: height width players beta / image / boxes / ball / params / grad_probs
int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  int height = 0, width = 0, players = 0; float beta = 0.0f;
  if (scanf("%d %d %d %f", &height, &width, &players, &beta) != 4) return 2;
  std::vector<unsigned char> image((size_t)height * width * 3);
  for (size_t i = 0; i < image.size(); ++i) {
    int v; if (scanf("%d", &v) != 1) return 2; image[i] = (unsigned char)v;
  }
  auto read = [](int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) { if (scanf("%f", &v[i]) != 1) exit(2); }
    return v;
  };
  std::vector<float> boxes = read(players * 4);
  std::vector<float> ball = read(3);
  std::vector<float> first = read(kHidden * kFeatures);
  std::vector<float> first_bias = read(kHidden);
  std::vector<float> second = read(kHidden);
  std::vector<float> second_bias = read(1);
  std::vector<float> nobody = read(1);
  std::vector<float> region = read(4);
  std::vector<float> feature_mean = read(kFeatures);
  std::vector<float> feature_scale = read(kFeatures);
  std::vector<float> grad_probs = read(players + 1);

  std::vector<float> features((size_t)players * kFeatures, 0.0f);
  std::vector<float> probs(players + 1, 0.0f);

  blockDim.x = threads;
  g_block_threads = threads;
  shared_storage.assign((size_t)threads * 4 + players + 2, 0.0f);

  auto run = [&](void (*body)(int)) {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back([&, t] { threadIdx.x = t; body(t); });
    for (auto& th : pool) th.join();
  };

  {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        threadIdx.x = t;
        possession_forward_kernel(image.data(), boxes.data(), ball.data(),
                                  first.data(), first_bias.data(), second.data(),
                                  second_bias.data(), nobody.data(), region.data(),
                                  feature_mean.data(), feature_scale.data(),
                                  features.data(), probs.data(),
                                  height, width, players, beta);
      });
    }
    for (auto& th : pool) th.join();
  }

  std::vector<float> g_first(kHidden * kFeatures, 0.0f), g_first_bias(kHidden, 0.0f);
  std::vector<float> g_second(kHidden, 0.0f), g_second_bias(1, 0.0f);
  std::vector<float> g_nobody(1, 0.0f), g_region(4, 0.0f);
  shared_storage.assign((size_t)threads * 4 + players + 2, 0.0f);
  {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        threadIdx.x = t;
        possession_backward_kernel(image.data(), boxes.data(), first.data(),
                                   first_bias.data(), second.data(), region.data(),
                                   feature_mean.data(), feature_scale.data(),
                                   features.data(), probs.data(), grad_probs.data(),
                                   g_first.data(), g_first_bias.data(),
                                   g_second.data(), g_second_bias.data(),
                                   g_nobody.data(), g_region.data(),
                                   height, width, players, beta);
      });
    }
    for (auto& th : pool) th.join();
  }

  for (float v : features) printf("%.8f ", v);
  for (float v : probs) printf("%.8f ", v);
  for (float v : g_first) printf("%.8f ", v);
  for (float v : g_first_bias) printf("%.8f ", v);
  for (float v : g_second) printf("%.8f ", v);
  printf("%.8f %.8f ", g_second_bias[0], g_nobody[0]);
  for (float v : g_region) printf("%.8f ", v);
  printf("\n");
  return 0;
}
"""


TEMPORAL_MAIN = r"""
// ---- driver for the scan over time ----------------------------------------
// stdin: frames states target centre stay_raw / scores
int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  int frames = 0, states = 0, target = 0, centre = 0; float stay_raw = 0.0f;
  if (scanf("%d %d %d %d %f", &frames, &states, &target, &centre, &stay_raw) != 5)
    return 2;
  std::vector<float> scores((size_t)frames * states);
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scanf("%f", &scores[i]) != 1) return 2;
  }
  std::vector<float> posterior(states, 0.0f), loss(1, 0.0f);
  std::vector<float> grad_scores((size_t)frames * states, 0.0f), grad_stay(1, 0.0f);
  std::vector<float> stay_holder(1, stay_raw);

  blockDim.x = threads;
  g_block_threads = threads;
  shared_storage.assign((size_t)4 * frames * states + threads + 8, 0.0f);
  {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        threadIdx.x = t;
        temporal_kernel(scores.data(), stay_holder.data(), target, centre,
                        posterior.data(), loss.data(), grad_scores.data(),
                        grad_stay.data(), frames, states);
      });
    }
    for (auto& th : pool) th.join();
  }
  for (float v : posterior) printf("%.8f ", v);
  printf("%.8f ", loss[0]);
  for (float v : grad_scores) printf("%.8f ", v);
  printf("%.8f\n", grad_stay[0]);
  return 0;
}
"""


def kernel_body(text: str) -> str:
    """Every anonymous-namespace block in the real .cu, and nothing else.

    The file alternates device code with the torch wrappers that launch it,
    and the wrappers need headers this harness does not have. A device section
    runs from a `namespace {` to the first wrapper after it -- which is a wider
    net than the namespace's own closing brace, because the __global__ entry
    points sit outside the anonymous namespace so the launchers can see them.
    """
    body = []
    cursor = 0
    while True:
        start = text.find("namespace {", cursor)
        if start < 0:
            break
        end = text.find("std::vector<torch::Tensor> ", start)
        end = len(text) if end < 0 else end
        body.append(text[start:end])
        cursor = end + 1
    joined = "\n".join(body)
    return joined.replace("extern __shared__ float shared[];", "extern_shared_decl")


def build_source(text: str) -> str:
    """The per-frame kernel behind host stubs, with its driver."""
    return STUBS + kernel_body(text) + MAIN


def build_temporal_source(text: str) -> str:
    """The same device code, with the driver for the scan over time."""
    return STUBS + kernel_body(text) + TEMPORAL_MAIN


def main() -> int:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        print("SKIP - no host C++ compiler")
        return 0
    if not KERNEL.exists():
        print(f"FAIL - {KERNEL} missing")
        return 1

    from courtvision.kernels.possession import (BETA, FEATURES, HIDDEN,
                                                default_parameters,
                                                possession_reference)

    source = build_source(KERNEL.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "possession_host.cpp"
        path.write_text(source)
        binary = Path(tmp) / "possession_host"
        build = subprocess.run([compiler, "-std=c++17", "-O2", "-pthread",
                                str(path), "-o", str(binary)],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL - the kernel did not compile as host C++")
            print(build.stderr[-2500:])
            return 1

        rng = np.random.default_rng(11)
        height, width, players = 96, 160, 4
        image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        image[40:48, 60:68] = (20, 110, 240)          # something orange
        boxes = np.array([[50., 30., 78., 86.], [90., 28., 118., 84.],
                          [20., 34., 46., 88.], [120., 26., 148., 82.]])
        ball = np.array([64., 44., 0.9])
        parameters = default_parameters()
        # Non-unit scaling on purpose: with mean 0 and scale 1 a kernel that
        # forgot the standardisation entirely would still pass.
        parameters["feature_mean"] = rng.normal(0.0, 0.3, size=FEATURES)
        parameters["feature_scale"] = np.abs(rng.normal(1.0, 0.4, size=FEATURES)) + 0.2
        grad_probs = rng.normal(0.0, 1.0, size=players + 1)

        payload = [f"{height} {width} {players} {BETA}"]
        payload.append(" ".join(str(int(v)) for v in image.ravel()))
        for block in (boxes.ravel(), ball, parameters["first"].ravel(),
                      parameters["first_bias"], parameters["second"],
                      np.atleast_1d(parameters["second_bias"]),
                      np.atleast_1d(parameters["nobody"]),
                      parameters["region"], parameters["feature_mean"],
                      parameters["feature_scale"], grad_probs):
            payload.append(" ".join(f"{float(v):.10g}" for v in np.asarray(block).ravel()))
        stdin = "\n".join(payload) + "\n"

        outputs = {}
        for threads in (1, 64):
            run = subprocess.run([str(binary), str(threads)], input=stdin,
                                 capture_output=True, text=True)
            if run.returncode != 0:
                print(f"FAIL - the host build crashed at {threads} threads")
                print(run.stderr[-1500:])
                return 1
            outputs[threads] = np.array([float(v) for v in run.stdout.split()])

    n_feat = players * FEATURES
    n_prob = players + 1
    n_grad = HIDDEN * FEATURES + HIDDEN + HIDDEN + 1 + 1 + 4
    expected = n_feat + n_prob + n_grad
    if outputs[1].size != expected:
        print(f"FAIL - kernel produced {outputs[1].size} numbers, expected {expected}")
        return 1

    # 1. the reductions: one thread against sixty-four
    spread = np.abs(outputs[1] - outputs[64]).max()
    print(f"  1 thread vs 64 threads      max |diff| {spread:.3e}")

    # 2. the forward arithmetic, against the NumPy oracle
    reference_probs, reference_features = possession_reference(
        image, boxes, ball, parameters)
    got_features = outputs[64][:n_feat].reshape(players, FEATURES)
    got_probs = outputs[64][n_feat:n_feat + n_prob]
    feature_error = np.abs(got_features - reference_features).max()
    prob_error = np.abs(got_probs - reference_probs).max()
    print(f"  features vs NumPy oracle    max |diff| {feature_error:.3e}")
    print(f"  probabilities vs oracle     max |diff| {prob_error:.3e}")

    # 3. the backward, against autograd through the portable torch path
    import torch

    from courtvision.kernels.possession import possession_torch

    names = ["first", "first_bias", "second", "second_bias", "nobody", "region"]
    tensors = {n: torch.tensor(np.atleast_1d(parameters[n]).reshape(
        np.shape(parameters[n])), dtype=torch.float64, requires_grad=n in names)
        for n in parameters}
    probs_t, _ = possession_torch(image, boxes, ball, tensors)
    probs_t.backward(torch.tensor(grad_probs, dtype=torch.float64))
    offset = n_feat + n_prob
    sizes = [(HIDDEN * FEATURES, "first"), (HIDDEN, "first_bias"),
             (HIDDEN, "second"), (1, "second_bias"), (1, "nobody"), (4, "region")]
    worst, worst_name = 0.0, ""
    for size, name in sizes:
        got = outputs[64][offset:offset + size]
        offset += size
        want = tensors[name].grad.detach().numpy().ravel()
        error = np.abs(got - want).max()
        if error > worst:
            worst, worst_name = error, name
        print(f"    d/d{name:<12} max |diff| {error:.3e}")
    print(f"  backward vs autograd        worst {worst:.3e} on {worst_name}")

    # ---- kernel 2: the scan over time -------------------------------------
    from courtvision.kernels.possession import (temporal_reference,
                                                temporal_torch)

    print()
    temporal_worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "temporal_host.cpp"
        path.write_text(build_temporal_source(KERNEL.read_text()))
        binary = Path(tmp) / "temporal_host"
        build = subprocess.run([compiler, "-std=c++17", "-O2", "-pthread",
                                str(path), "-o", str(binary)],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL - the temporal kernel did not compile as host C++")
            print(build.stderr[-2500:])
            return 1

        rng = np.random.default_rng(29)
        frames, states = 7, 6
        centre = frames // 2
        # Scores with real spread, so a scan that dropped the transition term
        # entirely would not still land on the right answer by symmetry.
        scores = rng.normal(0.0, 1.8, size=(frames, states))
        stay_raw = float(rng.normal(0.4, 0.8))
        target = int(rng.integers(0, states))

        payload = [f"{frames} {states} {target} {centre} {stay_raw:.10g}",
                   " ".join(f"{v:.10g}" for v in scores.ravel())]
        stdin = "\n".join(payload) + "\n"
        temporal = {}
        for threads in (1, 64):
            run = subprocess.run([str(binary), str(threads)], input=stdin,
                                 capture_output=True, text=True)
            if run.returncode != 0:
                print(f"FAIL - the temporal kernel crashed at {threads} threads")
                print(run.stderr[-1500:])
                return 1
            temporal[threads] = np.array([float(v) for v in run.stdout.split()])

    want = states + 1 + frames * states + 1
    if temporal[1].size != want:
        print(f"FAIL - temporal kernel produced {temporal[1].size} numbers, "
              f"expected {want}")
        return 1
    scan_spread = np.abs(temporal[1] - temporal[64]).max()
    print(f"  temporal 1 vs 64 threads    max |diff| {scan_spread:.3e}")

    reference_post, reference_loss, reference_grads = temporal_reference(
        scores, stay_raw, target, centre)
    got = temporal[64]
    post_error = np.abs(got[:states] - reference_post).max()
    loss_error = abs(float(got[states]) - reference_loss)
    grad_error = np.abs(got[states + 1:states + 1 + frames * states]
                        .reshape(frames, states) - reference_grads["scores"]).max()
    stay_error = abs(float(got[-1]) - reference_grads["stay_raw"])
    print(f"  posterior vs NumPy oracle   max |diff| {post_error:.3e}")
    print(f"  loss vs NumPy oracle            |diff| {loss_error:.3e}")

    # The hand-derived free-minus-clamped gradient, against autograd through
    # the portable scan -- the independent witness, exactly as for kernel 1.
    scores_t = torch.tensor(scores, requires_grad=True)
    stay_t = torch.tensor(stay_raw, dtype=torch.float64, requires_grad=True)
    (-torch.log(temporal_torch(scores_t, stay_t, centre)[target])).backward()
    autograd_error = np.abs(
        got[states + 1:states + 1 + frames * states].reshape(frames, states)
        - scores_t.grad.numpy()).max()
    autograd_stay = abs(float(got[-1]) - float(stay_t.grad))
    print(f"    d/dscores  vs oracle {grad_error:.3e}   vs autograd {autograd_error:.3e}")
    print(f"    d/dstay    vs oracle {stay_error:.3e}   vs autograd {autograd_stay:.3e}")
    temporal_worst = max(post_error, loss_error, grad_error, stay_error,
                         autograd_error, autograd_stay)
    print(f"  temporal backward           worst {temporal_worst:.3e}")

    # float32 kernel against float64 oracles; 1e-4 is the honest bar
    ok = (spread < 1e-5 and feature_error < 1e-4 and prob_error < 1e-4
          and worst < 1e-4 and scan_spread < 1e-5 and temporal_worst < 1e-4)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
