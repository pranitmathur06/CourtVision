"""Check the CUDA kernel's arithmetic on the host, with no GPU.

`nvcc` does not exist for Apple Silicon, so torso_color.cu has never been
compiled and the runbook warns to "expect build errors first time". Finding a
typo or a swapped constant on rented hardware is the expensive way to find it.

The kernel is written so a single thread computes a whole box: the grid-stride
loop covers every pixel at stride 1, the tree reduction is skipped because
blockDim.x / 2 == 0, and thread 0 writes the mean it just accumulated. So with
blockDim.x = 1 the real source can be compiled by a host C++ compiler behind
CUDA stubs and checked against the same OpenCV oracle the GPU path uses.

What this proves: the colour maths, the sRGB->XYZ constants, the BGR channel
order, the OpenCV 8-bit Lab scaling, the pixel indexing, and the mean.
What it does NOT prove: the shared-memory tree reduction, launch configuration,
or anything about occupancy. Those still need real hardware.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

KERNEL = Path("src/courtvision/kernels/torso_color.cu")

STUBS = r"""
// ---- host-side CUDA stubs -------------------------------------------------
// threadIdx is thread_local and __syncthreads() is a real barrier, so the block
// runs with genuine concurrency. That exercises the shared-memory tree
// reduction as written, not a serial paraphrase of it.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#define __device__
#define __global__
#define __forceinline__ inline
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
static inline float __powf(float a, float b) { return powf(a, b); }
static std::vector<float> shared_storage;
#define extern_shared_decl float* shared = shared_storage.data();
"""

MAIN = r"""
// ---- driver ---------------------------------------------------------------
int main(int argc, char** argv) {
  int threads_per_block = argc > 1 ? atoi(argv[1]) : 1;
  int height = 0, width = 0, n = 0;
  if (scanf("%d %d %d", &height, &width, &n) != 3) return 2;
  std::vector<unsigned char> image(static_cast<size_t>(height) * width * 3);
  for (size_t i = 0; i < image.size(); ++i) {
    int v; if (scanf("%d", &v) != 1) return 2; image[i] = (unsigned char)v;
  }
  std::vector<int> boxes(static_cast<size_t>(n) * 4);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (scanf("%d", &boxes[i]) != 1) return 2;
  }
  std::vector<float> out(static_cast<size_t>(n) * 3, 0.0f);

  blockDim.x = threads_per_block;
  g_block_threads = threads_per_block;
  shared_storage.assign(static_cast<size_t>(threads_per_block) * 3, 0.0f);

  for (int b = 0; b < n; ++b) {
    blockIdx.x = b;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads_per_block; ++t) {
      pool.emplace_back([&, t] {
        threadIdx.x = t;
        torso_mean_lab_kernel(image.data(), boxes.data(), out.data(),
                              height, width);
      });
    }
    for (auto& th : pool) th.join();
  }
  for (int i = 0; i < n * 3; ++i) printf("%.6f\n", out[i]);
  return 0;
}
"""

def build_source() -> str:
    text = KERNEL.read_text()
    end = text.index("torch::Tensor torso_mean_lab(")
    body = text[text.index("namespace {"):end]
    # The host stub supplies `shared`; drop the CUDA dynamic-shared declaration.
    body = body.replace("extern __shared__ float shared[];  // 3 * blockDim.x",
                        "extern_shared_decl")
    if "extern_shared_decl" not in body:
        raise SystemExit("could not substitute the shared-memory declaration")
    return STUBS + body + MAIN


def oracle(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    import cv2

    means = []
    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        means.append(lab.reshape(-1, 3).mean(axis=0))
    return np.asarray(means, dtype=np.float64)


def main() -> int:
    if not KERNEL.exists():
        print(f"FAIL — no kernel at {KERNEL}")
        return 1
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        print("SKIP — no host C++ compiler found")
        return 0

    work = Path(tempfile.mkdtemp(prefix="kernel-check-"))
    source = work / "check.cpp"
    source.write_text(build_source())
    binary = work / "check"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-o", str(binary), str(source)],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        print("FAIL — the kernel does not compile as host C++:\n")
        print(build.stderr[:4000])
        return 1
    print("compiles clean under a host C++ compiler")

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 80, 64], [10, 5, 40, 30], [55, 40, 79, 63],
                      [3, 3, 4, 4]], dtype=np.int32)

    payload = [f"{image.shape[0]} {image.shape[1]} {len(boxes)}"]
    payload.append(" ".join(map(str, image.reshape(-1).tolist())))
    payload.append(" ".join(map(str, boxes.reshape(-1).tolist())))
    results = {}
    for threads in (1, 256):
        run = subprocess.run([str(binary), str(threads)], input="\n".join(payload),
                             capture_output=True, text=True)
        if run.returncode != 0:
            print(f"FAIL — driver exited {run.returncode} at {threads} threads\n"
                  f"{run.stderr[:2000]}")
            return 1
        results[threads] = np.array(
            [float(v) for v in run.stdout.split()]).reshape(-1, 3)
    got = results[256]
    want = oracle(image, boxes)

    # 1 thread skips the tree reduction entirely (blockDim.x / 2 == 0); 256
    # exercises every level of it. Agreement between them is the reduction
    # working, independently of whether either matches OpenCV.
    reduction_err = float(np.abs(results[1] - results[256]).max())
    print(f"  single-thread vs 256-thread block: max diff {reduction_err:.4f}")
    if reduction_err > 0.01:
        print("  FAIL — the shared-memory tree reduction does not agree with the")
        print("  serial path, so it is losing or double-counting contributions.")
        return 1
    print("  tree reduction agrees with the serial path\n")

    print(f"\n  {'box':<22}{'kernel (L,a,b)':<30}{'OpenCV (L,a,b)':<30}{'max err':>9}")
    worst = 0.0
    for box, g, w in zip(boxes, got, want):
        err = float(np.abs(g - w).max())
        worst = max(worst, err)
        label = "(" + ", ".join(str(int(v)) for v in box) + ")"
        print(f"  {label:<22}{str(np.round(g, 2).tolist()):<30}"
              f"{str(np.round(w, 2).tolist()):<30}{err:>9.3f}")

    # OpenCV's 8-bit Lab path quantises to uint8 per pixel before averaging;
    # the kernel averages in float. A small bias is expected, a large one is a
    # real disagreement about the colour transform.
    TOLERANCE = 1.5
    print(f"\n  worst channel error: {worst:.3f} (tolerance {TOLERANCE})")
    if worst > TOLERANCE:
        print("  FAIL — the kernel disagrees with OpenCV about the colour transform.")
        return 1
    print("  PASS — arithmetic, channel order and Lab scaling agree with OpenCV,")
    print("  and the tree reduction matches the serial path at 256 threads.")
    print("  Still needs real hardware: nvcc itself, the launch configuration,")
    print("  memory coalescing and occupancy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
