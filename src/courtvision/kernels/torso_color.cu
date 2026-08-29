// v2 §7.1 — fused torso-crop -> CIELAB -> mean colour.
//
// Chosen by profiling, not intuition. docs/profile-v1.md found detection and
// action classification dominate runtime, but both are standard forward passes
// already running vendor-tuned cuDNN kernels; hand-written code will not beat
// them. Team assignment is the largest stage with NO optimised implementation
// behind it — plain NumPy and OpenCV, one Python call per player per frame.
//
// The win here is not arithmetic, it is memory traffic and launch overhead. The
// unfused version materialises a crop, writes it, reads it back for the colour
// conversion, writes Lab pixels, reads them again to reduce. This kernel reads
// each pixel exactly once, converts in registers, and reduces in shared memory.
// Nothing intermediate is ever written to global memory.
//
// One block per (frame, player) pair; threads stride over the torso pixels.
//
// Build: see load_fused_kernel() in torso_color.py, which compiles this with
// torch.utils.cpp_extension.load_inline at first use.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// sRGB -> XYZ (D65), matching OpenCV.
__device__ __forceinline__ float srgb_to_linear(float c) {
  return c > 0.04045f ? __powf((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
}

__device__ __forceinline__ float lab_f(float t) {
  const float eps = 216.0f / 24389.0f;
  const float kappa = 24389.0f / 27.0f;
  return t > eps ? cbrtf(t) : (kappa * t + 16.0f) / 116.0f;
}

}  // namespace

// image:  (H, W, 3) uint8, BGR, contiguous
// boxes:  (N, 4) int32, [x1, y1, x2, y2] already clamped to the image
// out:    (N, 3) float32, OpenCV-scaled 8-bit Lab means
__global__ void torso_mean_lab_kernel(
    const unsigned char* __restrict__ image,
    const int* __restrict__ boxes,
    float* __restrict__ out,
    int height, int width) {
  const int box = blockIdx.x;
  const int x1 = boxes[box * 4 + 0];
  const int y1 = boxes[box * 4 + 1];
  const int x2 = boxes[box * 4 + 2];
  const int y2 = boxes[box * 4 + 3];

  const int box_w = x2 - x1;
  const int box_h = y2 - y1;
  const int count = box_w * box_h;

  extern __shared__ float shared[];  // 3 * blockDim.x
  float sum_l = 0.0f, sum_a = 0.0f, sum_b = 0.0f;

  if (count > 0) {
    // Grid-stride over the crop. Each pixel is read exactly once, from global
    // memory, and never written back in any intermediate form.
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
      const int py = y1 + i / box_w;
      const int px = x1 + i % box_w;
      const int base = (py * width + px) * 3;

      // Stored BGR; convert to RGB on the fly.
      const float b_in = image[base + 0] / 255.0f;
      const float g_in = image[base + 1] / 255.0f;
      const float r_in = image[base + 2] / 255.0f;

      const float r = srgb_to_linear(r_in);
      const float g = srgb_to_linear(g_in);
      const float b = srgb_to_linear(b_in);

      const float X = 0.412453f * r + 0.357580f * g + 0.180423f * b;
      const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
      const float Z = 0.019334f * r + 0.119193f * g + 0.950227f * b;

      const float fx = lab_f(X / 0.950456f);
      const float fy = lab_f(Y);
      const float fz = lab_f(Z / 1.088754f);

      // OpenCV 8-bit Lab: L scaled by 255/100, a and b offset by 128.
      sum_l += (116.0f * fy - 16.0f) * (255.0f / 100.0f);
      sum_a += 500.0f * (fx - fy) + 128.0f;
      sum_b += 200.0f * (fy - fz) + 128.0f;
    }
  }

  shared[threadIdx.x * 3 + 0] = sum_l;
  shared[threadIdx.x * 3 + 1] = sum_a;
  shared[threadIdx.x * 3 + 2] = sum_b;
  __syncthreads();

  // Tree reduction in shared memory.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x * 3 + 0] += shared[(threadIdx.x + stride) * 3 + 0];
      shared[threadIdx.x * 3 + 1] += shared[(threadIdx.x + stride) * 3 + 1];
      shared[threadIdx.x * 3 + 2] += shared[(threadIdx.x + stride) * 3 + 2];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float inv = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;
    out[box * 3 + 0] = shared[0] * inv;
    out[box * 3 + 1] = shared[1] * inv;
    out[box * 3 + 2] = shared[2] * inv;
  }
}

torch::Tensor torso_mean_lab(torch::Tensor image, torch::Tensor boxes) {
  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
  TORCH_CHECK(image.dtype() == torch::kUInt8, "image must be uint8 BGR");
  TORCH_CHECK(boxes.dtype() == torch::kInt32, "boxes must be int32");
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "image must be (H, W, 3)");
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N, 4)");

  image = image.contiguous();
  boxes = boxes.contiguous();
  const int n = boxes.size(0);
  auto out = torch::zeros({n, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));
  if (n == 0) return out;

  const int threads = 256;
  const size_t shared_bytes = threads * 3 * sizeof(float);
  torso_mean_lab_kernel<<<n, threads, shared_bytes>>>(
      image.data_ptr<unsigned char>(),
      boxes.data_ptr<int>(),
      out.data_ptr<float>(),
      static_cast<int>(image.size(0)),
      static_cast<int>(image.size(1)));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torso_mean_lab", &torso_mean_lab,
        "Fused torso crop, BGR->CIELAB and per-box mean (CUDA)");
}
