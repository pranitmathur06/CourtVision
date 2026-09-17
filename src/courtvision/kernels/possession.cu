// Fused possession attention: pixel sweep -> features -> head -> softmax,
// forward and backward, one block per frame.
//
// WHY FUSE. Unfused, one frame is: gather P crops, convert each to a
// chromaticity field, bilinearly sample a G x G grid in each, reduce twice per
// player (a smooth maximum and a mean), pad the ragged player count to a
// maximum, mask, build features, run a tiny matmul, and softmax over the
// players. That is seven passes over data that fits in registers, six of them
// writing an intermediate to global memory that the next one reads back, and
// every one paying the padding waste of the widest frame in the batch. Player
// counts here run 4 to 12.
//
// This does it in one pass. A block owns a frame; the players are walked in
// turn and the block's threads cooperate on that player's 64 samples with a
// shared-memory tree reduction. Nothing between the image and the output
// probabilities is ever written to global memory except what the backward pass
// genuinely needs.
//
// WHAT IS DELIBERATELY NOT HERE. No warp intrinsics (__shfl_down_sync and
// friends) and no atomics. That is not an oversight: scripts/
// verify_possession_numerics.py compiles this same file as HOST C++ with
// __global__/__device__/__shared__ stubbed out, threadIdx thread-local, and
// __syncthreads() as a real condition_variable barrier, so that the kernel's
// arithmetic can be tested on a machine with no GPU at all. Warp intrinsics
// have no host equivalent and would put the numerics beyond the reach of that
// harness, which is where every bug in the one existing kernel was caught.
//
// Build: load_fused_kernel() in possession.py compiles this with
// torch.utils.cpp_extension.load_inline at first use.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kGrid = 8;              // samples per side of the sweep
constexpr int kSamples = kGrid * kGrid;
constexpr int kFeatures = 7;
constexpr int kHidden = 8;
constexpr float kEps = 1e-6f;
// Radius of the surround ring as a fraction of the player's box height. A
// basketball is ~0.13 of a body height. Orange-ness alone cannot tell a ball
// from the floor it bounces on -- measured, hardwood reads 0.099 and the crowd
// 0.037 -- so the sweep scores centre MINUS surround, which is near zero on
// uniform wood and large on a compact ball.
constexpr float kSurround = 0.13f;

// How orange a BGR pixel is: (r - (g+b)/2) / (r+g+b). Scale-free, so it does
// not drift with the arena's exposure the way a raw channel would.
__device__ __forceinline__ float chroma_at(const unsigned char* image,
                                           int width, int y, int x) {
  const int base = (y * width + x) * 3;
  const float blue = image[base];
  const float green = image[base + 1];
  const float red = image[base + 2];
  return (red - 0.5f * (green + blue)) / (red + green + blue + kEps);
}

// Bilinear read of the chromaticity field, clamped at the edges. Returns the
// value and, when asked, the spatial derivatives -- the backward pass needs
// d(chroma)/dx and d(chroma)/dy to move the sampling region.
__device__ __forceinline__ float sample_chroma(const unsigned char* image,
                                               int height, int width,
                                               float x, float y,
                                               float* dx_out, float* dy_out) {
  x = fminf(fmaxf(x, 0.0f), width - 1.0f - 1e-6f);
  y = fminf(fmaxf(y, 0.0f), height - 1.0f - 1e-6f);
  const int x0 = (int)floorf(x);
  const int y0 = (int)floorf(y);
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const float fx = x - x0;
  const float fy = y - y0;
  const float c00 = chroma_at(image, width, y0, x0);
  const float c01 = chroma_at(image, width, y0, x1);
  const float c10 = chroma_at(image, width, y1, x0);
  const float c11 = chroma_at(image, width, y1, x1);
  if (dx_out) {
    *dx_out = (1.0f - fy) * (c01 - c00) + fy * (c11 - c10);
    *dy_out = (1.0f - fx) * (c10 - c00) + fx * (c11 - c01);
  }
  return (1.0f - fx) * (1.0f - fy) * c00 + fx * (1.0f - fy) * c01
       + (1.0f - fx) * fy * c10 + fx * fy * c11;
}

// Centre minus the mean of four points a ball-radius away, and its spatial
// derivative -- which is what carries the gradient back to the sampling region.
__device__ __forceinline__ float ball_response(const unsigned char* image,
                                               int height, int width,
                                               float x, float y, float radius,
                                               float* dx_out, float* dy_out) {
  float cdx = 0.0f, cdy = 0.0f;
  const float centre = sample_chroma(image, height, width, x, y,
                                     dx_out ? &cdx : nullptr,
                                     dx_out ? &cdy : nullptr);
  float ring = 0.0f, rdx = 0.0f, rdy = 0.0f;
  const float offsets[4][2] = {{-radius, 0.0f}, {radius, 0.0f},
                               {0.0f, -radius}, {0.0f, radius}};
  for (int k = 0; k < 4; ++k) {
    float gx = 0.0f, gy = 0.0f;
    ring += sample_chroma(image, height, width, x + offsets[k][0],
                          y + offsets[k][1], dx_out ? &gx : nullptr,
                          dx_out ? &gy : nullptr);
    rdx += gx;
    rdy += gy;
  }
  if (dx_out) {
    *dx_out = cdx - 0.25f * rdx;
    *dy_out = cdy - 0.25f * rdy;
  }
  return centre - 0.25f * ring;
}

// Where sample (u, v) of player p lands, given the learned region fractions.
__device__ __forceinline__ void sample_position(const float* box,
                                                const float* region,
                                                int u, int v,
                                                float* x, float* y,
                                                float* dx_dregion0,
                                                float* dx_dregion1,
                                                float* dy_dregion2,
                                                float* dy_dregion3) {
  const float bw = box[2] - box[0];
  const float bh = box[3] - box[1];
  const float su = (u + 0.5f) / kGrid;
  const float sv = (v + 0.5f) / kGrid;
  *x = box[0] + bw * (region[0] + (region[1] - region[0]) * su);
  *y = box[1] + bh * (region[2] + (region[3] - region[2]) * sv);
  if (dx_dregion0) {
    *dx_dregion0 = bw * (1.0f - su);
    *dx_dregion1 = bw * su;
    *dy_dregion2 = bh * (1.0f - sv);
    *dy_dregion3 = bh * sv;
  }
}

}  // namespace

// ---------------------------------------------------------------- forward --
//
// image    (H, W, 3) uint8 BGR, contiguous
// boxes    (P, 4) float32
// ball     (3,) float32  x, y, confidence; confidence 0 zeroes the geometry
// weights  first (HIDDEN, FEATURES), first_bias (HIDDEN), second (HIDDEN),
//          second_bias (1), nobody (1), region (4)
// features (P, FEATURES) float32   out, kept for the backward pass
// probs    (P + 1,) float32        out
__global__ void possession_forward_kernel(
    const unsigned char* __restrict__ image,
    const float* __restrict__ boxes,
    const float* __restrict__ ball,
    const float* __restrict__ first,
    const float* __restrict__ first_bias,
    const float* __restrict__ second,
    const float* __restrict__ second_bias,
    const float* __restrict__ nobody,
    const float* __restrict__ region,
    const float* __restrict__ feature_mean,
    const float* __restrict__ feature_scale,
    float* __restrict__ features,
    float* __restrict__ probs,
    int height, int width, int players, float beta) {
  extern __shared__ float shared[];
  float* reduce_exp = shared;                       // blockDim.x
  float* reduce_sum = shared + blockDim.x;          // blockDim.x
  float* scores = shared + 2 * blockDim.x;          // players + 1

  for (int p = 0; p < players; ++p) {
    const float* box = boxes + p * 4;
    float local_exp = 0.0f;
    float local_sum = 0.0f;
    for (int s = threadIdx.x; s < kSamples; s += blockDim.x) {
      float x, y;
      sample_position(box, region, s % kGrid, s / kGrid, &x, &y,
                      nullptr, nullptr, nullptr, nullptr);
      const float value = ball_response(image, height, width, x, y,
                                        kSurround * fmaxf(box[3] - box[1], kEps),
                                        nullptr, nullptr);
      local_exp += __expf(beta * value);
      local_sum += value;
    }
    reduce_exp[threadIdx.x] = local_exp;
    reduce_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_exp[threadIdx.x] += reduce_exp[threadIdx.x + stride];
        reduce_sum[threadIdx.x] += reduce_sum[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float* row = features + p * kFeatures;
      // Geometry, in body heights so it does not change meaning as a player
      // moves up the floor and gets smaller on screen.
      if (ball[2] > 0.0f) {
        const float body = fmaxf(box[3] - box[1], kEps);
        const float dx = (ball[0] - 0.5f * (box[0] + box[2])) / body;
        const float dy = (ball[1] - 0.5f * (box[1] + box[3])) / body;
        row[0] = dx;
        row[1] = dy;
        row[2] = sqrtf(dx * dx + dy * dy);
        row[3] = (ball[0] >= box[0] && ball[0] <= box[2] &&
                  ball[1] >= box[1] && ball[1] <= box[3]) ? 1.0f : 0.0f;
      } else {
        row[0] = row[1] = row[2] = row[3] = 0.0f;
      }
      // Smooth maximum over the sweep, and its mean.
      row[4] = __logf(reduce_exp[0] / kSamples) / beta;
      row[5] = reduce_sum[0] / kSamples;
      row[6] = ball[2];

      // Centre and scale before the head. Without this the geometry (+/-2)
      // saturates the tanh and the centre-surround response (~0.01) never
      // reaches it, which cost 10 points of accuracy when it was missing.
      float standard[kFeatures];
      for (int f = 0; f < kFeatures; ++f) {
        standard[f] = (row[f] - feature_mean[f]) / fmaxf(feature_scale[f], kEps);
      }
      float score = second_bias[0];
      for (int h = 0; h < kHidden; ++h) {
        float pre = first_bias[h];
        for (int f = 0; f < kFeatures; ++f) {
          pre += first[h * kFeatures + f] * standard[f];
        }
        score += second[h] * tanhf(pre);
      }
      scores[p] = score;
    }
    __syncthreads();
  }

  // Softmax over the players plus a learned "nobody has it" logit.
  if (threadIdx.x == 0) {
    scores[players] = nobody[0];
    float top = scores[0];
    for (int i = 1; i <= players; ++i) top = fmaxf(top, scores[i]);
    float total = 0.0f;
    for (int i = 0; i <= players; ++i) {
      probs[i] = __expf(scores[i] - top);
      total += probs[i];
    }
    for (int i = 0; i <= players; ++i) probs[i] /= total;
  }
}

// --------------------------------------------------------------- backward --
//
// Given dL/dprobs, produce gradients for every parameter including the region
// the sweep reads from -- which is where the spatial derivative of the image
// enters, through the bilinear sample and the smooth maximum's softmax.
__global__ void possession_backward_kernel(
    const unsigned char* __restrict__ image,
    const float* __restrict__ boxes,
    const float* __restrict__ first,
    const float* __restrict__ first_bias,
    const float* __restrict__ second,
    const float* __restrict__ region,
    const float* __restrict__ feature_mean,
    const float* __restrict__ feature_scale,
    const float* __restrict__ features,
    const float* __restrict__ probs,
    const float* __restrict__ grad_probs,
    float* __restrict__ grad_first,
    float* __restrict__ grad_first_bias,
    float* __restrict__ grad_second,
    float* __restrict__ grad_second_bias,
    float* __restrict__ grad_nobody,
    float* __restrict__ grad_region,
    int height, int width, int players, float beta) {
  extern __shared__ float shared[];
  float* reduce = shared;                    // blockDim.x * 4, for the region
  float* grad_scores = shared + 4 * blockDim.x;

  // Softmax backward: dL/dscore_i = p_i * (g_i - sum_j g_j p_j).
  if (threadIdx.x == 0) {
    float weighted = 0.0f;
    for (int i = 0; i <= players; ++i) weighted += grad_probs[i] * probs[i];
    for (int i = 0; i <= players; ++i) {
      grad_scores[i] = probs[i] * (grad_probs[i] - weighted);
    }
    grad_nobody[0] = grad_scores[players];
    grad_second_bias[0] = 0.0f;
    for (int p = 0; p < players; ++p) grad_second_bias[0] += grad_scores[p];
  }
  __syncthreads();

  for (int p = 0; p < players; ++p) {
    const float* row = features + p * kFeatures;
    const float upstream = grad_scores[p];

    // Head backward, and the gradient arriving at the peak feature, which is
    // the only one the sampling region can move.
    float grad_peak = 0.0f;
    float grad_mean = 0.0f;
    if (threadIdx.x == 0) {
      float standard[kFeatures];
      for (int f = 0; f < kFeatures; ++f) {
        standard[f] = (row[f] - feature_mean[f]) / fmaxf(feature_scale[f], kEps);
      }
      for (int h = 0; h < kHidden; ++h) {
        float pre = first_bias[h];
        for (int f = 0; f < kFeatures; ++f) pre += first[h * kFeatures + f] * standard[f];
        const float activated = tanhf(pre);
        const float grad_pre = upstream * second[h] * (1.0f - activated * activated);
        grad_second[h] += upstream * activated;
        grad_first_bias[h] += grad_pre;
        for (int f = 0; f < kFeatures; ++f) {
          grad_first[h * kFeatures + f] += grad_pre * standard[f];
        }
        // ...and back through the standardisation to the raw features, which
        // is where the sampling region's gradient comes from.
        grad_peak += grad_pre * first[h * kFeatures + 4] / fmaxf(feature_scale[4], kEps);
        grad_mean += grad_pre * first[h * kFeatures + 5] / fmaxf(feature_scale[5], kEps);
      }
      reduce[0] = grad_peak;
      reduce[1] = grad_mean;
    }
    __syncthreads();
    grad_peak = reduce[0];
    grad_mean = reduce[1];
    __syncthreads();

    // The sweep backward. d(peak)/d(e_k) is a softmax over the samples at
    // inverse temperature beta -- which is the whole reason the pooling is a
    // smooth maximum and not a hard one: a hard max would put all the gradient
    // on one sample and none anywhere else.
    const float* box = boxes + p * 4;
    const float peak = row[4];
    float local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = threadIdx.x; s < kSamples; s += blockDim.x) {
      float x, y, dxr0, dxr1, dyr2, dyr3;
      sample_position(box, region, s % kGrid, s / kGrid, &x, &y,
                      &dxr0, &dxr1, &dyr2, &dyr3);
      float dchroma_dx, dchroma_dy;
      const float value = ball_response(image, height, width, x, y,
                                        kSurround * fmaxf(box[3] - box[1], kEps),
                                        &dchroma_dx, &dchroma_dy);
      const float softmax_weight = __expf(beta * (value - peak)) / kSamples;
      const float grad_value = grad_peak * softmax_weight + grad_mean / kSamples;
      local[0] += grad_value * dchroma_dx * dxr0;
      local[1] += grad_value * dchroma_dx * dxr1;
      local[2] += grad_value * dchroma_dy * dyr2;
      local[3] += grad_value * dchroma_dy * dyr3;
    }
    for (int k = 0; k < 4; ++k) reduce[threadIdx.x * 4 + k] = local[k];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int k = 0; k < 4; ++k) {
          reduce[threadIdx.x * 4 + k] += reduce[(threadIdx.x + stride) * 4 + k];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      for (int k = 0; k < 4; ++k) grad_region[k] += reduce[k];
    }
    __syncthreads();
  }
}

std::vector<torch::Tensor> possession_forward(
    torch::Tensor image, torch::Tensor boxes, torch::Tensor ball,
    torch::Tensor first, torch::Tensor first_bias, torch::Tensor second,
    torch::Tensor second_bias, torch::Tensor nobody, torch::Tensor region,
    torch::Tensor feature_mean, torch::Tensor feature_scale, double beta) {
  const int players = boxes.size(0);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(image.device());
  auto features = torch::zeros({players, kFeatures}, options);
  auto probs = torch::zeros({players + 1}, options);
  const int threads = 64;
  const size_t bytes = (2 * threads + players + 1) * sizeof(float);
  possession_forward_kernel<<<1, threads, bytes>>>(
      image.data_ptr<unsigned char>(), boxes.data_ptr<float>(),
      ball.data_ptr<float>(), first.data_ptr<float>(),
      first_bias.data_ptr<float>(), second.data_ptr<float>(),
      second_bias.data_ptr<float>(), nobody.data_ptr<float>(),
      region.data_ptr<float>(), feature_mean.data_ptr<float>(),
      feature_scale.data_ptr<float>(), features.data_ptr<float>(),
      probs.data_ptr<float>(), image.size(0), image.size(1), players,
      (float)beta);
  return {probs, features};
}

std::vector<torch::Tensor> possession_backward(
    torch::Tensor image, torch::Tensor boxes, torch::Tensor first,
    torch::Tensor first_bias, torch::Tensor second, torch::Tensor region,
    torch::Tensor feature_mean, torch::Tensor feature_scale,
    torch::Tensor features, torch::Tensor probs, torch::Tensor grad_probs,
    double beta) {
  const int players = boxes.size(0);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(image.device());
  auto grad_first = torch::zeros({kHidden, kFeatures}, options);
  auto grad_first_bias = torch::zeros({kHidden}, options);
  auto grad_second = torch::zeros({kHidden}, options);
  auto grad_second_bias = torch::zeros({1}, options);
  auto grad_nobody = torch::zeros({1}, options);
  auto grad_region = torch::zeros({4}, options);
  const int threads = 64;
  const size_t bytes = (4 * threads + players + 1) * sizeof(float);
  possession_backward_kernel<<<1, threads, bytes>>>(
      image.data_ptr<unsigned char>(), boxes.data_ptr<float>(),
      first.data_ptr<float>(), first_bias.data_ptr<float>(),
      second.data_ptr<float>(), region.data_ptr<float>(),
      feature_mean.data_ptr<float>(), feature_scale.data_ptr<float>(),
      features.data_ptr<float>(), probs.data_ptr<float>(),
      grad_probs.data_ptr<float>(), grad_first.data_ptr<float>(),
      grad_first_bias.data_ptr<float>(), grad_second.data_ptr<float>(),
      grad_second_bias.data_ptr<float>(), grad_nobody.data_ptr<float>(),
      grad_region.data_ptr<float>(), image.size(0), image.size(1), players,
      (float)beta);
  return {grad_first, grad_first_bias, grad_second, grad_second_bias,
          grad_nobody, grad_region};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("possession_forward", &possession_forward, "fused possession forward");
  m.def("possession_backward", &possession_backward, "fused possession backward");
}
