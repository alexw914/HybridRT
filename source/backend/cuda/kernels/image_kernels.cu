// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/cuda/kernels/image_kernels.cu
// Description: Implements the fused CUDA color conversion, resize, padding
//              and normalization kernel.
// -----------------------------------------------------------------------------

#include <algorithm>

#include "include/backend/cuda/kernels/image_kernels.hpp"

namespace hrt::backend {

namespace {

// The params of one launch, passed by value: kernel parameters are captured at
// launch, so no device copy has to outlive the call.
struct PreprocessBatch {
  PreprocessParam params[kPreprocessParamsPerLaunch];
};

constexpr int kBlockWidth = 32;
constexpr int kBlockHeight = 8;

// OpenCV's fixed-point BT.601 video-range YUV to RGB (color_yuv): 20 fractional bits.
constexpr int kYuvShift = 20;
constexpr int kYuvY = 1220542;
constexpr int kYuvUB = 2116026;
constexpr int kYuvUG = -409993;
constexpr int kYuvVG = -852492;
constexpr int kYuvVR = 1673527;

__device__ auto saturate(int value) -> int {
  return min(max(value, 0), 255);
}

// Pixel (x, y) of the source region: its channels for interleaved sources,
// BGR as OpenCV's cvtColor gives it for NV12 and NV21.
__device__ auto fetch(const PreprocessParam& s, int x, int y, int (&out)[3]) -> void {
  const auto* row = s.src + static_cast<std::int64_t>(y) * s.src_pitch;
  if (s.source == InputFormat::Bgr || s.source == InputFormat::Rgb) {
    const auto* p = row + static_cast<std::int64_t>(x) * 3;
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    return;
  }
  const int gx = s.origin_x + x;
  const int gy = s.origin_y + y;
  int u = 0, v = 0;
  if (s.source == InputFormat::Nv12 || s.source == InputFormat::Nv21) {
    const auto* uv = s.chroma + static_cast<std::int64_t>(gy / 2) * s.src_pitch + static_cast<std::int64_t>(gx / 2) * 2;
    u = s.source == InputFormat::Nv12 ? uv[0] : uv[1];
    v = s.source == InputFormat::Nv12 ? uv[1] : uv[0];
  } else {
    // Chroma rows are half as wide as the image rows holding them, so OpenCV packs two per row: half row k
    // after the Y plane starts at (k / 2) * pitch + (k % 2) * width / 2, and the second plane follows the first.
    const auto half = [&](int k) -> const std::uint8_t* {
      return s.chroma + static_cast<std::int64_t>(k >> 1) * s.src_pitch +
             static_cast<std::int64_t>(k & 1) * (s.image_width / 2);
    };
    const auto first = half(gy / 2)[gx / 2];
    const auto second = half(s.image_height / 2 + gy / 2)[gx / 2];
    u = s.source == InputFormat::I420 ? first : second;
    v = s.source == InputFormat::I420 ? second : first;
  }
  u -= 128;
  v -= 128;
  const int luma = max(0, static_cast<int>(row[x]) - 16) * kYuvY;
  const int half = 1 << (kYuvShift - 1);
  out[0] = saturate((luma + half + kYuvUB * u) >> kYuvShift);
  out[1] = saturate((luma + half + kYuvVG * v + kYuvUG * u) >> kYuvShift);
  out[2] = saturate((luma + half + kYuvVR * v) >> kYuvShift);
}

// OpenCV's fixed-point INTER_LINEAR weights: 11 fractional bits, each rounded on its own.
constexpr float kCoefficientScale = 2048.0F;

// Taps and weights along one axis as OpenCV's INTER_LINEAR: half-pixel centers
// in double, clamped to the region.
__device__ auto linear_tap(int d, double scale, int size, int& first, int& second, int& w0, int& w1) -> void {
  float f = static_cast<float>((static_cast<double>(d) + 0.5) * scale - 0.5);
  int s = static_cast<int>(floorf(f));
  f -= static_cast<float>(s);
  if (s < 0) {
    s = 0;
    f = 0.0F;
  }
  if (s >= size - 1) {
    s = size - 1;
    f = 0.0F;
  }
  first = s;
  second = min(s + 1, size - 1);
  w0 = __float2int_rn((1.0F - f) * kCoefficientScale);
  w1 = __float2int_rn(f * kCoefficientScale);
}

// Taps and weights along one axis as OpenCV's INTER_CUBIC (A = -0.75): half-pixel
// centers in double, four taps that repeat the region's border.
__device__ auto cubic_taps(int d, double scale, int size, int (&index)[4], int (&weight)[4]) -> void {
  float f = static_cast<float>((static_cast<double>(d) + 0.5) * scale - 0.5);
  const int s = static_cast<int>(floorf(f));
  f -= static_cast<float>(s);
  constexpr float kA = -0.75F;
  float c[4];
  c[0] = ((kA * (f + 1.0F) - 5.0F * kA) * (f + 1.0F) + 8.0F * kA) * (f + 1.0F) - 4.0F * kA;
  c[1] = ((kA + 2.0F) * f - (kA + 3.0F)) * f * f + 1.0F;
  c[2] = ((kA + 2.0F) * (1.0F - f) - (kA + 3.0F)) * (1.0F - f) * (1.0F - f) + 1.0F;
  c[3] = 1.0F - c[0] - c[1] - c[2];
  for (int k = 0; k < 4; ++k) {
    index[k] = min(max(s - 1 + k, 0), size - 1);
    weight[k] = __float2int_rn(c[k] * kCoefficientScale);
  }
}

// OpenCV's vectorized vertical pass: rows of horizontal sums are shifted by 4,
// multiplied by the weights keeping the high 16 bits, then rounded by 2 bits.
__device__ auto linear_value(int top, int bottom, int w0, int w1) -> float {
  const auto high = [](int sum, int weight) -> int { return ((sum >> 4) * weight) >> 16; };
  return static_cast<float>(saturate((high(top, w0) + high(bottom, w1) + 2) >> 2));
}

__device__ auto store(const PreprocessParam& s, int x, int y, int c, float value) -> void {
  auto* row = s.dst + static_cast<std::int64_t>(y) * s.dst_pitch;
  std::int64_t element = static_cast<std::int64_t>(x) * 3 + c;
  if (s.dst_plane != 0) {
    row += static_cast<std::int64_t>(c) * s.dst_plane;
    element = x;
  }
  if (s.float_output)
    reinterpret_cast<float*>(row)[element] = value;
  else
    row[element] = static_cast<std::uint8_t>(saturate(__float2int_rn(value)));
}

__global__ auto preprocess_kernel(PreprocessBatch batch) -> void {
  const auto& s = batch.params[blockIdx.z];
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= s.dst_width || y >= s.dst_height)
    return;
  const int rx = x - s.roi_x;
  const int ry = y - s.roi_y;
  if (rx < 0 || ry < 0 || rx >= s.roi_width || ry >= s.roi_height) {
    for (int c = 0; c < 3; ++c)
      store(s, x, y, c, s.pad[c]);
    return;
  }
  float value[3];
  if (s.interpolation == ResizeInterpolation::Linear) {
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0, ax0 = 0, ax1 = 0, ay0 = 0, ay1 = 0;
    linear_tap(rx, s.scale_x, s.src_width, x0, x1, ax0, ax1);
    linear_tap(ry, s.scale_y, s.src_height, y0, y1, ay0, ay1);
    int p00[3], p01[3], p10[3], p11[3];
    fetch(s, x0, y0, p00);
    fetch(s, x1, y0, p01);
    fetch(s, x0, y1, p10);
    fetch(s, x1, y1, p11);
    for (int c = 0; c < 3; ++c)
      value[c] = linear_value(p00[c] * ax0 + p01[c] * ax1, p10[c] * ax0 + p11[c] * ax1, ay0, ay1);
  } else if (s.interpolation == ResizeInterpolation::Nearest) {
    // In double as OpenCV: exact products such as 78 * 50 / 156 must floor the same way.
    const int sx = min(static_cast<int>(floor(static_cast<double>(rx) * s.scale_x)), s.src_width - 1);
    const int sy = min(static_cast<int>(floor(static_cast<double>(ry) * s.scale_y)), s.src_height - 1);
    int p[3];
    fetch(s, sx, sy, p);
    for (int c = 0; c < 3; ++c)
      value[c] = static_cast<float>(p[c]);
  } else {
    int xs[4], ys[4], wx[4], wy[4];
    cubic_taps(rx, s.scale_x, s.src_width, xs, wx);
    cubic_taps(ry, s.scale_y, s.src_height, ys, wy);
    // OpenCV's vectorized vertical pass sums the rows in float, scaled back by 2048 squared.
    constexpr float kScale = 1.0F / (kCoefficientScale * kCoefficientScale);
    float sum[3] = {0.0F, 0.0F, 0.0F};
    for (int r = 0; r < 4; ++r) {
      int row[3] = {0, 0, 0};
      for (int k = 0; k < 4; ++k) {
        int p[3];
        fetch(s, xs[k], ys[r], p);
        for (int c = 0; c < 3; ++c)
          row[c] += p[c] * wx[k];
      }
      for (int c = 0; c < 3; ++c)
        sum[c] = fmaf(static_cast<float>(row[c]), static_cast<float>(wy[r]) * kScale, sum[c]);
    }
    for (int c = 0; c < 3; ++c)
      value[c] = static_cast<float>(saturate(__float2int_rn(sum[c])));
  }
  for (int c = 0; c < 3; ++c)
    store(s, x, y, c, value[s.channel[c]] * s.alpha[c] + s.beta[c]);
}

} // namespace

auto launch_cuda_preprocess(const PreprocessParam* params, std::size_t count, cudaStream_t stream) -> cudaError_t {
  for (std::size_t first = 0; first < count; first += kPreprocessParamsPerLaunch) {
    const auto n = std::min(kPreprocessParamsPerLaunch, count - first);
    PreprocessBatch batch{};
    int width = 0, height = 0;
    for (std::size_t i = 0; i < n; ++i) {
      batch.params[i] = params[first + i];
      width = std::max(width, params[first + i].dst_width);
      height = std::max(height, params[first + i].dst_height);
    }
    const dim3 block(kBlockWidth, kBlockHeight);
    const dim3 grid(static_cast<unsigned int>((width + kBlockWidth - 1) / kBlockWidth),
                    static_cast<unsigned int>((height + kBlockHeight - 1) / kBlockHeight),
                    static_cast<unsigned int>(n));
    preprocess_kernel<<<grid, block, 0, stream>>>(batch);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
      return error;
  }
  return cudaSuccess;
}

} // namespace hrt::backend
