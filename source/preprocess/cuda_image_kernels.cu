// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/cuda_image_kernels.cu
// Description: Implements the fused CUDA resize, padding, conversion and
//              normalization kernel.
// -----------------------------------------------------------------------------

#include <algorithm>

#include "include/preprocess/cuda_image_kernels.hpp"

namespace hrt::preprocess {

namespace {

// The tasks of one launch, passed by value: kernel parameters are captured at
// launch, so no device copy has to outlive the call.
struct ResizeLaunch {
  CudaResizeTask tasks[kCudaResizeTasksPerLaunch];
};

constexpr int kBlockWidth = 32;
constexpr int kBlockHeight = 8;

__device__ auto pixel(const CudaResizeTask& s, int x, int y) -> const std::uint8_t* {
  return s.src + static_cast<std::int64_t>(y) * s.src_pitch + static_cast<std::int64_t>(x) * 3;
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

// OpenCV's vectorized vertical pass: rows of horizontal sums are shifted by 4,
// multiplied by the weights keeping the high 16 bits, then rounded by 2 bits.
__device__ auto linear_value(int top, int bottom, int w0, int w1) -> float {
  const auto high = [](int sum, int weight) -> int { return ((sum >> 4) * weight) >> 16; };
  return static_cast<float>(min(max((high(top, w0) + high(bottom, w1) + 2) >> 2, 0), 255));
}

__device__ auto store(const CudaResizeTask& s, int x, int y, int c, float value) -> void {
  auto* row = s.dst + static_cast<std::int64_t>(y) * s.dst_pitch;
  std::int64_t element = static_cast<std::int64_t>(x) * 3 + c;
  if (s.dst_plane != 0) {
    row += static_cast<std::int64_t>(c) * s.dst_plane;
    element = x;
  }
  if (s.float_output)
    reinterpret_cast<float*>(row)[element] = value;
  else
    row[element] = static_cast<std::uint8_t>(min(max(__float2int_rn(value), 0), 255));
}

__global__ auto resize_kernel(ResizeLaunch launch) -> void {
  const auto& s = launch.tasks[blockIdx.z];
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
  if (s.nearest) {
    // In double as OpenCV: exact products such as 78 * 50 / 156 must floor the same way.
    const int sx = min(static_cast<int>(floor(static_cast<double>(rx) * s.scale_x)), s.src_width - 1);
    const int sy = min(static_cast<int>(floor(static_cast<double>(ry) * s.scale_y)), s.src_height - 1);
    const auto* p = pixel(s, sx, sy);
    for (int c = 0; c < 3; ++c)
      value[c] = static_cast<float>(p[c]);
  } else {
    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    int ax0 = 0;
    int ax1 = 0;
    int ay0 = 0;
    int ay1 = 0;
    linear_tap(rx, s.scale_x, s.src_width, x0, x1, ax0, ax1);
    linear_tap(ry, s.scale_y, s.src_height, y0, y1, ay0, ay1);
    const auto* p00 = pixel(s, x0, y0);
    const auto* p01 = pixel(s, x1, y0);
    const auto* p10 = pixel(s, x0, y1);
    const auto* p11 = pixel(s, x1, y1);
    for (int c = 0; c < 3; ++c)
      value[c] = linear_value(p00[c] * ax0 + p01[c] * ax1, p10[c] * ax0 + p11[c] * ax1, ay0, ay1);
  }
  for (int c = 0; c < 3; ++c)
    store(s, x, y, c, value[s.channel[c]] * s.alpha[c] + s.beta[c]);
}

} // namespace

auto launch_cuda_resize(const CudaResizeTask* tasks, std::size_t count, cudaStream_t stream) -> cudaError_t {
  for (std::size_t first = 0; first < count; first += kCudaResizeTasksPerLaunch) {
    const auto n = std::min(kCudaResizeTasksPerLaunch, count - first);
    ResizeLaunch launch{};
    int width = 0;
    int height = 0;
    for (std::size_t i = 0; i < n; ++i) {
      launch.tasks[i] = tasks[first + i];
      width = std::max(width, tasks[first + i].dst_width);
      height = std::max(height, tasks[first + i].dst_height);
    }
    const dim3 block(kBlockWidth, kBlockHeight);
    const dim3 grid(static_cast<unsigned int>((width + kBlockWidth - 1) / kBlockWidth),
                    static_cast<unsigned int>((height + kBlockHeight - 1) / kBlockHeight),
                    static_cast<unsigned int>(n));
    resize_kernel<<<grid, block, 0, stream>>>(launch);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
      return error;
  }
  return cudaSuccess;
}

} // namespace hrt::preprocess
