// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/cuda_image_kernels.hpp
// Description: Declares the fused CUDA color conversion, resize, padding
//              and normalization kernel.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_CUDA_IMAGE_KERNELS_HPP
#define HRT_PREPROCESS_CUDA_IMAGE_KERNELS_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hrt::preprocess {

enum class CudaResizeSource {
  Interleaved, // uint8 BGR or RGB pixels.
  Nv12,        // Y plane, then interleaved U and V at half resolution.
  Nv21,        // Y plane, then interleaved V and U at half resolution.
};

// One uint8 source region resized into the region of a 3-channel input and
// padded around it. Every output pixel is written.
struct CudaResizeTask {
  CudaResizeSource source = CudaResizeSource::Interleaved;
  const std::uint8_t* src = nullptr;    // First pixel of the source region; the Y plane for NV12/NV21.
  std::int64_t src_pitch = 0;           // Bytes between source rows, also of the chroma plane.
  const std::uint8_t* chroma = nullptr; // NV12/NV21: first row of the chroma plane of the whole image.
  std::int32_t origin_x = 0;            // NV12/NV21: region position in the image, for its chroma.
  std::int32_t origin_y = 0;
  std::int32_t src_width = 0; // Source region size; reads replicate its border.
  std::int32_t src_height = 0;
  double scale_x = 1.0; // Source pixels per region pixel, as OpenCV: 1 / (region / source) in double.
  double scale_y = 1.0;
  std::uint8_t* dst = nullptr; // First byte of the input.
  std::int64_t dst_pitch = 0;  // Bytes between input rows.
  std::int64_t dst_plane = 0;  // Bytes between planes; 0 for interleaved.
  std::int32_t dst_width = 0;
  std::int32_t dst_height = 0;
  std::int32_t roi_x = 0; // Region receiving the resized source.
  std::int32_t roi_y = 0;
  std::int32_t roi_width = 0;
  std::int32_t roi_height = 0;
  std::int32_t channel[3] = {0, 1, 2}; // Source channel of each input channel; NV12/NV21 sample as BGR.
  float alpha[3] = {1.0F, 1.0F, 1.0F}; // Per input channel: value * alpha + beta.
  float beta[3] = {0.0F, 0.0F, 0.0F};
  float pad[3] = {0.0F, 0.0F, 0.0F}; // Final padding value per input channel.
  bool float_output = true;          // float32, else uint8.
  bool nearest = false;              // OpenCV INTER_NEAREST, else INTER_LINEAR.
};

// Launches the tasks on `stream`, at most kCudaResizeTasksPerLaunch per
// kernel, with OpenCV's uint8 sampling (its fixed-point INTER_LINEAR) before
// alpha and beta apply.
auto launch_cuda_resize(const CudaResizeTask* tasks, std::size_t count, cudaStream_t stream) -> cudaError_t;

inline constexpr std::size_t kCudaResizeTasksPerLaunch = 16;

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_CUDA_IMAGE_KERNELS_HPP
