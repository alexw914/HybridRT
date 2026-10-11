// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/cuda/cuda_context.cpp
// Description: Implements the CUDA device context and stream.
// -----------------------------------------------------------------------------

#include "include/backend/cuda/cuda_context.hpp"

#include <kitzoo/log/logger.hpp>

#include "include/backend/cuda/cuda_common.hpp"

namespace hrt::backend {

CudaStream::CudaStream() {
  cudaStream_t stream = nullptr;
  KZ_CHECK_MSG(cudaStreamCreate(&stream) == cudaSuccess, "cuda: cudaStreamCreate failed: {}",
               cudaGetErrorString(cudaGetLastError()));
  handle_ = stream;
}

CudaStream::~CudaStream() {
  if (handle_ != nullptr && cudaStreamDestroy(static_cast<cudaStream_t>(handle_)) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaStreamDestroy failed: {}", cudaGetErrorString(cudaGetLastError()));
}

auto CudaStream::synchronize() -> void {
  KZ_CHECK_RETURN(cudaStreamSynchronize(static_cast<cudaStream_t>(handle_)) == cudaSuccess, ,
                  "cuda: cudaStreamSynchronize failed: {}", cudaGetErrorString(cudaGetLastError()));
}

auto CudaContext::available(int device_id) -> bool {
  int count = 0;
  // No log: machines without a GPU take this path when choosing backends.
  return cudaGetDeviceCount(&count) == cudaSuccess && device_id >= 0 && device_id < count;
}

auto CudaContext::make_current() -> void {
  int device = 0;
  KZ_CHECK_RETURN(cudaGetDevice(&device) == cudaSuccess, , "cuda: cudaGetDevice failed: {}",
                  cudaGetErrorString(cudaGetLastError()));
  if (device != device_id_ && cudaSetDevice(device_id_) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaSetDevice({}) failed: {}", device_id_, cudaGetErrorString(cudaGetLastError()));
}

auto CudaContext::create_stream() -> UniquePtr<core::Stream> {
  make_current();
  return std::make_unique<CudaStream>();
}

} // namespace hrt::backend
