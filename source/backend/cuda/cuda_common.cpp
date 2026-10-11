// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/cuda/cuda_common.cpp
// Description: Implements NPP stream contexts.
// -----------------------------------------------------------------------------

#include "include/backend/cuda/cuda_common.hpp"

#include <kitzoo/log/logger.hpp>

#include <mutex>

namespace hrt::backend {

auto npp_context(core::Stream* stream) -> NppStreamContext {
  static std::mutex mutex;
  static Map<int, cudaDeviceProp> properties;
  NppStreamContext context{};
  int device = 0;
  KZ_CHECK_RETURN(cudaGetDevice(&device) == cudaSuccess, context, "cuda: cudaGetDevice failed: {}",
                  cudaGetErrorString(cudaGetLastError()));
  {
    const std::lock_guard lock(mutex);
    auto it = properties.find(device);
    if (it == properties.end()) {
      cudaDeviceProp prop{};
      KZ_CHECK_RETURN(cudaGetDeviceProperties(&prop, device) == cudaSuccess, context,
                      "cuda: cudaGetDeviceProperties failed: {}", cudaGetErrorString(cudaGetLastError()));
      it = properties.emplace(device, prop).first;
    }
    const auto& prop = it->second;
    context.nCudaDeviceId = device;
    context.nMultiProcessorCount = prop.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    context.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    context.nSharedMemPerBlock = prop.sharedMemPerBlock;
    context.nCudaDevAttrComputeCapabilityMajor = prop.major;
    context.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  }
  context.hStream = cuda_stream(stream);
  unsigned int flags = 0;
  if (context.hStream != nullptr && cudaStreamGetFlags(context.hStream, &flags) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaStreamGetFlags failed: {}", cudaGetErrorString(cudaGetLastError()));
  context.nStreamFlags = flags;
  return context;
}

} // namespace hrt::backend
