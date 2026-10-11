// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/cuda/cuda_common.hpp
// Description: Declares CUDA and NPP error checks and stream helpers.
// -----------------------------------------------------------------------------

#ifndef HRT_CUDA_CUDA_COMMON_HPP
#define HRT_CUDA_CUDA_COMMON_HPP

#include <hrt/core/context.hpp>

#include <cuda_runtime_api.h>
#include <npp.h>

namespace hrt::backend {

inline auto cuda_stream(core::Stream* stream) -> cudaStream_t {
  return stream == nullptr ? nullptr : static_cast<cudaStream_t>(stream->handle());
}

// NPP stream context for the current device; device properties are cached.
auto npp_context(core::Stream* stream) -> NppStreamContext;

} // namespace hrt::backend

#endif // HRT_CUDA_CUDA_COMMON_HPP
