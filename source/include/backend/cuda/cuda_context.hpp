// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/cuda/cuda_context.hpp
// Description: Declares the CUDA device context and stream.
// -----------------------------------------------------------------------------

#ifndef HRT_CUDA_CUDA_CONTEXT_HPP
#define HRT_CUDA_CUDA_CONTEXT_HPP

#include <hrt/core/context.hpp>

namespace hrt::backend {

class CudaStream final : public core::Stream {
public:
  CudaStream();

  ~CudaStream() override;

  CudaStream(const CudaStream&) = delete;
  auto operator=(const CudaStream&) -> CudaStream& = delete;
  CudaStream(CudaStream&&) = delete;
  auto operator=(CudaStream&&) -> CudaStream& = delete;

  auto synchronize() -> void override;
};

class CudaContext final : public core::Context {
public:
  static auto available(int device_id) -> bool;

  explicit CudaContext(int device_id) : core::Context(core::DeviceType::Cuda, device_id) {}

  auto set_current() -> void override;

  auto create_stream() -> UniquePtr<core::Stream> override;
};

} // namespace hrt::backend

#endif // HRT_CUDA_CUDA_CONTEXT_HPP
