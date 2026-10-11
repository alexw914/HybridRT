// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/cuda/cuda_memory.hpp
// Description: Declares CUDA device memory and copies to and from it.
// -----------------------------------------------------------------------------

#ifndef HRT_CUDA_CUDA_MEMORY_HPP
#define HRT_CUDA_CUDA_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::backend {

// Device-only memory; host access goes through copies to host memory.
class CudaMemory final : public core::Memory {
public:
  // Owning; `stride` must already be validated.
  CudaMemory(const core::Shape& shape, const core::Stride& stride);

  CudaMemory(std::uintptr_t device, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  CudaMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;
};

// Copies between any of host and CUDA memory on `stream`; one side must be
// CUDA. Asynchronous for pinned host memory.
auto copy_cuda(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool;

} // namespace hrt::backend

#endif // HRT_CUDA_CUDA_MEMORY_HPP
