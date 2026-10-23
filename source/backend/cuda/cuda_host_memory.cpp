// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/cuda/cuda_host_memory.cpp
// Description: Implements CUDA page-locked host memory.
// -----------------------------------------------------------------------------

#include "include/backend/cuda/cuda_host_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <cuda_runtime_api.h>

#include "include/memory/base_pointer.hpp"

namespace hrt::backend {

namespace {

auto host_release(memory::BasePointer& pointer) -> void {
  if (pointer.cpu_ != nullptr && cudaFreeHost(pointer.cpu_) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaFreeHost failed: {}", cudaGetErrorString(cudaGetLastError()));
  pointer.cpu_ = nullptr;
  pointer.capacity_ = 0;
}

auto host_allocate(memory::BasePointer& pointer, std::size_t size) -> bool {
  host_release(pointer);
  if (size == 0)
    return true;
  void* host = nullptr;
  const auto status = cudaHostAlloc(&host, size, cudaHostAllocDefault);
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: cudaHostAlloc({}) failed: {}", size, cudaGetErrorString(status));
  pointer.cpu_ = host;
  pointer.capacity_ = size;
  return true;
}

class CudaHostMemory final : public core::Memory {
public:
  CudaHostMemory() : core::Memory(core::MemoryType::Cpu) {
    pointer_->owned_ = true;
    pointer_->release_ = host_release;
  }

  auto ref() -> SharedPtr<core::Memory> override { return std::make_shared<CudaHostMemory>(*this); }

protected:
  auto reallocate(std::size_t size) -> bool override { return host_allocate(*pointer_, size); }
};

} // namespace

auto create_cuda_host_memory() -> SharedPtr<core::Memory> {
  return std::make_shared<CudaHostMemory>();
}

} // namespace hrt::backend
