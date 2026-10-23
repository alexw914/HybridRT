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
  if (pointer.cpu != nullptr && cudaFreeHost(pointer.cpu) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaFreeHost failed: {}", cudaGetErrorString(cudaGetLastError()));
  pointer.cpu = nullptr;
  pointer.capacity = 0;
}

auto host_allocate(memory::BasePointer& pointer, std::size_t size) -> bool {
  host_release(pointer);
  if (size == 0)
    return true;
  void* host = nullptr;
  const auto status = cudaHostAlloc(&host, size, cudaHostAllocDefault);
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: cudaHostAlloc({}) failed: {}", size, cudaGetErrorString(status));
  pointer.cpu = host;
  pointer.capacity = size;
  return true;
}

class CudaHostMemory final : public core::Memory {
public:
  CudaHostMemory() : core::Memory(core::MemoryType::Cpu) {
    pointer_->owned = true;
    pointer_->release = host_release;
  }

  CudaHostMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride)
      : core::Memory(std::move(pointer)) {
    shape_ = shape;
    stride_ = stride;
  }

  auto ref() -> SharedPtr<core::Memory> override { return std::make_shared<CudaHostMemory>(pointer_, shape_, stride_); }

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override {
    auto memory_stride = stride;
    if (memory_stride.empty())
      memory_stride = core::contiguous_stride(shape);
    else if (!memory::check_stride(shape, memory_stride))
      return false;
    if (memory_stride.front() > pointer_->capacity && !host_allocate(*pointer_, memory_stride.front()))
      return false;
    shape_ = shape;
    stride_ = std::move(memory_stride);
    return true;
  }

  auto shrink_to_fit() -> bool override {
    const auto size = stride_.empty() ? 0 : stride_.front();
    return size == pointer_->capacity || host_allocate(*pointer_, size);
  }
};

} // namespace

auto create_cuda_host_memory() -> SharedPtr<core::Memory> {
  return std::make_shared<CudaHostMemory>();
}

} // namespace hrt::backend
