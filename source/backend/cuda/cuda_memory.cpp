// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/cuda/cuda_memory.cpp
// Description: Implements CUDA device memory and copies to and from it.
// -----------------------------------------------------------------------------

#include "include/backend/cuda/cuda_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include "include/backend/cuda/cuda_common.hpp"

namespace hrt::backend {

namespace {

auto cuda_release(memory::BasePointer& pointer) -> void {
  if (pointer.device != 0 && cudaFree(reinterpret_cast<void*>(pointer.device)) != cudaSuccess)
    KZ_LOG_ERROR("cuda: cudaFree failed: {}", cudaGetErrorString(cudaGetLastError()));
  pointer.device = 0;
  pointer.capacity = 0;
}

auto cuda_allocate(memory::BasePointer& pointer, std::size_t size) -> bool {
  cuda_release(pointer);
  if (size == 0)
    return true;
  void* device = nullptr;
  KZ_CHECK_RETURN(cudaMalloc(&device, size) == cudaSuccess, false, "cuda: cudaMalloc({}) failed: {}", size,
                  cudaGetErrorString(cudaGetLastError()));
  pointer.device = reinterpret_cast<std::uintptr_t>(device);
  pointer.capacity = size;
  return true;
}

auto address(core::Memory& memory, bool write, core::Stream* stream) -> void* {
  if (memory.memory_type() == core::MemoryType::Cuda)
    return reinterpret_cast<void*>(write ? memory.mutable_device(stream) : memory.device(stream));
  return write ? memory.mutable_cpu(stream) : const_cast<void*>(memory.cpu(stream));
}

} // namespace

CudaMemory::CudaMemory(const core::Shape& shape, const core::Stride& stride) : core::Memory(core::MemoryType::Cuda) {
  shape_ = shape;
  stride_ = stride;
  pointer_->owned = true;
  pointer_->release = cuda_release;
  cuda_allocate(*pointer_, stride_.empty() ? 0 : stride_[0]);
}

CudaMemory::CudaMemory(std::uintptr_t device, std::size_t size, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(core::MemoryType::Cuda) {
  shape_ = shape;
  stride_ = stride;
  pointer_->device = device;
  pointer_->capacity = size;
}

CudaMemory::CudaMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(std::move(pointer)) {
  shape_ = shape;
  stride_ = stride;
}

auto CudaMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<CudaMemory>(pointer_, shape_, stride_);
}

auto CudaMemory::resize(const core::Shape& shape, const core::Stride& stride) -> bool {
  auto resolved = memory::resolve_stride(shape, stride);
  if (!resolved)
    return false;
  const auto size = resolved->front();
  if (size > pointer_->capacity) {
    if (!pointer_->owned) {
      KZ_LOG_ERROR("CudaMemory: cannot grow wrapped memory from {} to {} bytes", pointer_->capacity, size);
      return false;
    }
    if (!cuda_allocate(*pointer_, size))
      return false;
  }
  shape_ = shape;
  stride_ = std::move(*resolved);
  return true;
}

auto CudaMemory::shrink_to_fit() -> bool {
  const auto size = stride_.empty() ? 0 : stride_[0];
  if (!pointer_->owned || size == pointer_->capacity)
    return true;
  return cuda_allocate(*pointer_, size);
}

auto copy_cuda(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool {
  const auto to_device = dst.memory_type() == core::MemoryType::Cuda;
  const auto from_device = src.memory_type() == core::MemoryType::Cuda;
  const auto kind =
      to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : cudaMemcpyDeviceToHost;
  auto* to = address(dst, true, stream);
  auto* from = address(src, false, stream);
  if (to == nullptr || from == nullptr)
    return false;

  const auto& shape = src.shape();
  std::size_t chunk = 1;
  for (auto i = axis; i < shape.size(); ++i)
    chunk *= static_cast<std::size_t>(shape[i]);
  const auto total = static_cast<std::size_t>(core::volume(shape));
  if (chunk == 0 || total == 0)
    return true;
  const auto status = axis == 0 ? cudaMemcpyAsync(to, from, total, kind, cuda_stream(stream))
                                : cudaMemcpy2DAsync(to, dst.stride()[axis], from, src.stride()[axis], chunk,
                                                    total / chunk, kind, cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: copy failed: {}", cudaGetErrorString(status));
  return true;
}

} // namespace hrt::backend
