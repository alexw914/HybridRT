// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/memory/cpu_memory.cpp
// Description: Implements host memory allocation and host-to-host copies.
// -----------------------------------------------------------------------------

#include "include/memory/cpu_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <cstring>

#ifdef HRT_USE_DMA_BUFFER
#include "include/memory/dma_buffer_memory.hpp"
#endif

namespace hrt::memory {

namespace {

// Matches common SIMD widths so kernels can use aligned loads.
constexpr std::size_t kCpuAlignment = 64;

// ByteBuffer growth keeps contents; callers treat resized memory as undefined.
auto cpu_allocate(BasePointer& pointer, std::size_t size) -> void {
  pointer.host_storage.resize(size);
  pointer.cpu = pointer.host_storage.data();
  pointer.capacity = size;
}

} // namespace

CpuMemory::CpuMemory(const core::Shape& shape, const core::Stride& stride) : core::Memory(core::MemoryType::Cpu) {
  shape_ = shape;
  stride_ = stride;
  pointer_->owned = true;
  pointer_->host_storage = kitzoo::memory::ByteBuffer(0, kCpuAlignment);
  cpu_allocate(*pointer_, stride_.empty() ? 0 : stride_[0]);
}

CpuMemory::CpuMemory(void* data, std::size_t size, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(core::MemoryType::Cpu) {
  shape_ = shape;
  stride_ = stride;
  pointer_->cpu = data;
  pointer_->capacity = size;
}

CpuMemory::CpuMemory(SharedPtr<BasePointer> pointer, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(std::move(pointer)) {
  shape_ = shape;
  stride_ = stride;
}

auto CpuMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<CpuMemory>(pointer_, shape_, stride_);
}

auto CpuMemory::resize(const core::Shape& shape, const core::Stride& stride) -> bool {
  auto memory_stride = stride;
  if (memory_stride.empty())
    memory_stride = core::contiguous_stride(shape);
  else if (!check_stride(shape, memory_stride))
    return false;
  const auto size = memory_stride.front();
  if (size > pointer_->capacity) {
    if (!pointer_->owned) {
      KZ_LOG_ERROR("CpuMemory: cannot grow wrapped memory from {} to {} bytes", pointer_->capacity, size);
      return false;
    }
    cpu_allocate(*pointer_, size);
  }
  shape_ = shape;
  stride_ = std::move(memory_stride);
  return true;
}

auto CpuMemory::shrink_to_fit() -> bool {
  const auto size = stride_.empty() ? 0 : stride_[0];
  if (!pointer_->owned || size == pointer_->capacity)
    return true;
  pointer_->host_storage.resize(size);
  pointer_->host_storage.shrink_to_fit();
  pointer_->cpu = pointer_->host_storage.data();
  pointer_->capacity = size;
  return true;
}

auto copy_cpu_to_cpu(core::Memory& dst, core::Memory& src, std::size_t axis) -> bool {
  const auto& shape = src.shape();
  std::size_t chunk = 1;
  for (auto i = axis; i < shape.size(); ++i)
    chunk *= static_cast<std::size_t>(shape[i]);
  const auto total = static_cast<std::size_t>(core::volume(shape));
  if (chunk == 0 || total == 0)
    return true;

  auto* dst_ptr = static_cast<std::byte*>(dst.mutable_cpu());
  const auto* src_ptr = static_cast<const std::byte*>(src.cpu());
  if (dst_ptr == nullptr || src_ptr == nullptr)
    return false;
  // stride[0] is the whole region, so a fully contiguous copy is one chunk.
  const auto dst_step = axis == 0 ? chunk : dst.stride()[axis];
  const auto src_step = axis == 0 ? chunk : src.stride()[axis];
  for (std::size_t i = 0; i < total / chunk; ++i) {
    std::memcpy(dst_ptr, src_ptr, chunk);
    dst_ptr += dst_step;
    src_ptr += src_step;
  }
  return true;
}

auto host_engine_input_memory() -> core::MemoryType {
#if defined(HRT_USE_DMA_BUFFER) && defined(HRT_USE_RKNN)
  if (dma_heap_available())
    return core::MemoryType::DmaBuffer;
#endif
  return core::MemoryType::Cpu;
}

} // namespace hrt::memory
