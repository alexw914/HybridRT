// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/memory/cpu_memory.cpp
// Description: Implements host memory allocation and host-to-host copies.
// -----------------------------------------------------------------------------

#include "include/memory/cpu_memory.hpp"

#include <cstring>

#ifdef HRT_USE_DMA_BUFFER
#include "include/memory/dma_buffer_memory.hpp"
#endif

namespace hrt::memory {

namespace {

// Matches common SIMD widths so kernels can use aligned loads.
constexpr std::size_t kCpuAlignment = 64;

} // namespace

CpuMemory::CpuMemory(const core::Shape& shape, const core::Stride& stride) : core::Memory(core::MemoryType::Cpu) {
  pointer_->owned_ = true;
  pointer_->host_storage_ = kitzoo::memory::ByteBuffer(0, kCpuAlignment);
  resize(shape, stride);
}

CpuMemory::CpuMemory(void* data, std::size_t size, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(core::MemoryType::Cpu) {
  shape_ = shape;
  stride_ = stride;
  pointer_->cpu_ = data;
  pointer_->capacity_ = size;
}

auto CpuMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<CpuMemory>(*this);
}

// The old block is freed first: contents are undefined after reallocation, so this neither copies them nor
// overallocates as ByteBuffer's geometric growth would.
auto CpuMemory::reallocate(std::size_t size) -> bool {
  pointer_->host_storage_.reset();
  pointer_->host_storage_.resize(size);
  pointer_->cpu_ = pointer_->host_storage_.data();
  pointer_->capacity_ = size;
  return true;
}

auto copy_cpu_to_cpu(core::Memory& dst, core::Memory& src, std::size_t axis) -> bool {
  const auto& shape = src.shape();
  std::size_t chunk = 1;
  for (auto i = axis; i < shape.size(); ++i)
    chunk *= static_cast<std::size_t>(shape[i]);
  const auto total = static_cast<std::size_t>(core::get_volume(shape));
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
