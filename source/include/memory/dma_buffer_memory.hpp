// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/dma_buffer_memory.hpp
// Description: Declares Linux dma-heap memory shared by media devices and the
//              host.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_DMA_BUFFER_MEMORY_HPP
#define HRT_MEMORY_DMA_BUFFER_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::memory {

// Which heaps a dma-buf may come from.
enum class DmaHeapUse {
  // Pages below 4 GB first (system-dma32, cma), for devices with 32-bit DMA such as RGA2; when those are exhausted
  // the system heap is used and below_4gb() turns false.
  Device32,
  // Any pages (system heap first), for devices behind an IOMMU such as the RK3588 NPU.
  Any,
};

// DmaBuffer memory: a dma-heap buffer mapped to the host; device() is the dma-buf fd. Without an explicit stride, rows
// are aligned to 16 elements as media devices need.
class DmaBufferMemory final : public core::Memory {
public:
  DmaBufferMemory(const core::Shape& shape, const core::Stride& stride, DmaHeapUse use = DmaHeapUse::Device32);

  // Views a caller's dma-buf (decoder, camera, display) without owning `fd`, mapping it when `host` is null. Its
  // pages are taken to be reachable by 32-bit DMA, as media allocators provide.
  DmaBufferMemory(int fd, void* host, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  DmaBufferMemory(SharedPtr<BasePointer> pointer, DmaHeapUse use, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;

  // Whether every page lies below 4 GB, so devices with 32-bit DMA can address the buffer.
  KZ_NODISCARD auto below_4gb() const -> bool;

  static auto aligned_stride(const core::Shape& shape) -> core::Stride;

protected:
  auto device_to_cpu(core::Stream* stream) -> bool override;

  auto cpu_to_device(core::Stream* stream) -> bool override;

private:
  DmaHeapUse use_;
};

// Whether this process can allocate from a dma-heap; probed once.
auto dma_heap_available() -> bool;

} // namespace hrt::memory

#endif // HRT_MEMORY_DMA_BUFFER_MEMORY_HPP
