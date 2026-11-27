// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/dma_buffer_memory.hpp
// Description: Declares Linux dma-heap memory shared by media devices and the
//              host.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_DMA_BUFFER_MEMORY_HPP
#define HRT_MEMORY_DMA_BUFFER_MEMORY_HPP

#include <hrt/core/memory.hpp>

namespace hrt::memory {

// DmaBuffer memory: a dma-heap buffer mapped to the host; device() is the dma-buf fd. Without an explicit stride, rows
// are aligned to 16 elements as media devices need.
class DmaBufferMemory final : public core::Memory {
public:
  explicit DmaBufferMemory(const core::Shape& shape = {}, const core::Stride& stride = {});
  // Views a caller's dma-buf (decoder, camera, display) without owning `fd`, mapping it when `host` is null.
  DmaBufferMemory(int fd, void* host, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  KZ_NODISCARD auto aligned_stride(const core::Shape& shape) const -> core::Stride override;

protected:
  auto reallocate(std::size_t size) -> bool override;

  auto device_to_cpu(core::Stream* stream) -> bool override;

  auto cpu_to_device(core::Stream* stream) -> bool override;
};

// Whether this process can allocate from a dma-heap; probed once.
auto dma_heap_available() -> bool;

} // namespace hrt::memory

#endif // HRT_MEMORY_DMA_BUFFER_MEMORY_HPP
