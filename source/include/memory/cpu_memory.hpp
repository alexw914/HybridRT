// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/cpu_memory.hpp
// Description: Declares host memory.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_CPU_MEMORY_HPP
#define HRT_MEMORY_CPU_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::memory {

class CpuMemory final : public core::Memory {
public:
  // Owning; an empty stride takes aligned_stride().
  explicit CpuMemory(const core::Shape& shape = {}, const core::Stride& stride = {});

  CpuMemory(void* data, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

protected:
  auto reallocate(std::size_t size) -> bool override;
};

auto copy_cpu_to_cpu(core::Memory& dst, core::Memory& src, std::size_t axis) -> bool;

// Memory for the inputs of engines that read them on the host: DmaBuffer when this build has a preprocessor that writes
// dma-bufs (RGA) and a dma-heap is usable, so preprocessing runs there and fills them in place; Cpu otherwise.
auto host_engine_input_memory() -> core::MemoryType;

} // namespace hrt::memory

#endif // HRT_MEMORY_CPU_MEMORY_HPP
