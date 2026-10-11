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
  // Owning; `stride` must already be validated.
  CpuMemory(const core::Shape& shape, const core::Stride& stride);

  CpuMemory(void* data, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  CpuMemory(SharedPtr<BasePointer> pointer, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;
};

auto copy_cpu_to_cpu(core::Memory& dst, core::Memory& src, std::size_t axis) -> bool;

} // namespace hrt::memory

#endif // HRT_MEMORY_CPU_MEMORY_HPP
