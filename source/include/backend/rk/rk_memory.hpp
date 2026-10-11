// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/rk/rk_memory.hpp
// Description: Declares Rockchip DMA-heap memory shared by RGA and RKNN.
// -----------------------------------------------------------------------------

#ifndef HRT_RK_RK_MEMORY_HPP
#define HRT_RK_RK_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::backend {

// RkFd: DMA-heap buffer mapped to the host; device() is the dma-buf fd.
// RkVa: caller host memory RGA reads by virtual address.
// Without an explicit stride, rows are aligned to 16 elements as RGA needs.
class RkMemory final : public core::Memory {
public:
  RkMemory(const core::Shape& shape, const core::Stride& stride);

  RkMemory(void* host, std::size_t size, const core::Shape& shape, const core::Stride& stride);

  RkMemory(SharedPtr<memory::BasePointer> pointer, core::MemoryType type, const core::Shape& shape,
           const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;

  static auto rga_stride(const core::Shape& shape) -> core::Stride;

protected:
  auto device_to_cpu(core::Stream* stream) -> bool override;

  auto cpu_to_device(core::Stream* stream) -> bool override;
};

} // namespace hrt::backend

#endif // HRT_RK_RK_MEMORY_HPP
