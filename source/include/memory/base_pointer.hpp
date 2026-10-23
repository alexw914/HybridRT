// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/base_pointer.hpp
// Description: Declares the allocation state shared by memory views.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_BASE_POINTER_HPP
#define HRT_MEMORY_BASE_POINTER_HPP

#include <hrt/core/memory.hpp>

#include <kitzoo/memory/bytebuffer.hpp>

namespace hrt::memory {

struct BasePointer {
  ~BasePointer() {
    if (release_ != nullptr)
      release_(*this);
  }

  BasePointer(const BasePointer&) = delete;
  auto operator=(const BasePointer&) -> BasePointer& = delete;
  BasePointer(BasePointer&&) = delete;
  auto operator=(BasePointer&&) -> BasePointer& = delete;

  explicit BasePointer(core::MemoryType memory_type) : memory_type_(memory_type) {}

  const core::MemoryType memory_type_;
  void* cpu_ = nullptr;
  std::uintptr_t device_ = 0;
  // Usable bytes; only owned storage can grow.
  std::size_t capacity_ = 0;
  bool owned_ = false;
  // DmaBuffer: every page lies below 4 GB, where devices with 32-bit DMA reach it.
  bool below_4gb_ = false;
  core::SyncedHead head_ = core::SyncedHead::Synced;
  kitzoo::memory::ByteBuffer host_storage_;
  // Frees owned storage outside host_storage_ (device or pinned host); null
  // for host_storage_ and wrapped storage.
  void (*release_)(BasePointer&) = nullptr;
};

// Checks that the byte stride matches the non-empty static shape.
auto check_stride(const core::Shape& shape, const core::Stride& stride) -> bool;

} // namespace hrt::memory

#endif // HRT_MEMORY_BASE_POINTER_HPP
