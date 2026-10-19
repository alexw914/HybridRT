// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/base_pointer.hpp
// Description: Declares the allocation state shared by memory views.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_BASE_POINTER_HPP
#define HRT_MEMORY_BASE_POINTER_HPP

#include <hrt/core/memory.hpp>

#include <kitzoo/memory/bytebuffer.hpp>

#include <optional>

namespace hrt::memory {

struct BasePointer {
  explicit BasePointer(core::MemoryType memory_type) : type(memory_type) {}

  ~BasePointer() {
    if (release != nullptr)
      release(*this);
  }

  BasePointer(const BasePointer&) = delete;
  auto operator=(const BasePointer&) -> BasePointer& = delete;
  BasePointer(BasePointer&&) = delete;
  auto operator=(BasePointer&&) -> BasePointer& = delete;

  const core::MemoryType type;
  void* cpu = nullptr;
  std::uintptr_t device = 0;
  // Usable bytes; only owned storage can grow.
  std::size_t capacity = 0;
  bool owned = false;
  // DmaBuffer: every page lies below 4 GB, where devices with 32-bit DMA reach it.
  bool below_4gb = false;
  core::SyncedHead head = core::SyncedHead::Synced;
  kitzoo::memory::ByteBuffer host_storage;
  // Frees owned storage outside host_storage (device or pinned host); null
  // for host_storage and wrapped storage.
  void (*release)(BasePointer&) = nullptr;
};

// Validates `stride` against `shape`; an empty stride selects a packed one.
auto resolve_stride(const core::Shape& shape, const core::Stride& stride) -> std::optional<core::Stride>;

} // namespace hrt::memory

#endif // HRT_MEMORY_BASE_POINTER_HPP
