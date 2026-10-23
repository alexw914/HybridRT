// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/memory.hpp
// Description: Declares host/device memory with shape, stride and lazy
//              host-device synchronization.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_MEMORY_HPP
#define HRT_CORE_MEMORY_HPP

#include <hrt/core/context.hpp>
#include <hrt/core/define.hpp>

#include <kitzoo/core/macro.hpp>

#include <cstddef>
#include <cstdint>

namespace hrt::memory {
struct BasePointer;
} // namespace hrt::memory

namespace hrt::core {

enum class SyncedHead {
  AtCpu,
  AtDevice,
  Synced,
};

enum class CopyPolicy {
  KeepSrcStride, // dst takes src's shape and stride.
  AutoDstStride, // dst takes src's shape with a packed stride.
  DirectCopy,    // dst already has src's shape; only data is copied.
};

// Shape and stride describe bytes: the last dimension is in bytes and
// stride[0] is the byte size of the whole region. Views from ref() share the
// allocation and its sync state.
class Memory {
public:
  static auto create(MemoryType type, std::size_t size = 0) -> UniquePtr<Memory>;
  static auto create(MemoryType type, const Shape& shape, const Stride& stride = {}) -> UniquePtr<Memory>;

  // Views external storage without owning it; `size` bytes must stay valid. For DmaBuffer, `device` is the dma-buf fd
  // (still the caller's to close) and `host` its mapping, or null to map it here.
  static auto wrap(MemoryType type, void* host, std::uintptr_t device, std::size_t size, const Shape& shape = {},
                   const Stride& stride = {}) -> UniquePtr<Memory>;

  static auto copy(Memory& dst, Memory& src, CopyPolicy policy, Stream* stream = nullptr) -> bool;

  virtual ~Memory();

  auto operator=(const Memory&) -> Memory& = delete;
  Memory(Memory&&) = delete;
  auto operator=(Memory&&) -> Memory& = delete;

  virtual auto ref() -> SharedPtr<Memory> = 0;

  KZ_NODISCARD auto have_cpu() const -> bool;

  KZ_NODISCARD auto have_device() const -> bool;

  // Accessors sync the requested side first; mutable access marks it newest.
  // They return null (0) when that side does not exist.
  auto cpu(Stream* stream = nullptr) -> const void*;

  auto mutable_cpu(Stream* stream = nullptr) -> void*;

  auto device(Stream* stream = nullptr) -> std::uintptr_t;

  auto mutable_device(Stream* stream = nullptr) -> std::uintptr_t;

  KZ_NODISCARD auto bytes() const -> std::size_t;

  KZ_NODISCARD auto dims() const -> std::size_t { return shape_.size(); }

  KZ_NODISCARD auto shape() const -> const Shape& { return shape_; }

  KZ_NODISCARD auto stride() const -> const Stride& { return stride_; }

  KZ_NODISCARD auto is_contiguous() const -> bool;

  // Default stride of this memory type: packed, or rows padded to what its devices address.
  KZ_NODISCARD virtual auto aligned_stride(const Shape& shape) const -> Stride;

  // An empty stride takes aligned_stride(). Keeps the allocation when it is large enough; wrapped memory cannot
  // grow. Contents are undefined after the allocation grows or shrinks.
  auto resize(const Shape& shape, const Stride& stride = {}) -> bool;

  auto shrink_to_fit() -> bool;

  KZ_NODISCARD auto memory_type() const -> MemoryType;

  KZ_NODISCARD auto head() const -> SyncedHead;

  auto sync_to_cpu(Stream* stream = nullptr) -> bool;

  auto sync_to_device(Stream* stream = nullptr) -> bool;

protected:
  explicit Memory(MemoryType type);

  // Copies share the allocation and its sync state, as ref() views do.
  Memory(const Memory&) = default;

  // Replaces owned storage with exactly `size` bytes.
  virtual auto reallocate(std::size_t size) -> bool = 0;

  virtual auto device_to_cpu(Stream* stream) -> bool;

  virtual auto cpu_to_device(Stream* stream) -> bool;

  SharedPtr<memory::BasePointer> pointer_;
  Shape shape_;
  Stride stride_;
};

auto device_type_of(MemoryType type) -> DeviceType;

} // namespace hrt::core

#endif // HRT_CORE_MEMORY_HPP
