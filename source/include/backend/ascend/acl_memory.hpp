// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/ascend/acl_memory.hpp
// Description: Declares Ascend device and media memory and their copies.
// -----------------------------------------------------------------------------

#ifndef HRT_ASCEND_ACL_MEMORY_HPP
#define HRT_ASCEND_ACL_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::backend {

// Device-only memory. Acl uses aclrtMalloc; Dvpp uses the media allocator VPC
// requires and, without an explicit stride, pads rows to 16 elements.
class AclMemory final : public core::Memory {
public:
  AclMemory(core::MemoryType type, const core::Shape& shape, const core::Stride& stride);

  AclMemory(core::MemoryType type, std::uintptr_t device, std::size_t size, const core::Shape& shape,
            const core::Stride& stride);

  AclMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;

  // Rows padded to 16 elements, which VPC requires of the images it reads and writes.
  static auto dvpp_stride(const core::Shape& shape) -> core::Stride;
};

// Copies between host, Acl and Dvpp memory on `stream`; one side must be
// Ascend memory.
auto copy_acl(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool;

} // namespace hrt::backend

#endif // HRT_ASCEND_ACL_MEMORY_HPP
