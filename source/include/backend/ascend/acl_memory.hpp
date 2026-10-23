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
  // Owning; an empty stride takes aligned_stride().
  explicit AclMemory(core::MemoryType type, const core::Shape& shape = {}, const core::Stride& stride = {});

  AclMemory(core::MemoryType type, std::uintptr_t device, std::size_t size, const core::Shape& shape,
            const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  // Dvpp rows are padded to 16 elements, which VPC requires of the images it reads and writes.
  KZ_NODISCARD auto aligned_stride(const core::Shape& shape) const -> core::Stride override;

protected:
  auto reallocate(std::size_t size) -> bool override;
};

// Copies between host, Acl and Dvpp memory on `stream`; one side must be
// Ascend memory.
auto copy_acl(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool;

} // namespace hrt::backend

#endif // HRT_ASCEND_ACL_MEMORY_HPP
