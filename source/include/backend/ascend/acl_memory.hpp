// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/ascend/acl_memory.hpp
// Description: Declares Ascend device memory and its copies.
// -----------------------------------------------------------------------------

#ifndef HRT_ASCEND_ACL_MEMORY_HPP
#define HRT_ASCEND_ACL_MEMORY_HPP

#include "include/memory/base_pointer.hpp"

namespace hrt::backend {

// Device-only memory from aclrtMalloc, contiguous unless a caller passes a
// stride. DVPP skips images whose rows it cannot address; see is_vpc_image.
class AclMemory final : public core::Memory {
public:
  AclMemory(core::MemoryType type, const core::Shape& shape, const core::Stride& stride);

  AclMemory(core::MemoryType type, std::uintptr_t device, std::size_t size, const core::Shape& shape,
            const core::Stride& stride);

  AclMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride);

  auto ref() -> SharedPtr<core::Memory> override;

  auto resize(const core::Shape& shape, const core::Stride& stride) -> bool override;

  auto shrink_to_fit() -> bool override;
};

// Copies between host and Acl memory on `stream`; one side must be Acl memory.
auto copy_acl(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool;

} // namespace hrt::backend

#endif // HRT_ASCEND_ACL_MEMORY_HPP
