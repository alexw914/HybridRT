// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/ascend/acl_memory.cpp
// Description: Implements Ascend device and media memory and their copies.
// -----------------------------------------------------------------------------

#include "include/backend/ascend/acl_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <acl/dvpp/hi_dvpp.h>

#include "include/backend/ascend/acl_common.hpp"

namespace hrt::backend {

namespace {

auto is_ascend(core::MemoryType type) -> bool {
  return type == core::MemoryType::Acl || type == core::MemoryType::Dvpp;
}

auto acl_release(memory::BasePointer& pointer) -> void {
  if (pointer.device != 0 && aclrtFree(reinterpret_cast<void*>(pointer.device)) != ACL_SUCCESS)
    KZ_LOG_ERROR("acl: aclrtFree failed: {}", aclGetRecentErrMsg());
  pointer.device = 0;
  pointer.capacity = 0;
}

auto dvpp_release(memory::BasePointer& pointer) -> void {
  if (pointer.device != 0) {
    const auto status = hi_mpi_dvpp_free(reinterpret_cast<void*>(pointer.device));
    if (status != ACL_SUCCESS)
      KZ_LOG_ERROR("dvpp: hi_mpi_dvpp_free failed: 0x{:X}", static_cast<unsigned int>(status));
  }
  pointer.device = 0;
  pointer.capacity = 0;
}

auto ascend_allocate(memory::BasePointer& pointer, std::size_t size) -> bool {
  pointer.release(pointer);
  if (size == 0)
    return true;
  void* device = nullptr;
  // Both return a plain int, so the branches share a type without a cast.
  const int status = pointer.type == core::MemoryType::Dvpp ? hi_mpi_dvpp_malloc(0, &device, size)
                                                            : aclrtMalloc(&device, size, ACL_MEM_MALLOC_HUGE_FIRST);
  KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "acl: allocating {} bytes failed: 0x{:X} {}", size,
                  static_cast<unsigned int>(status), aclGetRecentErrMsg());
  pointer.device = reinterpret_cast<std::uintptr_t>(device);
  pointer.capacity = size;
  return true;
}

auto address(core::Memory& memory, bool write, core::Stream* stream) -> void* {
  if (is_ascend(memory.memory_type()))
    return reinterpret_cast<void*>(write ? memory.mutable_device(stream) : memory.device(stream));
  return write ? memory.mutable_cpu(stream) : const_cast<void*>(memory.cpu(stream));
}

} // namespace

auto AclMemory::dvpp_stride(const core::Shape& shape) -> core::Stride {
  if (shape.size() < 2)
    return core::contiguous_stride(shape);
  // Width is the second axis, or the third when the first is a single plane.
  auto aligned = shape;
  const auto axis = shape[0] == 1 && shape.size() > 2 ? 2U : 1U;
  aligned[axis] = (aligned[axis] + 15) / 16 * 16;
  return core::contiguous_stride(aligned);
}

AclMemory::AclMemory(core::MemoryType type, const core::Shape& shape, const core::Stride& stride) : core::Memory(type) {
  shape_ = shape;
  stride_ = stride;
  pointer_->owned = true;
  pointer_->release = type == core::MemoryType::Dvpp ? dvpp_release : acl_release;
  ascend_allocate(*pointer_, stride_.empty() ? 0 : stride_[0]);
}

AclMemory::AclMemory(core::MemoryType type, std::uintptr_t device, std::size_t size, const core::Shape& shape,
                     const core::Stride& stride)
    : core::Memory(type) {
  shape_ = shape;
  stride_ = stride;
  pointer_->device = device;
  pointer_->capacity = size;
}

AclMemory::AclMemory(SharedPtr<memory::BasePointer> pointer, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(std::move(pointer)) {
  shape_ = shape;
  stride_ = stride;
}

auto AclMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<AclMemory>(pointer_, shape_, stride_);
}

auto AclMemory::resize(const core::Shape& shape, const core::Stride& stride) -> bool {
  auto memory_stride = stride;
  if (memory_stride.empty())
    memory_stride = memory_type() == core::MemoryType::Dvpp ? dvpp_stride(shape) : core::contiguous_stride(shape);
  else if (!memory::check_stride(shape, memory_stride))
    return false;
  const auto size = memory_stride.front();
  if (size > pointer_->capacity) {
    if (!pointer_->owned) {
      KZ_LOG_ERROR("AclMemory: cannot grow wrapped memory from {} to {} bytes", pointer_->capacity, size);
      return false;
    }
    if (!ascend_allocate(*pointer_, size))
      return false;
  }
  shape_ = shape;
  stride_ = std::move(memory_stride);
  return true;
}

auto AclMemory::shrink_to_fit() -> bool {
  const auto size = stride_.empty() ? 0 : stride_[0];
  if (!pointer_->owned || size == pointer_->capacity)
    return true;
  return ascend_allocate(*pointer_, size);
}

auto copy_acl(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool {
  const auto to_device = is_ascend(dst.memory_type());
  const auto from_device = is_ascend(src.memory_type());
  const auto kind =
      to_device ? (from_device ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE) : ACL_MEMCPY_DEVICE_TO_HOST;
  auto* to = address(dst, true, stream);
  auto* from = address(src, false, stream);
  if (to == nullptr || from == nullptr)
    return false;

  const auto& shape = src.shape();
  std::size_t chunk = 1;
  for (auto i = axis; i < shape.size(); ++i)
    chunk *= static_cast<std::size_t>(shape[i]);
  const auto total = static_cast<std::size_t>(core::volume(shape));
  if (chunk == 0 || total == 0)
    return true;
  // Without a stream the copy runs synchronously.
  if (axis == 0) {
    const auto status = stream == nullptr ? aclrtMemcpy(to, dst.bytes(), from, total, kind)
                                          : aclrtMemcpyAsync(to, dst.bytes(), from, total, kind, acl_stream(stream));
    KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "acl: copy failed: {}", aclGetRecentErrMsg());
    return true;
  }
  const auto dst_pitch = dst.stride()[axis];
  const auto src_pitch = src.stride()[axis];
  const auto status = stream == nullptr ? aclrtMemcpy2d(to, dst_pitch, from, src_pitch, chunk, total / chunk, kind)
                                        : aclrtMemcpy2dAsync(to, dst_pitch, from, src_pitch, chunk, total / chunk, kind,
                                                             acl_stream(stream));
  KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "acl: 2D copy failed: {}", aclGetRecentErrMsg());
  return true;
}

} // namespace hrt::backend
