// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/memory/memory_copier.cpp
// Description: Implements stride validation and copy dispatch between memory
//              types.
// -----------------------------------------------------------------------------

#include "include/memory/memory_copier.hpp"

#include <kitzoo/log/logger.hpp>

#include "include/memory/cpu_memory.hpp"

#ifdef HRT_USE_CUDA
#include "include/backend/cuda/cuda_memory.hpp"
#endif
#ifdef HRT_USE_CANN
#include "include/backend/ascend/acl_memory.hpp"
#endif

namespace hrt::memory {

namespace {

// The single padded axis of a region (0 when packed); empty for more than one.
auto discontinuous_axis(const core::Shape& shape, const core::Stride& stride) -> std::optional<std::size_t> {
  const auto dims = shape.size();
  std::size_t axis = 0;
  if (static_cast<std::size_t>(shape[dims - 1]) != stride[dims - 1])
    axis = dims - 1;
  for (auto i = dims - 1; i > 1; --i) {
    if (stride[i - 1] != static_cast<std::size_t>(shape[i - 1]) * stride[i]) {
      if (axis != 0)
        return std::nullopt;
      axis = i - 1;
    }
  }
  return axis;
}

} // namespace

auto check_stride(const core::Shape& shape, const core::Stride& stride) -> bool {
  KZ_CHECK_RETURN(!shape.empty() && !core::is_dynamic(shape), false, "Memory: shape must be non-empty and static");
  KZ_CHECK_RETURN(stride.size() == shape.size(), false, "Memory: stride rank {} does not match shape rank {}",
                  stride.size(), shape.size());
  const auto packed = core::contiguous_stride(shape);
  for (std::size_t i = 0; i < stride.size(); ++i) {
    KZ_CHECK_RETURN(stride[i] >= packed[i], false, "Memory: stride[{}] = {} is smaller than packed {}", i, stride[i],
                    packed[i]);
  }
  return true;
}

auto contiguous_axis(const core::Memory& dst, const core::Memory& src) -> std::optional<std::size_t> {
  auto dst_axis = discontinuous_axis(dst.shape(), dst.stride());
  auto src_axis = discontinuous_axis(src.shape(), src.stride());
  if (!dst_axis || !src_axis)
    return std::nullopt;
  if (*dst_axis == *src_axis || *src_axis == 0)
    return dst_axis;
  if (*dst_axis == 0)
    return src_axis;
  return std::nullopt;
}

auto copy_memory(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool {
  // Device copies first, keyed by memory type; host-visible pairs use memcpy.
  static_cast<void>(stream);
#ifdef HRT_USE_CUDA
  if (dst.memory_type() == core::MemoryType::Cuda || src.memory_type() == core::MemoryType::Cuda)
    return backend::copy_cuda(dst, src, axis, stream);
#endif
#ifdef HRT_USE_CANN
  for (auto type : {dst.memory_type(), src.memory_type()}) {
    if (type == core::MemoryType::Acl || type == core::MemoryType::Dvpp)
      return backend::copy_acl(dst, src, axis, stream);
  }
#endif
  if (dst.have_cpu() && src.have_cpu())
    return copy_cpu_to_cpu(dst, src, axis);
  KZ_LOG_ERROR("Memory: no copy from {} to {}", to_string(src.memory_type()), to_string(dst.memory_type()));
  return false;
}

} // namespace hrt::memory
