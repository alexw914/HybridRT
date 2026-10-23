// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/memory.cpp
// Description: Implements memory creation, synchronization and copies.
// -----------------------------------------------------------------------------

#include <hrt/core/memory.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/memory/base_pointer.hpp"
#include "include/memory/cpu_memory.hpp"
#include "include/memory/memory_copier.hpp"

#ifdef HRT_USE_CUDA
#include "include/backend/cuda/cuda_memory.hpp"
#endif
#ifdef HRT_USE_DMA_BUFFER
#include "include/memory/dma_buffer_memory.hpp"
#endif
#ifdef HRT_USE_CANN

#include "include/backend/ascend/acl_memory.hpp"
#endif

namespace hrt::core {

auto device_type_of(MemoryType type) -> DeviceType {
  switch (type) {
  case MemoryType::Cuda:
    return DeviceType::Cuda;
  case MemoryType::Acl:
  case MemoryType::Dvpp:
    return DeviceType::Ascend;
  case MemoryType::Cpu:
  case MemoryType::DmaBuffer:
  case MemoryType::Unknown:
    break;
  }
  return DeviceType::Cpu;
}

auto Memory::create(MemoryType type, std::size_t size) -> UniquePtr<Memory> {
  return create(type, Shape{static_cast<std::int64_t>(size)});
}

auto Memory::create(MemoryType type, const Shape& shape, const Stride& stride) -> UniquePtr<Memory> {
  UniquePtr<Memory> memory;
  switch (type) {
  case MemoryType::Cpu:
    memory = std::make_unique<memory::CpuMemory>();
    break;
#ifdef HRT_USE_CUDA
  case MemoryType::Cuda:
    memory = std::make_unique<backend::CudaMemory>();
    break;
#endif
#ifdef HRT_USE_DMA_BUFFER
  case MemoryType::DmaBuffer:
    memory = std::make_unique<memory::DmaBufferMemory>();
    break;
#endif
#ifdef HRT_USE_CANN
  case MemoryType::Acl:
  case MemoryType::Dvpp:
    memory = std::make_unique<backend::AclMemory>(type);
    break;
#endif
  default:
    KZ_LOG_ERROR("Memory: type {} is not supported in this build", to_string(type));
    return nullptr;
  }
  if (!memory->resize(shape, stride))
    return nullptr;
  return memory;
}

auto Memory::wrap(MemoryType type, void* host, KZ_MAYBE_UNUSED std::uintptr_t device, std::size_t size,
                  const Shape& shape, const Stride& stride) -> UniquePtr<Memory> {
  auto view_shape = shape.empty() ? Shape{static_cast<std::int64_t>(size)} : shape;
  auto memory_stride = stride;
  if (memory_stride.empty())
    memory_stride = contiguous_stride(view_shape);
  else if (!memory::check_stride(view_shape, memory_stride))
    return nullptr;
  if (memory_stride.front() > size) {
    KZ_LOG_ERROR("Memory: wrapped size {} is smaller than the shape needs ({})", size, memory_stride.front());
    return nullptr;
  }
  switch (type) {
  case MemoryType::Cpu:
    if (host == nullptr)
      break;
    return std::make_unique<memory::CpuMemory>(host, size, view_shape, memory_stride);
#ifdef HRT_USE_CUDA
  case MemoryType::Cuda:
    if (device == 0)
      break;
    return std::make_unique<backend::CudaMemory>(device, size, view_shape, memory_stride);
#endif
#ifdef HRT_USE_DMA_BUFFER
  case MemoryType::DmaBuffer: {
    // `device` is the dma-buf fd; `host` an existing mapping or null.
    if (device == 0)
      break;
    auto memory =
        std::make_unique<memory::DmaBufferMemory>(static_cast<int>(device), host, size, view_shape, memory_stride);
    if (!memory->have_cpu())
      return nullptr;
    return memory;
  }
#endif
#ifdef HRT_USE_CANN
  case MemoryType::Acl:
  case MemoryType::Dvpp:
    if (device == 0)
      break;
    return std::make_unique<backend::AclMemory>(type, device, size, view_shape, memory_stride);
#endif
  default:
    break;
  }
  KZ_LOG_ERROR("Memory: cannot wrap {} storage", to_string(type));
  return nullptr;
}

auto Memory::copy(Memory& dst, Memory& src, CopyPolicy policy, Stream* stream) -> bool {
  switch (policy) {
  case CopyPolicy::AutoDstStride:
    if (!dst.resize(src.shape()))
      return false;
    break;
  case CopyPolicy::KeepSrcStride:
    if (!dst.resize(src.shape(), src.stride()))
      return false;
    break;
  case CopyPolicy::DirectCopy:
    if (dst.shape() != src.shape()) {
      KZ_LOG_ERROR("Memory: DirectCopy needs equal shapes");
      return false;
    }
    break;
  }
  auto axis = memory::contiguous_axis(dst, src);
  if (!axis) {
    KZ_LOG_ERROR("Memory: layouts are too fragmented to copy");
    return false;
  }
  return memory::copy_memory(dst, src, *axis, stream);
}

Memory::Memory(MemoryType type) : pointer_(std::make_shared<memory::BasePointer>(type)) {}

Memory::~Memory() = default;

auto Memory::have_cpu() const -> bool {
  return pointer_->cpu_ != nullptr;
}

auto Memory::have_device() const -> bool {
  return pointer_->device_ != 0;
}

auto Memory::cpu(Stream* stream) -> const void* {
  if (!have_cpu() || (pointer_->head_ == SyncedHead::AtDevice && !sync_to_cpu(stream)))
    return nullptr;
  return pointer_->cpu_;
}

auto Memory::mutable_cpu(Stream* stream) -> void* {
  if (cpu(stream) == nullptr)
    return nullptr;
  pointer_->head_ = SyncedHead::AtCpu;
  return pointer_->cpu_;
}

auto Memory::device(Stream* stream) -> std::uintptr_t {
  if (!have_device() || (pointer_->head_ == SyncedHead::AtCpu && !sync_to_device(stream)))
    return 0;
  return pointer_->device_;
}

auto Memory::mutable_device(Stream* stream) -> std::uintptr_t {
  if (device(stream) == 0)
    return 0;
  pointer_->head_ = SyncedHead::AtDevice;
  return pointer_->device_;
}

auto Memory::bytes() const -> std::size_t {
  return stride_.empty() ? 0 : stride_.front();
}

auto Memory::aligned_stride(const Shape& shape) const -> Stride {
  return contiguous_stride(shape);
}

auto Memory::resize(const Shape& shape, const Stride& stride) -> bool {
  Stride memory_stride;
  if (stride.empty()) {
    memory_stride = aligned_stride(shape);
  } else {
    if (shape == shape_ && stride == stride_ && stride.front() <= pointer_->capacity_)
      return true;
    if (!memory::check_stride(shape, stride))
      return false;
    memory_stride = stride;
  }
  const auto size = memory_stride.empty() ? 0 : memory_stride.front();
  if (size > pointer_->capacity_) {
    if (!pointer_->owned_) {
      KZ_LOG_ERROR("Memory: cannot grow wrapped {} memory from {} to {} bytes", to_string(memory_type()),
                   pointer_->capacity_, size);
      return false;
    }
    if (!reallocate(size))
      return false;
  }
  shape_ = shape;
  stride_ = std::move(memory_stride);
  return true;
}

auto Memory::shrink_to_fit() -> bool {
  const auto size = bytes();
  return !pointer_->owned_ || size == pointer_->capacity_ || reallocate(size);
}

auto Memory::is_contiguous() const -> bool {
  for (std::size_t i = 1; i < shape_.size(); ++i) {
    const auto inner = i + 1 < shape_.size() ? stride_[i + 1] : 1;
    if (stride_[i] != static_cast<std::size_t>(shape_[i]) * inner)
      return false;
  }
  return true;
}

auto Memory::memory_type() const -> MemoryType {
  return pointer_->memory_type_;
}

auto Memory::head() const -> SyncedHead {
  return pointer_->head_;
}

auto Memory::sync_to_cpu(Stream* stream) -> bool {
  if (!device_to_cpu(stream))
    return false;
  pointer_->head_ = SyncedHead::Synced;
  return true;
}

auto Memory::sync_to_device(Stream* stream) -> bool {
  if (!cpu_to_device(stream))
    return false;
  pointer_->head_ = SyncedHead::Synced;
  return true;
}

// Memory with a single side has nothing to sync; dual-sided types override.
auto Memory::device_to_cpu(Stream* /*stream*/) -> bool {
  return !(have_cpu() && have_device());
}

auto Memory::cpu_to_device(Stream* /*stream*/) -> bool {
  return !(have_cpu() && have_device());
}

} // namespace hrt::core
