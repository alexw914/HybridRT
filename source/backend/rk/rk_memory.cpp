// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/rk/rk_memory.cpp
// Description: Implements Rockchip DMA-heap memory.
// -----------------------------------------------------------------------------

#include "include/backend/rk/rk_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace hrt::backend {

namespace {

// Kernel uapi for dma-heap and dma-buf, mirrored to avoid header version skew.
struct DmaHeapAllocation {
  __u64 len;
  __u32 fd;
  __u32 fd_flags;
  __u64 heap_flags;
};

struct DmaBufSync {
  __u64 flags;
};

constexpr const char* kDmaHeapPath = "/dev/dma_heap/cma";
constexpr unsigned long kDmaHeapAlloc = _IOWR('H', 0x0, DmaHeapAllocation);
constexpr unsigned long kDmaBufSync = _IOW('b', 0, DmaBufSync);
constexpr __u64 kSyncReadWrite = 3;
constexpr __u64 kSyncStart = 0;
constexpr __u64 kSyncEnd = 4;

auto align(std::size_t value, std::size_t alignment) -> std::size_t {
  return (value + alignment - 1) / alignment * alignment;
}

auto dma_release(memory::BasePointer& pointer) -> void {
  if (pointer.cpu != nullptr)
    munmap(pointer.cpu, pointer.capacity);
  if (pointer.device != 0)
    close(static_cast<int>(pointer.device));
  pointer.cpu = nullptr;
  pointer.device = 0;
  pointer.capacity = 0;
}

auto dma_allocate(memory::BasePointer& pointer, std::size_t size) -> bool {
  dma_release(pointer);
  if (size == 0)
    return true;
  const int heap = open(kDmaHeapPath, O_RDWR);
  if (heap < 0) {
    KZ_LOG_ERROR("rk: cannot open {}: {}", kDmaHeapPath, std::strerror(errno));
    return false;
  }
  DmaHeapAllocation request{};
  request.len = size;
  request.fd_flags = O_CLOEXEC | O_RDWR;
  const auto allocated = ioctl(heap, kDmaHeapAlloc, &request);
  close(heap);
  if (allocated < 0) {
    KZ_LOG_ERROR("rk: dma-heap allocation of {} bytes failed: {}", size, std::strerror(errno));
    return false;
  }
  const auto fd = static_cast<int>(request.fd);
  auto* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    KZ_LOG_ERROR("rk: mmap failed: {}", std::strerror(errno));
    close(fd);
    return false;
  }
  pointer.cpu = mapped;
  pointer.device = static_cast<std::uintptr_t>(fd);
  pointer.capacity = size;
  return true;
}

auto dma_sync(const memory::BasePointer& pointer, __u64 flags) -> bool {
  if (pointer.device == 0)
    return true;
  DmaBufSync sync{flags | kSyncReadWrite};
  if (ioctl(static_cast<int>(pointer.device), kDmaBufSync, &sync) == 0)
    return true;
  KZ_LOG_ERROR("rk: dma-buf sync failed: {}", std::strerror(errno));
  return false;
}

} // namespace

auto RkMemory::rga_stride(const core::Shape& shape) -> core::Stride {
  auto stride = core::contiguous_stride(shape);
  if (shape.size() < 2)
    return stride;
  const auto last = shape.size() - 1;
  stride[last] = static_cast<std::size_t>(shape[last]);
  stride[last - 1] = stride[last] * align(static_cast<std::size_t>(shape[last - 1]), 16);
  for (auto i = last - 1; i > 0; --i)
    stride[i - 1] = stride[i] * static_cast<std::size_t>(shape[i - 1]);
  return stride;
}

RkMemory::RkMemory(const core::Shape& shape, const core::Stride& stride) : core::Memory(core::MemoryType::RkFd) {
  shape_ = shape;
  stride_ = stride;
  pointer_->owned = true;
  pointer_->release = dma_release;
  dma_allocate(*pointer_, stride_.empty() ? 0 : stride_[0]);
}

RkMemory::RkMemory(void* host, std::size_t size, const core::Shape& shape, const core::Stride& stride)
    : core::Memory(core::MemoryType::RkVa) {
  shape_ = shape;
  stride_ = stride;
  pointer_->cpu = host;
  pointer_->capacity = size;
}

RkMemory::RkMemory(SharedPtr<memory::BasePointer> pointer, core::MemoryType /*type*/, const core::Shape& shape,
                   const core::Stride& stride)
    : core::Memory(std::move(pointer)) {
  shape_ = shape;
  stride_ = stride;
}

auto RkMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<RkMemory>(pointer_, memory_type(), shape_, stride_);
}

auto RkMemory::resize(const core::Shape& shape, const core::Stride& stride) -> bool {
  auto resolved = memory::resolve_stride(shape, stride.empty() ? rga_stride(shape) : stride);
  if (!resolved)
    return false;
  const auto size = resolved->front();
  if (size > pointer_->capacity) {
    if (!pointer_->owned) {
      KZ_LOG_ERROR("RkMemory: cannot grow wrapped memory from {} to {} bytes", pointer_->capacity, size);
      return false;
    }
    if (!dma_allocate(*pointer_, size))
      return false;
  }
  shape_ = shape;
  stride_ = std::move(*resolved);
  return true;
}

auto RkMemory::shrink_to_fit() -> bool {
  const auto size = stride_.empty() ? 0 : stride_[0];
  if (!pointer_->owned || size == pointer_->capacity)
    return true;
  return dma_allocate(*pointer_, size);
}

// CPU access starts a dma-buf access window; device access ends it.
auto RkMemory::device_to_cpu(core::Stream* /*stream*/) -> bool {
  return dma_sync(*pointer_, kSyncStart);
}

auto RkMemory::cpu_to_device(core::Stream* /*stream*/) -> bool {
  return dma_sync(*pointer_, kSyncEnd);
}

} // namespace hrt::backend
