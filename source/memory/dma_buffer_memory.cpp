// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/memory/dma_buffer_memory.cpp
// Description: Implements Linux dma-heap memory.
// -----------------------------------------------------------------------------

#include "include/memory/dma_buffer_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "include/memory/base_pointer.hpp"

namespace hrt::memory {

namespace {

// Kernel uapi for dma-heap and dma-buf, mirrored to avoid header version skew.
struct DmaHeapAllocation {
  __u64 len;
  __u32 fd;
  __u32 fd_flags;
  __u64 heap_flags;
};

struct DmaBufferSync {
  __u64 flags;
};

constexpr unsigned long kDmaHeapAlloc = _IOWR('H', 0x0, DmaHeapAllocation);
constexpr unsigned long kDmaBufferSync = _IOW('b', 0, DmaBufferSync);
constexpr __u64 kSyncReadWrite = 3;
constexpr __u64 kSyncStart = 0;
constexpr __u64 kSyncEnd = 4;

// Prefer the system heap (full RAM behind IOMMU for RK3588 NPU and RGA3); fall back to 32-bit heaps if absent.
constexpr Array<const char*, 3> kHeaps{"/dev/dma_heap/system", "/dev/dma_heap/system-dma32", "/dev/dma_heap/cma"};

auto align(std::size_t value, std::size_t alignment) -> std::size_t {
  return (value + alignment - 1) / alignment * alignment;
}

auto dma_release(BasePointer& pointer) -> void {
  if (pointer.cpu_ != nullptr)
    munmap(pointer.cpu_, pointer.capacity_);
  if (pointer.device_ != 0)
    close(static_cast<int>(pointer.device_));
  pointer.cpu_ = nullptr;
  pointer.device_ = 0;
  pointer.capacity_ = 0;
}

// Wrapped dma-bufs keep the caller's fd and drop only a mapping made here.
auto unmap_release(BasePointer& pointer) -> void {
  if (pointer.cpu_ != nullptr)
    munmap(pointer.cpu_, pointer.capacity_);
  pointer.cpu_ = nullptr;
}

// Returns the dma-buf fd, or -1 with errno set.
auto allocate_from(const char* path, std::size_t size) -> int {
  const int heap = open(path, O_RDWR | O_CLOEXEC);
  if (heap < 0)
    return -1;
  DmaHeapAllocation request{};
  request.len = size;
  request.fd_flags = O_CLOEXEC | O_RDWR;
  const auto allocated = ioctl(heap, kDmaHeapAlloc, &request);
  // close() and the logger may overwrite errno.
  const auto error = errno;
  close(heap);
  errno = error;
  return allocated < 0 ? -1 : static_cast<int>(request.fd);
}

auto dma_sync(const BasePointer& pointer, __u64 flags) -> bool {
  if (pointer.device_ == 0)
    return true;
  DmaBufferSync sync{flags | kSyncReadWrite};
  if (ioctl(static_cast<int>(pointer.device_), kDmaBufferSync, &sync) == 0)
    return true;
  KZ_LOG_ERROR("dma-buf: sync failed: {}", std::strerror(errno));
  return false;
}

} // namespace

auto DmaBufferMemory::aligned_stride(const core::Shape& shape) const -> core::Stride {
  auto stride = core::contiguous_stride(shape);
  if (shape.size() < 2)
    return stride;
  const auto last = shape.size() - 1;
  // Planar/gray/2D layouts end with width; interleaved layouts end with pixel bytes (<= 4).
  const auto axis = shape[last] > 4 && (shape.size() == 2 || shape[last - 2] <= 4) ? last : last - 1;
  auto aligned = shape;
  aligned[axis] = static_cast<std::int64_t>(align(static_cast<std::size_t>(aligned[axis]), 16));
  return core::contiguous_stride(aligned);
}

DmaBufferMemory::DmaBufferMemory(const core::Shape& shape, const core::Stride& stride)
    : core::Memory(core::MemoryType::DmaBuffer) {
  pointer_->owned_ = true;
  pointer_->release_ = dma_release;
  resize(shape, stride);
}

DmaBufferMemory::DmaBufferMemory(int fd, void* host, std::size_t size, const core::Shape& shape,
                                 const core::Stride& stride)
    : core::Memory(core::MemoryType::DmaBuffer) {
  shape_ = shape;
  stride_ = stride;
  pointer_->device_ = static_cast<std::uintptr_t>(fd);
  pointer_->capacity_ = size;
  if (host != nullptr) {
    pointer_->cpu_ = host;
    return;
  }
  auto* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    KZ_LOG_ERROR("dma-buf: cannot map fd {}: {}", fd, std::strerror(errno));
    return;
  }
  pointer_->cpu_ = mapped;
  pointer_->release_ = unmap_release;
}

auto DmaBufferMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<DmaBufferMemory>(*this);
}

auto DmaBufferMemory::reallocate(std::size_t size) -> bool {
  dma_release(*pointer_);
  if (size == 0)
    return true;
  Vector<String> tried;
  int error = ENOENT;
  for (const auto* path : kHeaps) {
    char resolved[PATH_MAX];
    if (realpath(path, resolved) == nullptr)
      continue;
    if (std::find(tried.begin(), tried.end(), resolved) != tried.end())
      continue;
    tried.emplace_back(resolved);
    const int fd = allocate_from(path, size);
    if (fd < 0) {
      error = errno;
      continue;
    }
    auto* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
      error = errno;
      close(fd);
      continue;
    }
    pointer_->cpu_ = mapped;
    pointer_->device_ = static_cast<std::uintptr_t>(fd);
    pointer_->capacity_ = size;
    return true;
  }
  KZ_LOG_ERROR("dma-buf: allocation of {} bytes failed in every dma-heap: {}", size, std::strerror(error));
  return false;
}

// CPU access starts a dma-buf access window; device access ends it.
auto DmaBufferMemory::device_to_cpu(core::Stream* /*stream*/) -> bool {
  return dma_sync(*pointer_, kSyncStart);
}

auto DmaBufferMemory::cpu_to_device(core::Stream* /*stream*/) -> bool {
  return dma_sync(*pointer_, kSyncEnd);
}

auto dma_heap_available() -> bool {
  static const bool available = []() -> bool {
    for (const auto* path : kHeaps) {
      const int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd >= 0) {
        close(fd);
        return true;
      }
    }
    return false;
  }();
  return available;
}

} // namespace hrt::memory
