// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/memory/dma_buffer_memory.cpp
// Description: Implements Linux dma-heap memory.
// -----------------------------------------------------------------------------

#include "include/memory/dma_buffer_memory.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

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

struct Heap {
  const char* path;
  bool below_4gb;
};

// Vendor kernels may link system-dma32 to cma; both are tried once.
constexpr Heap kSystemDma32{"/dev/dma_heap/system-dma32", true};
constexpr Heap kCma{"/dev/dma_heap/cma", true};
constexpr Heap kSystem{"/dev/dma_heap/system", false};
constexpr Array<Heap, 3> kDevice32Order{kSystemDma32, kCma, kSystem};
constexpr Array<Heap, 3> kAnyOrder{kSystem, kSystemDma32, kCma};

auto align(std::size_t value, std::size_t alignment) -> std::size_t {
  return (value + alignment - 1) / alignment * alignment;
}

auto dma_release(BasePointer& pointer) -> void {
  if (pointer.cpu != nullptr)
    munmap(pointer.cpu, pointer.capacity);
  if (pointer.device != 0)
    close(static_cast<int>(pointer.device));
  pointer.cpu = nullptr;
  pointer.device = 0;
  pointer.capacity = 0;
}

// Wrapped dma-bufs keep the caller's fd and drop only a mapping made here.
auto unmap_release(BasePointer& pointer) -> void {
  if (pointer.cpu != nullptr)
    munmap(pointer.cpu, pointer.capacity);
  pointer.cpu = nullptr;
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

auto dma_allocate(BasePointer& pointer, std::size_t size, DmaHeapUse use) -> bool {
  dma_release(pointer);
  if (size == 0)
    return true;
  const auto& order = use == DmaHeapUse::Device32 ? kDevice32Order : kAnyOrder;
  Vector<String> tried;
  int error = ENOENT;
  for (const auto& heap : order) {
    char resolved[PATH_MAX];
    if (realpath(heap.path, resolved) == nullptr)
      continue;
    if (std::find(tried.begin(), tried.end(), resolved) != tried.end())
      continue;
    tried.emplace_back(resolved);
    const int fd = allocate_from(heap.path, size);
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
    if (use == DmaHeapUse::Device32 && !heap.below_4gb) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true))
        KZ_LOG_WARN("dma-buf: 32-bit DMA heaps are exhausted; using {}, where devices with 32-bit DMA (RGA2) "
                    "leave the work to the CPU",
                    heap.path);
    }
    pointer.cpu = mapped;
    pointer.device = static_cast<std::uintptr_t>(fd);
    pointer.capacity = size;
    pointer.below_4gb = heap.below_4gb;
    return true;
  }
  KZ_LOG_ERROR("dma-buf: allocation of {} bytes failed in every dma-heap: {}", size, std::strerror(error));
  return false;
}

auto dma_sync(const BasePointer& pointer, __u64 flags) -> bool {
  if (pointer.device == 0)
    return true;
  DmaBufferSync sync{flags | kSyncReadWrite};
  if (ioctl(static_cast<int>(pointer.device), kDmaBufferSync, &sync) == 0)
    return true;
  KZ_LOG_ERROR("dma-buf: sync failed: {}", std::strerror(errno));
  return false;
}

} // namespace

auto DmaBufferMemory::aligned_stride(const core::Shape& shape) -> core::Stride {
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

DmaBufferMemory::DmaBufferMemory(const core::Shape& shape, const core::Stride& stride, DmaHeapUse use)
    : core::Memory(core::MemoryType::DmaBuffer), use_(use) {
  shape_ = shape;
  stride_ = stride;
  pointer_->owned = true;
  pointer_->release = dma_release;
  dma_allocate(*pointer_, stride_.empty() ? 0 : stride_[0], use_);
}

DmaBufferMemory::DmaBufferMemory(int fd, void* host, std::size_t size, const core::Shape& shape,
                                 const core::Stride& stride)
    : core::Memory(core::MemoryType::DmaBuffer), use_(DmaHeapUse::Device32) {
  shape_ = shape;
  stride_ = stride;
  pointer_->device = static_cast<std::uintptr_t>(fd);
  pointer_->capacity = size;
  pointer_->below_4gb = true;
  if (host != nullptr) {
    pointer_->cpu = host;
    return;
  }
  auto* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    KZ_LOG_ERROR("dma-buf: cannot map fd {}: {}", fd, std::strerror(errno));
    return;
  }
  pointer_->cpu = mapped;
  pointer_->release = unmap_release;
}

DmaBufferMemory::DmaBufferMemory(SharedPtr<BasePointer> pointer, DmaHeapUse use, const core::Shape& shape,
                                 const core::Stride& stride)
    : core::Memory(std::move(pointer)), use_(use) {
  shape_ = shape;
  stride_ = stride;
}

auto DmaBufferMemory::ref() -> SharedPtr<core::Memory> {
  return std::make_shared<DmaBufferMemory>(pointer_, use_, shape_, stride_);
}

auto DmaBufferMemory::resize(const core::Shape& shape, const core::Stride& stride) -> bool {
  auto resolved = resolve_stride(shape, stride.empty() ? aligned_stride(shape) : stride);
  if (!resolved)
    return false;
  const auto size = resolved->front();
  if (size > pointer_->capacity) {
    if (!pointer_->owned) {
      KZ_LOG_ERROR("DmaBufferMemory: cannot grow wrapped memory from {} to {} bytes", pointer_->capacity, size);
      return false;
    }
    if (!dma_allocate(*pointer_, size, use_))
      return false;
  }
  shape_ = shape;
  stride_ = std::move(*resolved);
  return true;
}

auto DmaBufferMemory::shrink_to_fit() -> bool {
  const auto size = stride_.empty() ? 0 : stride_[0];
  if (!pointer_->owned || size == pointer_->capacity)
    return true;
  return dma_allocate(*pointer_, size, use_);
}

auto DmaBufferMemory::below_4gb() const -> bool {
  return pointer_->below_4gb;
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
    for (const auto& heap : kDevice32Order) {
      const int fd = open(heap.path, O_RDWR | O_CLOEXEC);
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
