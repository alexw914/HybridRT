// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/cuda/cuda_host_memory.hpp
// Description: Declares CUDA page-locked host memory.
// -----------------------------------------------------------------------------

#ifndef HRT_CUDA_CUDA_HOST_MEMORY_HPP
#define HRT_CUDA_CUDA_HOST_MEMORY_HPP

#include <hrt/core/memory.hpp>

namespace hrt::backend {

// Empty MemoryType::Cpu memory that allocates with cudaHostAlloc when resized,
// so CUDA copies to and from it run at full bandwidth without blocking the host.
auto create_cuda_host_memory() -> SharedPtr<core::Memory>;

} // namespace hrt::backend

#endif // HRT_CUDA_CUDA_HOST_MEMORY_HPP
