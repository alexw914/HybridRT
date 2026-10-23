// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/core/batch.hpp
// Description: Declares views of single samples of batched buffers.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_BATCH_HPP
#define HRT_CORE_BATCH_HPP

#include <hrt/core/buffer.hpp>

#include <cstddef>

namespace hrt::core {

// Sample `index` along the leading batch dim of `buffer`, as a buffer of shape
// {1, ...} sharing its storage and quantization. Host-visible memory is viewed
// on the host, device-only memory (CUDA, Ascend) through its device address;
// a dma-buf sample is therefore host memory. `write` marks the side being
// written newest instead of syncing it first. Null when `index` is out of range.
auto batch_sample(const Buffer& buffer, std::size_t index, bool write) -> SharedPtr<Buffer>;

} // namespace hrt::core

#endif // HRT_CORE_BATCH_HPP
