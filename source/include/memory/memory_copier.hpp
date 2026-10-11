// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/memory/memory_copier.hpp
// Description: Declares the strided copy dispatch between memory types.
// -----------------------------------------------------------------------------

#ifndef HRT_MEMORY_MEMORY_COPIER_HPP
#define HRT_MEMORY_MEMORY_COPIER_HPP

#include <hrt/core/memory.hpp>

#include <optional>

namespace hrt::memory {

// The axis above which both regions are contiguous: 0 when both are fully
// packed. Empty when the regions have different ranks or more than one
// padded axis.
auto contiguous_axis(const core::Memory& dst, const core::Memory& src) -> std::optional<std::size_t>;

// Copies src into dst, which must already have src's shape.
auto copy_memory(core::Memory& dst, core::Memory& src, std::size_t axis, core::Stream* stream) -> bool;

} // namespace hrt::memory

#endif // HRT_MEMORY_MEMORY_COPIER_HPP
