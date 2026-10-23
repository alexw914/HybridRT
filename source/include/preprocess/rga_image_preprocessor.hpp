// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/rga_image_preprocessor.hpp
// Description: Declares the Rockchip RGA image operators.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP

#include <kitzoo/core/macro.hpp>

#include "include/preprocess/cpu_image_preprocessor.hpp"

namespace hrt::preprocess {

// Fill and resize interleaved RGB/BGR uint8 dma-bufs on the RGA: rows must be
// a multiple of 16 pixels and pages below 4 GB (RGA2 has 32-bit DMA). Other
// images, conversion and normalization stay on the CPU; host images are staged
// into dma-bufs first.
class RgaImagePreprocessor final : public ImagePreprocessor {
public:
  // RGA rejects region offsets of exactly 1.
  auto adjust_roi(core::Rect& dst_roi, core::Rect& src_roi, const core::Image& dst, const core::Image& src) const
      -> void override;

  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  // Scales outside RGA's 1/16..16 range run on the CPU.
  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Rga; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::DmaBuffer; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override;

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

private:
  CpuImagePreprocessor cpu_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP
