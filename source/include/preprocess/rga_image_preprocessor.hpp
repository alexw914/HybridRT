// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/rga_image_preprocessor.hpp
// Description: Declares the Rockchip RGA image operators.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>

namespace hrt::preprocess {

// Resize interleaved RGB/BGR uint8 dma-bufs on RGA3: rows must be a multiple
// of 16 pixels. Fill, other images, conversion and normalization stay on the
// CPU; host images are staged into dma-bufs first.
class RgaImagePreprocessor final : public ImagePreprocessor {
public:
  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Rga; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::DmaBuffer; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override;

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

protected:
  // RGA rejects region offsets of exactly 1.
  auto adjust_roi(core::Rect& dst_roi, core::Rect& src_roi, const core::Image& dst, const core::Image& src) const
      -> void override;

  // Scales outside RGA3's 1/8..8 range or sizes outside 128..8128 run on the CPU.
  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_RGA_IMAGE_PREPROCESSOR_HPP
