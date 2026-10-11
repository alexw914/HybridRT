// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/dvpp_image_preprocessor.hpp
// Description: Declares the Ascend DVPP (VPC) image operators.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_DVPP_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_DVPP_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>

namespace hrt::preprocess {

// Crop, resize and border fill in one VPC call (the fused step) for uint8
// RGB/BGR/NV12/NV21/gray images in DVPP memory. Models normalize via AIPP.
class DvppImagePreprocessor final : public ImagePreprocessor {
public:
  DvppImagePreprocessor();

  ~DvppImagePreprocessor() override;

  DvppImagePreprocessor(const DvppImagePreprocessor&) = delete;
  auto operator=(const DvppImagePreprocessor&) -> DvppImagePreprocessor& = delete;
  DvppImagePreprocessor(DvppImagePreprocessor&&) = delete;
  auto operator=(DvppImagePreprocessor&&) -> DvppImagePreprocessor& = delete;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Dvpp; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::Dvpp; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override { return type == core::MemoryType::Dvpp; }

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

  // VPC needs even region offsets and sizes.
  auto adjust_roi(core::Rect& dst_roi, core::Rect& src_roi) const -> void override;

  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

  auto fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
             const ImagePreprocessorParam& param, core::Stream* stream) -> bool override;

private:
  int channel_ = -1;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_DVPP_IMAGE_PREPROCESSOR_HPP
