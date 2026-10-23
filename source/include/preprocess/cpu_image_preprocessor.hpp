// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/cpu_image_preprocessor.hpp
// Description: Declares the OpenCV-based host image operators.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_CPU_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_CPU_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>

#include <opencv2/core.hpp>

namespace hrt::preprocess {

// Operates on uint8 and float32 images in host-visible memory, including
// RK memory that maps to the host.
class CpuImagePreprocessor final : public ImagePreprocessor {
public:
  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Cpu; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::Cpu; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override;

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

private:
  auto convert_normalize(core::Image& dst, const core::Image& src, const core::Rect& roi, const Array<float, 3>& mean,
                         const Array<float, 3>& stddev, core::Stream* stream) -> bool override;

  // Reused intermediates for multi-step conversions.
  cv::Mat bgr_;
  Array<cv::Mat, 3> planes_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_CPU_IMAGE_PREPROCESSOR_HPP
