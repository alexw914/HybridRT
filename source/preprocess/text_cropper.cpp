// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/text_cropper.cpp
// Description: Implements host text box crops and tall text rotation.
// -----------------------------------------------------------------------------

#include <hrt/preprocess/text_cropper.hpp>

#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <opencv2/core.hpp>

namespace hrt::preprocess {

struct TextCropper::Impl {
  cv::Mat rotated;
};

TextCropper::TextCropper() : impl_(std::make_unique<Impl>()) {}

TextCropper::~TextCropper() = default;

auto TextCropper::process(const core::Image& src, const core::Rect2f& rect, bool rotate_tall)
    -> std::optional<core::Image> {
  KZ_CHECK_RETURN(src.valid() && src.memory()->have_cpu() && src.data_type() == core::DataType::Uint8 &&
                      (src.format() == core::ImageFormat::InterBgr || src.format() == core::ImageFormat::InterRgb),
                  std::nullopt, "TextCropper: expected a host uint8 BGR/RGB image");
  KZ_CHECK_RETURN(std::isfinite(rect.x) && std::isfinite(rect.y) && std::isfinite(rect.width) &&
                      std::isfinite(rect.height) && rect.x >= 0 && rect.y >= 0 &&
                      rect.x + rect.width <= static_cast<float>(src.size().width) &&
                      rect.y + rect.height <= static_cast<float>(src.size().height),
                  std::nullopt, "TextCropper: box leaves the source image");
  const cv::Rect roi(static_cast<int>(rect.x), static_cast<int>(rect.y), static_cast<int>(rect.width),
                     static_cast<int>(rect.height));
  KZ_CHECK_RETURN(roi.width >= 2 && roi.height >= 2, std::nullopt, "TextCropper: degenerate text box");
  auto* data = static_cast<std::uint8_t*>(const_cast<void*>(src.memory()->cpu()));
  KZ_CHECK_RETURN(data != nullptr, std::nullopt, "TextCropper: cannot map source image");
  const cv::Mat crop = cv::Mat(src.size().height, src.size().width, CV_8UC3, data, src.stride()[1])(roi);
  const cv::Mat* output = &crop;
  if (rotate_tall && roi.height >= roi.width * 1.5) {
    cv::rotate(crop, impl_->rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    output = &impl_->rotated;
  }
  return core::Image::wrap(core::MemoryType::Cpu, output->data, src.format(), {output->cols, output->rows},
                           core::DataType::Uint8, output->step);
}

} // namespace hrt::preprocess
