// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/preprocess/text_cropper.hpp
// Description: Declares host text box crops for recognition.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_TEXT_CROPPER_HPP
#define HRT_PREPROCESS_TEXT_CROPPER_HPP

#include <hrt/core/image.hpp>

namespace hrt::preprocess {

// Not thread-safe. Source must be host uint8 interleaved BGR or RGB.
class TextCropper {
public:
  TextCropper();

  ~TextCropper();

  TextCropper(const TextCropper&) = delete;
  auto operator=(const TextCropper&) -> TextCropper& = delete;

  // Truncates the box to whole pixels like PP-OCR. An unrotated crop views src; a rotated
  // one is valid until the next crop.
  auto process(const core::Image& src, const core::Rect2f& rect, bool rotate_tall = true) -> std::optional<core::Image>;

private:
  struct Impl;
  UniquePtr<Impl> impl_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_TEXT_CROPPER_HPP
