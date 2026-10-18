// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/ocr.hpp
// Description: Declares OCR regions, text results and stage postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_OCR_HPP
#define HRT_POSTPROCESS_OCR_HPP

#include <hrt/postprocess/postprocessor.hpp>

namespace hrt::postprocess {

struct OcrText {
  core::Rect2f rect; // Axis-aligned text box in source pixels.
  String text;       // UTF-8; empty for detection-only results.
  float detection_score = 0.0F;
  float recognition_score = 0.0F;
};

using OcrResult = Vector<OcrText>;
using OcrPostprocessor = Postprocessor<OcrResult>;

auto create_ocr_postprocessor(const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<OcrPostprocessor>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_OCR_HPP
