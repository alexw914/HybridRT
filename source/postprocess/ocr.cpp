// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/ocr.cpp
// Description: Implements the OCR postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/ocr.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto create_ocr(std::string_view type) -> UniquePtr<OcrPostprocessor> {
  return type == "Ocr" ? create_ocr() : nullptr;
}

auto create_ocr_postprocessor(const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<OcrPostprocessor> {
  return check_init(create_ocr("Ocr"), "Ocr", config, engine);
}

} // namespace hrt::postprocess
