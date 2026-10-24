// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/classify.cpp
// Description: Implements the classification postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/classify.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto create_classify(std::string_view type) -> UniquePtr<ClassifyPostprocessor> {
  return type == "GeneralClassify" ? create_general_classify() : nullptr;
}

auto create_classify_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<ClassifyPostprocessor> {
  return check_init(create_classify(type), type, config, engine);
}

} // namespace hrt::postprocess
