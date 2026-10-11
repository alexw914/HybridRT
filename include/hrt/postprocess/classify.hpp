// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/classify.hpp
// Description: Declares classification results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_CLASSIFY_HPP
#define HRT_POSTPROCESS_CLASSIFY_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <string_view>

namespace hrt::postprocess {

// One score vector per engine output, indexed by class_id.
using ClassifyResult = Vector<Vector<float>>;

using ClassifyPostprocessor = Postprocessor<ClassifyResult>;

// Aborts on an unknown type, a config check_config rejects, or engine outputs
// that do not fit the config.
auto create_classify_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<ClassifyPostprocessor>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_CLASSIFY_HPP
