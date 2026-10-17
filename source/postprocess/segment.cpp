// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment.cpp
// Description: Implements the segmentation postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/segment.hpp>

#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto make_segment(std::string_view type) -> UniquePtr<SegmentPostprocessor> {
  if (type == "InstanceSegment")
    return make_instance_segment();
  if (type == "E2ESegment")
    return make_e2e_segment();
  if (type == "SemanticSegment")
    return make_semantic_segment();
  return nullptr;
}

auto create_segment_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<SegmentPostprocessor> {
  return checked(make_segment(type), type, config, engine);
}

} // namespace hrt::postprocess
