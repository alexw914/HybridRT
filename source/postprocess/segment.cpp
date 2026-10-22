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

auto create_segment(std::string_view type) -> UniquePtr<SegmentPostprocessor> {
  if (type == "GeneralYoloSegment")
    return create_general_yolo_segment();
  if (type == "YoloV8Segment")
    return create_yolov8_segment();
  if (type == "Yolo26Segment")
    return create_yolo26_segment();
  if (type == "E2ESegment")
    return create_e2e_segment();
  if (type == "SemanticSegment")
    return create_semantic_segment();
  return nullptr;
}

auto create_segment_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<SegmentPostprocessor> {
  return check_init_postprocess(create_segment(type), type, config, engine);
}

} // namespace hrt::postprocess
