// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint.cpp
// Description: Implements the keypoint postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/keypoint.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto create_keypoint(std::string_view type) -> UniquePtr<KeypointPostprocessor> {
  if (type == "GeneralKeypoint")
    return create_general_keypoint();
  if (type == "GeneralYoloPose")
    return create_general_yolo_pose();
  if (type == "YoloV8Keypoint")
    return create_yolov8_keypoint();
  if (type == "Yolo26Keypoint")
    return create_yolo26_keypoint();
  if (type == "RtmoKeypoint")
    return create_rtmo_keypoint();
  if (type == "E2EKeypoint")
    return create_e2e_keypoint();
  return nullptr;
}

auto create_keypoint_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<KeypointPostprocessor> {
  return check_init(create_keypoint(type), type, config, engine);
}

} // namespace hrt::postprocess
