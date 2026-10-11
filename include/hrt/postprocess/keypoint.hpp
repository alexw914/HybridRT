// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/keypoint.hpp
// Description: Declares keypoint results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_KEYPOINT_HPP
#define HRT_POSTPROCESS_KEYPOINT_HPP

#include <hrt/postprocess/detect.hpp>
#include <hrt/postprocess/postprocessor.hpp>

#include <cstdint>
#include <string_view>

namespace hrt::postprocess {

// Visibility values as the model outputs them.
enum class KeypointVisible {
  Unlabeled = 0,
  Invisible = 1,
  Visible = 2,
};

struct Keypoint {
  core::Point2f point;
  KeypointVisible visible = KeypointVisible::Visible;
  float score = 1.0F;
};

// General keypoints use class 0 and a box covering the full input region.
struct KeypointObject {
  ObjectBox box;
  Vector<Keypoint> keypoints;
};

using KeypointResult = Vector<KeypointObject>;

using KeypointResultMap = Map<std::int64_t, KeypointResult>;

using KeypointPostprocessor = Postprocessor<KeypointResultMap>;

// Aborts on an unknown type, a config check_config rejects, or engine outputs
// that do not fit the config.
auto create_keypoint_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<KeypointPostprocessor>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_KEYPOINT_HPP
