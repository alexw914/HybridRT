// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/detect.hpp
// Description: Declares detection results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_DETECT_HPP
#define HRT_POSTPROCESS_DETECT_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hrt::postprocess {

struct ObjectBox {
  core::Rect2f rect;
  float score = 0.0F;
  std::int64_t type = 0;
  std::int64_t track_id = -1;
};

using DetectResult = Vector<ObjectBox>;

using DetectResultMap = Map<std::size_t, DetectResult>;

using DetectPostprocessor = Postprocessor<DetectResultMap>;

// Aborts on an unknown type, a config check_config rejects, or engine outputs
// that do not fit the config.
auto create_detect_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<DetectPostprocessor>;

auto nms(DetectResult& boxes, float iou_threshold) -> void;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_DETECT_HPP
