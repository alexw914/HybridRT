// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/rotated.hpp
// Description: Declares rotated-box detection results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_ROTATED_HPP
#define HRT_POSTPROCESS_ROTATED_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hrt::postprocess {

struct RotatedBox {
  core::RotatedRect rect;
  float score = 0.0F;
  std::int64_t type = 0;
  std::int64_t track_id = -1;
};

using RotatedResult = Vector<RotatedBox>;

using RotatedResultMap = Map<std::size_t, RotatedResult>;

// Source mapping requires uniform scaling, allowing integer letterbox rounding.
using RotatedPostprocessor = Postprocessor<RotatedResultMap>;

// Aborts on an unknown type, a config check_config rejects, or engine outputs
// that do not fit the config.
auto create_rotated_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<RotatedPostprocessor>;

// Greedy NMS within one class using polygon IoU.
auto nms_rotated(RotatedResult& boxes, float iou_threshold) -> void;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_ROTATED_HPP
