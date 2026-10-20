// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated.cpp
// Description: Implements rotated-box NMS and the rotated postprocessor
//              factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/rotated.hpp>

#include <kitzoo/log/logger.hpp>

#include <algorithm>

// OpenCV 5 moved rotatedRectangleIntersection and contourArea to geometry.
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#else
#include <opencv2/imgproc.hpp>
#endif

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

auto to_cv(const core::RotatedRect& rect) -> cv::RotatedRect {
  return {{rect.center.x, rect.center.y}, {rect.size.width, rect.size.height}, rect.angle};
}

auto rotated_iou(const core::RotatedRect& a, const core::RotatedRect& b) -> float {
  Vector<cv::Point2f> polygon;
  const auto kind = cv::rotatedRectangleIntersection(to_cv(a), to_cv(b), polygon);
  if (kind == cv::INTERSECT_NONE || polygon.empty())
    return 0.0F;
  const auto inter = kind == cv::INTERSECT_FULL ? std::min(a.size.area(), b.size.area())
                                                : static_cast<float>(cv::contourArea(polygon));
  return inter / (a.size.area() + b.size.area() - inter);
}

} // namespace

auto nms_rotated(RotatedResult& boxes, float iou_threshold) -> void {
  if (boxes.size() <= 1)
    return;
  std::sort(boxes.begin(), boxes.end(),
            [](const RotatedBox& a, const RotatedBox& b) -> bool { return a.score > b.score; });
  RotatedResult kept;
  for (const auto& box : boxes) {
    const auto overlaps = std::any_of(kept.begin(), kept.end(), [&](const RotatedBox& other) -> bool {
      return rotated_iou(box.rect, other.rect) >= iou_threshold;
    });
    if (!overlaps)
      kept.push_back(box);
  }
  boxes = std::move(kept);
}

auto make_rotated(std::string_view type) -> UniquePtr<RotatedPostprocessor> {
  if (type == "YoloV8Rotated")
    return make_yolov8_rotated();
  if (type == "Yolo26Rotated")
    return make_yolo26_rotated();
  if (type == "YoloV5Rotated")
    return make_yolov5_rotated();
  if (type == "GeneralRotated")
    return make_general_rotated();
  if (type == "E2ERotated")
    return make_e2e_rotated();
  return nullptr;
}

auto create_rotated_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<RotatedPostprocessor> {
  return checked(make_rotated(type), type, config, engine);
}

auto map_rotated(RotatedResultMap& result, const preprocess::ImageTransform& transform) -> bool {
  const auto& src = transform.src_roi;
  const auto& dst = transform.dst_roi;
  // Letterbox dimensions are rounded to integer pixels, and a device preprocessor may round
  // them again, as DVPP does to even values: each side can miss the exact ratio by two pixels.
  const auto difference =
      std::abs(static_cast<std::int64_t>(src.width) * dst.height - static_cast<std::int64_t>(src.height) * dst.width);
  const auto tolerance = 2 * (static_cast<std::int64_t>(src.width) + static_cast<std::int64_t>(src.height));
  const auto uniform = !src.empty() && !dst.empty() && difference <= tolerance;
  if (!uniform)
    result.clear();
  KZ_CHECK_RETURN(uniform, false, "rotated: source mapping requires uniform scaling and nonempty regions");
  for (auto& [cls, boxes] : result) {
    for (auto& box : boxes) {
      const auto mapped = transform.to_source(
          core::Rect2f{box.rect.center.x, box.rect.center.y, box.rect.size.width, box.rect.size.height});
      box.rect.center = {mapped.x, mapped.y};
      box.rect.size = {mapped.width, mapped.height};
    }
    std::erase_if(boxes, [](const RotatedBox& box) -> bool { return box.rect.size.empty(); });
  }
  return true;
}

auto finish_rotated(RotatedResultMap& result, const Vector<float>& iou_threshold,
                    const preprocess::ImageTransform& transform) -> bool {
  for (auto& [cls, boxes] : result)
    nms_rotated(boxes, iou_threshold[cls]);
  return map_rotated(result, transform);
}

} // namespace hrt::postprocess
