// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/nms.cpp
// Description: Implements the greedy non-maximum suppression shared by the
//              postprocessors and its mapping of kept boxes to the source.
// -----------------------------------------------------------------------------

#include "include/postprocess/nms.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

// OpenCV 5 moved rotatedRectangleIntersection and contourArea to geometry.
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#else
#include <opencv2/imgproc.hpp>
#endif

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

auto nms_indices(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t> {
  Vector<std::size_t> order(rects.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) -> bool { return scores[a] > scores[b]; });
  Vector<std::size_t> kept;
  for (auto index : order) {
    const auto& rect = rects[index];
    const auto overlaps = std::any_of(kept.begin(), kept.end(), [&](std::size_t other) -> bool {
      return core::iou(rect, rects[other]) >= iou_threshold;
    });
    if (!overlaps)
      kept.push_back(index);
  }
  return kept;
}

auto nms(DetectResult& boxes, float iou_threshold) -> void {
  if (boxes.size() <= 1)
    return;
  Vector<core::Rect2f> rects;
  Vector<float> scores;
  for (const auto& box : boxes) {
    rects.push_back(box.rect);
    scores.push_back(box.score);
  }
  DetectResult kept;
  for (auto index : nms_indices(rects, scores, iou_threshold))
    kept.push_back(boxes[index]);
  boxes = std::move(kept);
}

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

auto nms_reverse(DetectResultMap& result, const Vector<float>& iou_threshold,
                 const preprocess::ImageTransform& transform) -> void {
  for (auto& [cls, boxes] : result) {
    nms(boxes, iou_threshold[cls]);
    for (auto& box : boxes)
      box.rect = transform.clip_reverse(box.rect);
    std::erase_if(boxes, [](const ObjectBox& box) -> bool { return box.rect.empty(); });
  }
}

auto rotated_reverse(RotatedResultMap& result, const preprocess::ImageTransform& transform) -> bool {
  const auto& src = transform.src_roi;
  const auto& dst = transform.dst_roi;
  // Letterbox dimensions are rounded to integer pixels, and a device preprocessor may round
  // them again, as DVPP does to even values: each side can miss the exact ratio by two pixels.
  const auto difference =
      std::abs(static_cast<std::int64_t>(src.width) * dst.height - static_cast<std::int64_t>(src.height) * dst.width);
  const auto tolerance = 2 * (static_cast<std::int64_t>(src.width) + static_cast<std::int64_t>(src.height));
  const auto uniform = difference <= tolerance;
  if (!uniform)
    result.clear();
  KZ_CHECK_RETURN(uniform, false, "rotated: source mapping requires uniform scaling");
  for (auto& [cls, boxes] : result)
    for (auto& box : boxes)
      box.rect = transform.reverse(box.rect);
  return true;
}

auto nms_rotated_reverse(RotatedResultMap& result, const Vector<float>& iou_threshold,
                         const preprocess::ImageTransform& transform) -> bool {
  for (auto& [cls, boxes] : result)
    nms_rotated(boxes, iou_threshold[cls]);
  return rotated_reverse(result, transform);
}

} // namespace hrt::postprocess
