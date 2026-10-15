// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/view.cpp
// Description: Draws task results with shared class colors, labels and masks.
// -----------------------------------------------------------------------------

#include <hrt/core/view.hpp>

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace hrt::core {

namespace {

auto color(std::size_t cls) -> cv::Scalar {
  return {static_cast<double>(64 + (cls * 67) % 192), static_cast<double>(64 + (cls * 131 + 64) % 192),
          static_cast<double>(64 + (cls * 173 + 128) % 192)};
}

auto draw_label(cv::Mat& canvas, std::size_t cls, float score, cv::Point at) -> void {
  cv::putText(canvas, cv::format("%zu: %.2f", cls, static_cast<double>(score)), at, cv::FONT_HERSHEY_SIMPLEX, 0.8,
              color(cls), 2);
}

auto draw_box(cv::Mat& canvas, const postprocess::ObjectBox& box, std::size_t cls, int thickness) -> void {
  const auto& r = box.rect;
  const cv::Rect rect(static_cast<int>(r.x), static_cast<int>(r.y), static_cast<int>(r.width),
                      static_cast<int>(r.height));
  cv::rectangle(canvas, rect, color(cls), thickness);
  draw_label(canvas, cls, box.score, rect.tl() - cv::Point(0, 6));
}

auto blend_mask(cv::Mat& roi, const cv::Mat& bits, std::size_t cls) -> void {
  cv::Mat tinted;
  cv::addWeighted(roi, 0.5, cv::Mat(roi.size(), roi.type(), color(cls)), 0.5, 0.0, tinted);
  tinted.copyTo(roi, bits);
}

} // namespace

auto draw(cv::Mat& canvas, const postprocess::DetectResultMap& result) -> void {
  for (const auto& [cls, boxes] : result) {
    for (const auto& box : boxes)
      draw_box(canvas, box, cls, 3);
  }
}

auto draw(cv::Mat& canvas, const postprocess::RotatedResultMap& result) -> void {
  for (const auto& [cls, boxes] : result) {
    for (const auto& box : boxes) {
      const auto& r = box.rect;
      cv::Point2f corners[4];
      cv::RotatedRect({r.center.x, r.center.y}, {r.size.width, r.size.height}, r.angle).points(corners);
      for (int i = 0; i < 4; ++i)
        cv::line(canvas, corners[i], corners[(i + 1) % 4], color(cls), 3);
      draw_label(canvas, cls, box.score, corners[1]);
    }
  }
}

// Top class of each output, one label per line from the top-left corner.
auto draw(cv::Mat& canvas, const postprocess::ClassifyResult& result) -> void {
  int line = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto& scores = result[i];
    if (scores.empty())
      continue;
    const auto best = static_cast<std::size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    draw_label(canvas, best, scores[best], {8, 30 * ++line});
  }
}

// Box (when the model gives one), visible keypoints and, for 17-keypoint human
// poses, the skeleton.
auto draw(cv::Mat& canvas, const postprocess::KeypointResultMap& result) -> void {
  static constexpr Array<Array<int, 2>, 19> kSkeleton{{{15, 13},
                                                       {13, 11},
                                                       {16, 14},
                                                       {14, 12},
                                                       {11, 12},
                                                       {5, 11},
                                                       {6, 12},
                                                       {5, 6},
                                                       {5, 7},
                                                       {6, 8},
                                                       {7, 9},
                                                       {8, 10},
                                                       {1, 2},
                                                       {0, 1},
                                                       {0, 2},
                                                       {1, 3},
                                                       {2, 4},
                                                       {3, 5},
                                                       {4, 6}}};
  const auto visible = [](const postprocess::Keypoint& keypoint) -> bool {
    return keypoint.visible == postprocess::KeypointVisible::Visible;
  };
  const auto pixel = [](const postprocess::Keypoint& keypoint) -> cv::Point {
    return {static_cast<int>(keypoint.point.x), static_cast<int>(keypoint.point.y)};
  };
  for (const auto& [cls, objects] : result) {
    for (const auto& object : objects) {
      const auto& points = object.keypoints;
      if (cls >= 0 && !object.box.rect.empty())
        draw_box(canvas, object.box, static_cast<std::size_t>(cls), 2);
      if (points.size() == 17) {
        for (const auto& [a, b] : kSkeleton) {
          const auto& from = points[static_cast<std::size_t>(a)];
          const auto& to = points[static_cast<std::size_t>(b)];
          if (visible(from) && visible(to))
            cv::line(canvas, pixel(from), pixel(to), {255, 200, 0}, 2);
        }
      }
      for (std::size_t i = 0; i < points.size(); ++i) {
        if (visible(points[i]))
          cv::circle(canvas, pixel(points[i]), 4, color(i), cv::FILLED);
      }
    }
  }
}

auto draw(cv::Mat& canvas, const postprocess::SegmentResultMap& result) -> void {
  for (const auto& [cls, objects] : result) {
    for (const auto& object : objects) {
      if (cls < 0)
        continue;
      const auto& m = object.mask;
      const cv::Rect region(m.rect.x, m.rect.y, m.rect.width, m.rect.height);
      const cv::Mat bits(m.rect.height, m.rect.width, CV_8U, const_cast<std::uint8_t*>(m.data.data()));
      cv::Mat roi = canvas(region);
      blend_mask(roi, bits, static_cast<std::size_t>(cls));
    }
  }
  for (const auto& [cls, objects] : result) {
    if (cls < 0)
      continue;
    for (const auto& object : objects)
      draw_box(canvas, object.box, static_cast<std::size_t>(cls), 2);
  }
}

} // namespace hrt::core
