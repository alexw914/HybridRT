// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/yolov5_detect.cpp
// Description: Implements YOLOv5 anchor-based detection decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

class YoloV5Detect final : public DetectPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<YoloV5Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<YoloV5Param>();
    if (!check_yolov5(param_, engine, 0))
      return false;
    const auto min_threshold =
        *std::min_element(param_.confidence_threshold.begin(), param_.confidence_threshold.end());
    min_objectness_logit_ = logit(min_threshold);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const auto ok =
          visit_head(*outputs[head], layout_, [&](const auto& grid) -> void { decode_head(result, grid, head); });
      if (!ok)
        return false;
    }
    finish_detect(result, param_.iou_threshold, transform);
    return true;
  }

private:
  template <typename Grid>
  auto decode_head(DetectResultMap& result, const Grid& grid, std::size_t head) const -> void {
    const auto stride = param_.strides[head];
    const auto& anchors = param_.anchors[head];
    const auto min_objectness = grid.raw_bound(min_objectness_logit_);
    const auto channels = param_.class_num + 5;
    for (std::int64_t y = 0; y < grid.rows(); ++y) {
      for (std::int64_t x = 0; x < grid.cols(); ++x) {
        for (std::size_t anchor = 0; anchor < anchors.size() / 2; ++anchor) {
          const auto base = channels * anchor;
          if (grid.raw(y, x, base + 4) < min_objectness)
            continue;
          const auto objectness = sigmoid(grid.at(y, x, base + 4));
          bool decoded = false;
          core::Rect2f box;
          for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
            const auto threshold = param_.confidence_threshold[cls];
            if (objectness < threshold)
              continue;
            const auto score = sigmoid(grid.at(y, x, base + 5 + cls)) * objectness;
            if (score <= threshold)
              continue;
            if (!decoded) {
              const auto cx = (sigmoid(grid.at(y, x, base)) * 2.0F - 0.5F + static_cast<float>(x)) * stride;
              const auto cy = (sigmoid(grid.at(y, x, base + 1)) * 2.0F - 0.5F + static_cast<float>(y)) * stride;
              const auto w = std::pow(sigmoid(grid.at(y, x, base + 2)) * 2.0F, 2.0F) * anchors[anchor * 2];
              const auto h = std::pow(sigmoid(grid.at(y, x, base + 3)) * 2.0F, 2.0F) * anchors[anchor * 2 + 1];
              box = {cx - w / 2.0F, cy - h / 2.0F, w, h};
              decoded = true;
            }
            result[cls].push_back({box, score, static_cast<std::int64_t>(cls)});
          }
        }
      }
    }
  }

  YoloV5Param param_;
  float min_objectness_logit_ = 0.0F;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolov5_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<YoloV5Detect>();
}

} // namespace hrt::postprocess
