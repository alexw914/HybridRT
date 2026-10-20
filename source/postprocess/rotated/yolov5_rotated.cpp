// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/yolov5_rotated.cpp
// Description: Implements YOLOv5 oriented-box decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// Channels per anchor: x, y, w, h, theta, objectness, classes.
class YoloV5Rotated final : public RotatedPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<YoloV5Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!float_outputs(engine, "yolov5_rotated"))
      return false;
    param_ = config.get<YoloV5Param>();
    if (!check_yolov5(param_, engine, 1))
      return false;
    const auto min_threshold =
        *std::min_element(param_.confidence_threshold.begin(), param_.confidence_threshold.end());
    min_objectness_logit_ = core::logit(min_threshold);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGrid grid(*outputs[head], layout_);
      decode_head(result, grid, head);
    }
    return finish_rotated(result, param_.iou_threshold, transform);
  }

private:
  auto decode_head(RotatedResultMap& result, const HeadGrid& grid, std::size_t head) -> void {
    const auto stride = param_.strides[head];
    const auto& anchors = param_.anchors[head];
    const auto min_objectness = grid.raw_bound(min_objectness_logit_);
    const auto channels = param_.class_num + 6;
    for (std::size_t anchor = 0; anchor < anchors.size() / 2; ++anchor) {
      const auto base = channels * anchor;
      for (const auto cell : scan_.scan(grid, base + 5, 1, min_objectness)) {
        const std::int64_t y = cell / grid.cols();
        const std::int64_t x = cell % grid.cols();
        const auto objectness = core::sigmoid(grid.at(y, x, base + 5));
        bool decoded = false;
        core::RotatedRect rect;
        for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
          const auto threshold = param_.confidence_threshold[cls];
          if (objectness < threshold)
            continue;
          const auto score = core::sigmoid(grid.at(y, x, base + 6 + cls)) * objectness;
          if (score <= threshold)
            continue;
          if (!decoded) {
            rect.center = {(core::sigmoid(grid.at(y, x, base)) * 2.0F - 0.5F + static_cast<float>(x)) * stride.x,
                           (core::sigmoid(grid.at(y, x, base + 1)) * 2.0F - 0.5F + static_cast<float>(y)) * stride.y};
            rect.size = {std::pow(core::sigmoid(grid.at(y, x, base + 2)) * 2.0F, 2.0F) * anchors[anchor * 2],
                         std::pow(core::sigmoid(grid.at(y, x, base + 3)) * 2.0F, 2.0F) * anchors[anchor * 2 + 1]};
            rect.angle = -(core::sigmoid(grid.at(y, x, base + 4)) - 0.5F) * 180.0F;
            decoded = true;
          }
          result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
        }
      }
    }
  }

  YoloV5Param param_;
  float min_objectness_logit_ = 0.0F;
  CellScanner scan_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolov5_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<YoloV5Rotated>();
}

} // namespace hrt::postprocess
