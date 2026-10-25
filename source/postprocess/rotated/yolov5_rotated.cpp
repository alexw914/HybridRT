// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/yolov5_rotated.cpp
// Description: Implements YOLOv5 oriented-box decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"
#include "include/postprocess/yolo_heads.hpp"

namespace hrt::postprocess {

namespace {

// Channels per anchor: x, y, w, h, theta, objectness, classes.
class YoloV5Rotated final : public RotatedPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolov5_rotated"))
      return false;
    param_ = config.get<YoloV5Param>();
    if (!resolve_yolov5_heads(param_, engine, 1))
      return false;
    const auto min_threshold =
        *std::min_element(param_.confidence_threshold.begin(), param_.confidence_threshold.end());
    min_objectness_logit_ = core::logit(min_threshold);
    layout_ = engine.proto().output_layout;
    tables_.resize(engine.output_protos().size());
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<YoloV5Param>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], index, layout_);
      tables_[head].bind(grid);
      decode_head(result, grid, head);
    }
    return nms_rotated_reverse(result, param_.iou_threshold, transform);
  }

private:
  auto decode_head(RotatedResultMap& result, const HeadGridView& grid, std::size_t head) -> void {
    const auto& tables = tables_[head];
    const auto stride = param_.strides[head];
    const auto& anchors = param_.anchors[head];
    const auto min_objectness = grid.raw_bound(min_objectness_logit_);
    const auto channels = param_.class_num + 6;
    for (std::size_t anchor = 0; anchor < anchors.size() / 2; ++anchor) {
      const auto base = channels * anchor;
      for (const auto cell : scan_.scan_max_score(grid, base + 5, 1, min_objectness)) {
        const std::int64_t y = cell / grid.cols();
        const std::int64_t x = cell % grid.cols();
        const auto objectness = sigmoid_at(grid, tables, y, x, base + 5);
        bool decoded = false;
        core::RotatedRect rect;
        for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
          const auto threshold = param_.confidence_threshold[cls];
          if (objectness < threshold)
            continue;
          const auto score = sigmoid_at(grid, tables, y, x, base + 6 + cls) * objectness;
          if (score <= threshold)
            continue;
          if (!decoded) {
            rect.center = {(sigmoid_at(grid, tables, y, x, base) * 2.0F - 0.5F + static_cast<float>(x)) * stride.x,
                           (sigmoid_at(grid, tables, y, x, base + 1) * 2.0F - 0.5F + static_cast<float>(y)) * stride.y};
            rect.size = {std::pow(sigmoid_at(grid, tables, y, x, base + 2) * 2.0F, 2.0F) * anchors[anchor * 2],
                         std::pow(sigmoid_at(grid, tables, y, x, base + 3) * 2.0F, 2.0F) * anchors[anchor * 2 + 1]};
            rect.angle = -(sigmoid_at(grid, tables, y, x, base + 4) - 0.5F) * 180.0F;
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
  Vector<QuantTables> tables_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto create_yolov5_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<YoloV5Rotated>();
}

} // namespace hrt::postprocess
