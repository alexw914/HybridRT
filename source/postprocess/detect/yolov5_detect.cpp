// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/yolov5_detect.cpp
// Description: Implements YOLOv5 anchor-based detection decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"
#include "include/postprocess/yolo_heads.hpp"

namespace hrt::postprocess {

namespace {

class YoloV5Detect final : public DetectPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolov5_detect"))
      return false;
    param_ = config.get<YoloV5Param>();
    if (!resolve_yolov5_heads(param_, engine, 0))
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
  auto decode(DetectResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], index, layout_);
      tables_[head].bind(grid);
      decode_head(result, grid, head);
    }
    nms_reverse(result, param_.iou_threshold, transform);
    return true;
  }

private:
  auto decode_head(DetectResultMap& result, const HeadGridView& grid, std::size_t head) -> void {
    const auto& tables = tables_[head];
    const auto stride = param_.strides[head];
    const auto& anchors = param_.anchors[head];
    const auto min_objectness = grid.raw_bound(min_objectness_logit_);
    const auto channels = param_.class_num + 5;
    for (std::size_t anchor = 0; anchor < anchors.size() / 2; ++anchor) {
      const auto base = channels * anchor;
      for (const auto cell : scan_.scan_max_score(grid, base + 4, 1, min_objectness)) {
        const std::int64_t y = cell / grid.cols();
        const std::int64_t x = cell % grid.cols();
        const auto objectness = sigmoid_at(grid, tables, y, x, base + 4);
        bool decoded = false;
        core::Rect2f box;
        for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
          const auto threshold = param_.confidence_threshold[cls];
          if (objectness < threshold)
            continue;
          const auto score = sigmoid_at(grid, tables, y, x, base + 5 + cls) * objectness;
          if (score <= threshold)
            continue;
          if (!decoded) {
            const auto cx = (sigmoid_at(grid, tables, y, x, base) * 2.0F - 0.5F + static_cast<float>(x)) * stride.x;
            const auto cy = (sigmoid_at(grid, tables, y, x, base + 1) * 2.0F - 0.5F + static_cast<float>(y)) * stride.y;
            const auto w = std::pow(sigmoid_at(grid, tables, y, x, base + 2) * 2.0F, 2.0F) * anchors[anchor * 2];
            const auto h = std::pow(sigmoid_at(grid, tables, y, x, base + 3) * 2.0F, 2.0F) * anchors[anchor * 2 + 1];
            box = {cx - w / 2.0F, cy - h / 2.0F, w, h};
            decoded = true;
          }
          result[cls].push_back({box, score, static_cast<std::int64_t>(cls)});
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

auto create_yolov5_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<YoloV5Detect>();
}

} // namespace hrt::postprocess
