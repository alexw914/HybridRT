// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/yolo26_detect.cpp
// Description: Implements YOLO26 raw-head detection decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"
#include "include/postprocess/yolo_heads.hpp"

namespace hrt::postprocess {

namespace {

// One output per stride with channels left, top, right and bottom distances
// in cells (no distance bins), then class logits, as the one-to-many head of a
// YOLO26 model emits them before decoding. NMS still runs.
class Yolo26Detect final : public DetectPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolo26_detect"))
      return false;
    param_ = config.get<Yolo26Param>();
    if (!resolve_yolo26_heads(param_, engine, 0))
      return false;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   core::logit);
    layout_ = engine.proto().output_layout;
    tables_.resize(engine.output_protos().size());
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<Yolo26Param>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], index, layout_);
      tables_[head].bind(grid, logit_threshold_);
      decode_head(result, grid, tables_[head], param_.strides[head]);
    }
    nms_reverse(result, param_.iou_threshold, transform);
    sigmoid_scores(result);
    return true;
  }

private:
  auto decode_head(DetectResultMap& result, const HeadGridView& grid, const QuantTables& tables, core::Point2f stride)
      -> void {
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan_max_score(grid, 4, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::Rect2f box;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, 4 + cls) < tables.bound(cls))
          continue;
        // The logit ranks candidates in NMS; only kept ones take the sigmoid.
        const auto score = grid.at(y, x, 4 + cls);
        if (!decoded) {
          const auto cx = static_cast<float>(x) + 0.5F;
          const auto cy = static_cast<float>(y) + 0.5F;
          box = core::Rect2f({(cx - grid.at(y, x, 0)) * stride.x, (cy - grid.at(y, x, 1)) * stride.y},
                             {(cx + grid.at(y, x, 2)) * stride.x, (cy + grid.at(y, x, 3)) * stride.y});
          decoded = true;
        }
        result[cls].push_back({box, score, static_cast<std::int64_t>(cls)});
      }
    }
  }

  Yolo26Param param_;
  CellScanner scan_;
  Vector<QuantTables> tables_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto create_yolo26_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<Yolo26Detect>();
}

} // namespace hrt::postprocess
