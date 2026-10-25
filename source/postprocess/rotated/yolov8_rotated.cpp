// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/yolov8_rotated.cpp
// Description: Implements YOLOv8 oriented-box decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>
#include <numbers>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

// Channels per cell: 4 * reg_max distance bins, theta, classes.
class YoloV8Rotated final : public RotatedPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolov8_rotated"))
      return false;
    param_ = config.get<YoloV8Param>();
    KZ_CHECK_RETURN(!config.contains("reg_max") || param_.reg_max > 0, false, "reg_max must be positive when supplied");
    if (!check_yolov8_heads(param_, engine, 1))
      return false;
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   core::logit);
    layout_ = engine.proto().output_layout;
    tables_.resize(engine.output_protos().size());
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<YoloV8Param>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGridView grid(*outputs[head], index, layout_);
      tables_[head].bind(grid, logit_threshold_);
      decode_head(result, grid, tables_[head], param_.strides[head]);
    }
    const auto mapped = nms_rotated_reverse(result, param_.iou_threshold, transform);
    sigmoid_scores(result);
    return mapped;
  }

private:
  auto decode_head(RotatedResultMap& result, const HeadGridView& grid, const QuantTables& tables, core::Point2f stride)
      -> void {
    const auto bins = param_.reg_max;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan_max_score(grid, bins * 4 + 1, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::RotatedRect rect;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, bins * 4 + 1 + cls) < tables.bound(cls))
          continue;
        // The logit ranks candidates in NMS; only kept ones take the sigmoid.
        const auto score = grid.at(y, x, bins * 4 + 1 + cls);
        if (!decoded) {
          const auto theta = (sigmoid_at(grid, tables, y, x, bins * 4) - 0.25F) * 180.0F;
          const auto rad = theta / 180.0F * std::numbers::pi_v<float>;
          const auto cos = std::cos(rad);
          const auto sin = std::sin(rad);
          const auto l = dfl(grid, tables, y, x, 0, bins);
          const auto t = dfl(grid, tables, y, x, bins, bins);
          const auto r = dfl(grid, tables, y, x, bins * 2, bins);
          const auto b = dfl(grid, tables, y, x, bins * 3, bins);
          const auto xf = (r - l) / 2.0F;
          const auto yf = (b - t) / 2.0F;
          rect.center = {(xf * cos - yf * sin + static_cast<float>(x) + 0.5F) * stride.x,
                         (xf * sin + yf * cos + static_cast<float>(y) + 0.5F) * stride.y};
          rect.size = {(r + l) * stride.x, (b + t) * stride.y};
          rect.angle = theta;
          decoded = true;
        }
        // The box is shared by every class of this cell; also skips NaN sizes.
        if (!(rect.size.width > 0.0F && rect.size.height > 0.0F))
          break;
        result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
      }
    }
  }

  YoloV8Param param_;
  CellScanner scan_;
  Vector<QuantTables> tables_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto create_yolov8_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<YoloV8Rotated>();
}

} // namespace hrt::postprocess
