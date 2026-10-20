// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/yolo26_detect.cpp
// Description: Implements YOLO26 raw-head detection decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// One output per stride with channels left, top, right and bottom distances
// in cells (no distance bins), then class logits, as the one-to-many head of a
// YOLO26 model emits them before decoding. NMS still runs.
class Yolo26Detect final : public DetectPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<Yolo26Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<Yolo26Param>();
    if (!check_yolo26(param_, engine, 0))
      return false;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const auto ok = HeadGrid<>::visit(
          *outputs[head], layout_, [&](const auto& grid) -> void { decode_head(result, grid, param_.strides[head]); });
      if (!ok)
        return false;
    }
    finish_detect(result, param_.iou_threshold, transform);
    return true;
  }

private:
  template <typename Grid>
  auto decode_head(DetectResultMap& result, const Grid& grid, float stride) -> void {
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan(grid, 4, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::Rect2f box;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, 4 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, 4 + cls);
        if (score < logit_threshold_[cls])
          continue;
        if (!decoded) {
          const auto cx = static_cast<float>(x) + 0.5F;
          const auto cy = static_cast<float>(y) + 0.5F;
          box = core::Rect2f::from_points({(cx - grid.at(y, x, 0)) * stride, (cy - grid.at(y, x, 1)) * stride},
                                          {(cx + grid.at(y, x, 2)) * stride, (cy + grid.at(y, x, 3)) * stride});
          decoded = true;
        }
        result[cls].push_back({box, sigmoid(score), static_cast<std::int64_t>(cls)});
      }
    }
  }

  Yolo26Param param_;
  CellScanner scan_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolo26_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<Yolo26Detect>();
}

} // namespace hrt::postprocess
