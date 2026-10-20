// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/yolo26_rotated.cpp
// Description: Implements YOLO26 raw-head oriented box decoding.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// Channels per cell: left, top, right and bottom distances in cells (no
// distance bins), theta in radians, then class logits, as the one-to-many
// head of a YOLO26 OBB model emits them. NMS still runs.
class Yolo26Rotated final : public RotatedPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<Yolo26Param>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<Yolo26Param>();
    if (!check_yolo26(param_, engine, 1))
      return false;
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    if (outputs.size() != param_.strides.size())
      return false;
    for (std::size_t head = 0; head < outputs.size(); ++head) {
      const HeadGrid grid(*outputs[head], layout_);
      if (!grid.valid())
        return false;
      decode_head(result, grid, param_.strides[head]);
    }
    return finish_rotated(result, param_.iou_threshold, transform);
  }

private:
  auto decode_head(RotatedResultMap& result, const HeadGrid& grid, float stride) -> void {
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan(grid, 5, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      bool decoded = false;
      core::RotatedRect rect;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, 5 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, 5 + cls);
        if (score < logit_threshold_[cls])
          continue;
        if (!decoded) {
          const auto rad = grid.at(y, x, 4);
          const auto cos = std::cos(rad);
          const auto sin = std::sin(rad);
          const auto l = grid.at(y, x, 0);
          const auto t = grid.at(y, x, 1);
          const auto r = grid.at(y, x, 2);
          const auto b = grid.at(y, x, 3);
          const auto xf = (r - l) / 2.0F;
          const auto yf = (b - t) / 2.0F;
          rect.center = {(xf * cos - yf * sin + static_cast<float>(x) + 0.5F) * stride,
                         (xf * sin + yf * cos + static_cast<float>(y) + 0.5F) * stride};
          rect.size = {(r + l) * stride, (b + t) * stride};
          rect.angle = rad * 180.0F / std::numbers::pi_v<float>;
          decoded = true;
        }
        result[cls].push_back({rect, sigmoid(score), static_cast<std::int64_t>(cls)});
      }
    }
  }

  Yolo26Param param_;
  CellScanner scan_;
  Vector<float> logit_threshold_;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
};

} // namespace

auto make_yolo26_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<Yolo26Rotated>();
}

} // namespace hrt::postprocess
