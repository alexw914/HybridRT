// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/general_detect.cpp
// Description: Implements decoding of single-output detection heads,
//              [1, 4 + classes, N].
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// One output [1, 4 + classes, N]: rows center x, center y, width, height, then
// one score per class.
class GeneralDetect final : public DetectPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<GeneralDetectParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<GeneralDetectParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || engine.input_protos().size() != 1 ||
        outputs.size() != 1 || outputs[0].shape.size() != 3 || outputs[0].shape[0] != 1 ||
        outputs[0].shape[1] != static_cast<std::int64_t>(param_.class_num) + 4) {
      KZ_LOG_ERROR("general_detect: needs one input and one [1, 4 + class_num, N] output");
      return false;
    }
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const RowView rows(*outputs[0], scratch_);
    if (!rows.valid())
      return false;
    // One class row at a time, in memory order; boxes only for scores above
    // the threshold. Anchors stay in order within a class.
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      candidates_.clear();
      rows.above(4 + cls, param_.confidence_threshold[cls], candidates_);
      for (const auto i : candidates_) {
        const auto [l, t, r, b] = center_box(rows.at(0, i), rows.at(1, i), rows.at(2, i), rows.at(3, i));
        // Also skips NaN boxes.
        if (!(r - l > 0.0F && b - t > 0.0F))
          continue;
        result[cls].push_back(
            {core::Rect2f::from_points({l, t}, {r, b}), rows.at(4 + cls, i), static_cast<std::int64_t>(cls)});
      }
    }
    finish_detect(result, param_.iou_threshold, transform);
    return true;
  }

private:
  GeneralDetectParam param_;
  Vector<std::uint32_t> candidates_;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_general_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<GeneralDetect>();
}

} // namespace hrt::postprocess
