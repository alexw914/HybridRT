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
    if (!is_float_outputs(engine, "general_detect"))
      return false;
    param_ = config.get<GeneralDetectParam>();
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
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
    const HeadGridView rows(*outputs[0], std::nullopt);
    decode_rows(result, rows);
    nms_to_source(result, param_.iou_threshold, transform);
    return true;
  }

private:
  // One class row at a time, in memory order; boxes only for scores above
  // the threshold. Anchors stay in order within a class.
  auto decode_rows(DetectResultMap& result, const HeadGridView& rows) -> void {
    const auto at = [&](std::size_t row, std::uint32_t i) -> float { return rows.at(0, i, row); };
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      for (const auto i : scan_.scan_channel(rows, 4 + cls, param_.confidence_threshold[cls])) {
        const auto rect = core::center_box(at(0, i), at(1, i), at(2, i), at(3, i));
        if (!core::valid_box(rect))
          continue;
        result[cls].push_back({rect, at(4 + cls, i), static_cast<std::int64_t>(cls)});
      }
    }
  }

  GeneralDetectParam param_;
  CellScanner scan_;
};

} // namespace

auto create_general_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<GeneralDetect>();
}

} // namespace hrt::postprocess
