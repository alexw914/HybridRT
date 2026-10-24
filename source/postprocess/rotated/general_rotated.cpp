// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/general_rotated.cpp
// Description: Implements decoding of single-output oriented-box heads,
//              [1, 4 + classes + 1, N].
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <numbers>

#include "include/postprocess/common.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/params.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

// One output [1, 4 + classes + 1, N]: rows center x, center y, width, height,
// one score per class, then the angle in radians.
class GeneralRotated final : public RotatedPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "general_rotated"))
      return false;
    param_ = config.get<GeneralDetectParam>();
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || engine.input_protos().size() != 1 ||
        outputs.size() != 1 || outputs[0].shape.size() != 3 || outputs[0].shape[0] < 1 ||
        outputs[0].shape[1] != static_cast<std::int64_t>(param_.class_num) + 5) {
      KZ_LOG_ERROR("general_rotated: needs one input and one [batch, 4 + class_num + 1, N] output");
      return false;
    }
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<GeneralDetectParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView rows(*outputs[0], std::nullopt);
    decode_rows(result, rows);
    return nms_rotated_reverse(result, param_.iou_threshold, transform);
  }

private:
  auto decode_rows(RotatedResultMap& result, const HeadGridView& rows) -> void {
    const auto at = [&](std::size_t row, std::uint32_t i) -> float { return rows.at(0, i, row); };
    const auto angle_row = param_.class_num + 4;
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      for (const auto i : scan_.scan_channel(rows, 4 + cls, param_.confidence_threshold[cls])) {
        const core::RotatedRect rect{
            {at(0, i), at(1, i)}, {at(2, i), at(3, i)}, at(angle_row, i) * 180.0F / std::numbers::pi_v<float>};
        // Also skips NaN boxes.
        if (!(rect.size.width > 0.0F && rect.size.height > 0.0F) || std::isnan(rect.angle))
          continue;
        result[cls].push_back({rect, at(4 + cls, i), static_cast<std::int64_t>(cls)});
      }
    }
  }

  GeneralDetectParam param_;
  CellScanner scan_;
};

} // namespace

auto create_general_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<GeneralRotated>();
}

} // namespace hrt::postprocess
