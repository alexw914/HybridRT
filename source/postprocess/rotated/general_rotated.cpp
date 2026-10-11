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
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// One output [1, 4 + classes + 1, N]: rows center x, center y, width, height,
// one score per class, then the angle in radians.
class GeneralRotated final : public RotatedPostprocessor {
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
        outputs[0].shape[1] != static_cast<std::int64_t>(param_.class_num) + 5) {
      KZ_LOG_ERROR("general_rotated: needs one input and one [1, 4 + class_num + 1, N] output");
      return false;
    }
    return true;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto* data = float_data(*outputs[0], scratch_);
    if (data == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[2]);
    const auto angle_row = (param_.class_num + 4) * count;
    for (std::size_t i = 0; i < count; ++i) {
      const core::RotatedRect rect{{data[i], data[count + i]},
                                   {data[count * 2 + i], data[count * 3 + i]},
                                   data[angle_row + i] * 180.0F / std::numbers::pi_v<float>};
      // Also skips NaN boxes.
      if (!(rect.size.width > 0.0F && rect.size.height > 0.0F) || std::isnan(rect.angle))
        continue;
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        const auto score = data[(4 + cls) * count + i];
        if (score > param_.confidence_threshold[cls])
          result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
      }
    }
    return finish_rotated(result, param_.iou_threshold, transform);
  }

private:
  GeneralDetectParam param_;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_general_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<GeneralRotated>();
}

} // namespace hrt::postprocess
