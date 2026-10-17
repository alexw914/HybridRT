// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/rotated/e2e_rotated.cpp
// Description: Implements decoding of Ultralytics end-to-end OBB exports,
//              [1, N, 7] without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <numbers>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct E2eRotatedParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2eRotatedParam, class_num, confidence_threshold)

// One output [1, N, 7]: per row center x, center y, width, height in
// network-input pixels, score, class id and the angle in radians, as YOLO26
// (nms=False) and YOLOv8 (nms=True) OBB exports write them. Rows are already
// deduplicated; unused rows score 0.
class E2eRotated final : public RotatedPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<E2eRotatedParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<E2eRotatedParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 1 && outputs[0].shape.size() == 3 &&
                        outputs[0].shape[0] == 1 && outputs[0].shape[1] > 0 && outputs[0].shape[2] == 7,
                    false, "e2e_rotated: needs one input and one [1, N, 7] output");
    return true;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs,
              const preprocess::ImageTransform& transform) -> bool override {
    const auto* data = float_data(*outputs[0], scratch_);
    if (data == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[1]);
    for (std::size_t i = 0; i < count; ++i) {
      const auto* row = data + i * 7;
      const auto label = row[5];
      // Also skips NaN class ids, scores, sizes and angles.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      if (!(row[4] > param_.confidence_threshold[cls]) || !(row[2] > 0.0F && row[3] > 0.0F) || !std::isfinite(row[6]))
        continue;
      result[cls].push_back({{{row[0], row[1]}, {row[2], row[3]}, row[6] * 180.0F / std::numbers::pi_v<float>},
                             row[4],
                             static_cast<std::int64_t>(cls)});
    }
    return map_rotated(result, transform);
  }

private:
  E2eRotatedParam param_;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_e2e_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<E2eRotated>();
}

} // namespace hrt::postprocess
