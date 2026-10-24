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

struct E2ERotatedParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2ERotatedParam, class_num, confidence_threshold)

// One output [1, N, 7]: per row center x, center y, width, height in
// network-input pixels, score, class id and the angle in radians, as YOLO26
// (nms=False) and YOLOv8 (nms=True) OBB exports write them. Rows are already
// deduplicated; unused rows score 0.
class E2ERotated final : public RotatedPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "e2e_rotated"))
      return false;
    param_ = config.get<E2ERotatedParam>();
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 1 && outputs[0].shape.size() == 3 &&
                        outputs[0].shape[0] >= 1 && outputs[0].shape[1] > 0 && outputs[0].shape[2] == 7,
                    false, "e2e_rotated: needs one input and one [batch, N, 7] output");
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<E2ERotatedParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(RotatedResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView output(*outputs[0], std::nullopt);
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[1]);
    for (std::size_t i = 0; i < count; ++i) {
      const auto label = output.at(i, 5);
      // Also skips NaN class ids, scores, sizes and angles.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      const auto score = output.at(i, 4);
      if (!(score > param_.confidence_threshold[cls]))
        continue;
      const auto width = output.at(i, 2);
      const auto height = output.at(i, 3);
      const auto angle = output.at(i, 6);
      if (!(width > 0.0F && height > 0.0F) || !std::isfinite(angle))
        continue;
      result[cls].push_back(
          {{{output.at(i, 0), output.at(i, 1)}, {width, height}, angle * 180.0F / std::numbers::pi_v<float>},
           score,
           static_cast<std::int64_t>(cls)});
    }
    return rotated_reverse(result, transform);
  }

private:
  E2ERotatedParam param_;
};

} // namespace

auto create_e2e_rotated() -> UniquePtr<RotatedPostprocessor> {
  return std::make_unique<E2ERotated>();
}

} // namespace hrt::postprocess
