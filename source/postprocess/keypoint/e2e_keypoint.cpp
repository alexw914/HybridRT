// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/e2e_keypoint.cpp
// Description: Implements decoding of Ultralytics end-to-end pose exports,
//              [1, N, 6 + keypoints * 3] without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct E2EKeypointParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 17;
  Vector<float> confidence_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2EKeypointParam, class_num, keypoint_num, confidence_threshold,
                                                keypoint_threshold)

// One output [1, N, 6 + keypoints * 3]: per row x1, y1, x2, y2, score
// and class id, then x, y and visibility per keypoint, all in network-input
// pixels, as YOLO26 (nms=False) and YOLOv8 (nms=True) pose exports write
// them. Rows are already deduplicated; unused rows score 0.
class E2EKeypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<E2EKeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "e2e_keypoint"))
      return false;
    param_ = config.get<E2EKeypointParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "e2e_keypoint: keypoint_num must be positive");
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 1 && outputs[0].shape.size() == 3 &&
                        outputs[0].shape[0] == 1 && outputs[0].shape[1] > 0,
                    false, "e2e_keypoint: needs one input and one [1, N, columns] output");
    KZ_CHECK_RETURN(outputs[0].shape[2] == 6 + static_cast<std::int64_t>(param_.keypoint_num * kKeypointValues), false,
                    "e2e_keypoint: output columns do not match keypoint_num");
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView output(*outputs[0], std::nullopt);
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[1]);
    const auto columns = outputs[0]->shape()[2];
    for (std::size_t i = 0; i < count; ++i) {
      const auto label = output.at(i, 5);
      // Also skips NaN class ids, scores and boxes.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      const auto score = output.at(i, 4);
      if (!(score > param_.confidence_threshold[cls]))
        continue;
      const core::Point2f tl{output.at(i, 0), output.at(i, 1)};
      const core::Point2f br{output.at(i, 2), output.at(i, 3)};
      if (!(br.x - tl.x > 0.0F && br.y - tl.y > 0.0F))
        continue;
      const auto box = transform.clip_to_source(core::Rect2f(tl, br));
      if (box.empty())
        continue;
      auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
      object.box = {box, score, static_cast<std::int64_t>(cls), -1};
      object.keypoints.reserve(param_.keypoint_num);
      for (std::int64_t c = 6; c < columns; c += static_cast<std::int64_t>(kKeypointValues)) {
        const auto visibility = output.at(i, c + 2);
        object.keypoints.push_back(
            {transform.to_source(core::Point2f{output.at(i, c), output.at(i, c + 1)}),
             visibility > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible,
             visibility});
      }
    }
    return true;
  }

private:
  E2EKeypointParam param_;
};

} // namespace

auto make_e2e_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<E2EKeypoint>();
}

} // namespace hrt::postprocess
