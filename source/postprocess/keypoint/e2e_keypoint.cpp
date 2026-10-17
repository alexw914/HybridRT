// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/e2e_keypoint.cpp
// Description: Implements decoding of Ultralytics end-to-end pose exports,
//              [1, N, 6 + keypoints * (2 | 3)] without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct E2EKeypointParam {
  unsigned int class_num = 0;
  unsigned int keypoint_num = 0;
  Vector<float> confidence_threshold;
  // Visibility score above which a keypoint counts as visible.
  float keypoint_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2EKeypointParam, class_num, keypoint_num, confidence_threshold,
                                                keypoint_threshold)

// One output [1, N, 6 + keypoints * (2 | 3)]: per row x1, y1, x2, y2, score
// and class id, then x, y (and visibility) per keypoint, all in network-input
// pixels, as YOLO26 (nms=False) and YOLOv8 (nms=True) pose exports write
// them. Rows are already deduplicated; unused rows score 0.
class E2EKeypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<E2EKeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<E2EKeypointParam>();
    KZ_CHECK_RETURN(param_.keypoint_num > 0, false, "e2e_keypoint: keypoint_num must be positive");
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 1 && outputs[0].shape.size() == 3 &&
                        outputs[0].shape[0] == 1 && outputs[0].shape[1] > 0,
                    false, "e2e_keypoint: needs one input and one [1, N, columns] output");
    const auto extra = outputs[0].shape[2] - 6;
    const auto keypoints = static_cast<std::int64_t>(param_.keypoint_num);
    dims_ = extra == keypoints * 3 ? 3 : (extra == keypoints * 2 ? 2 : 0);
    KZ_CHECK_RETURN(dims_ != 0, false, "e2e_keypoint: {} columns do not fit 6 + {} keypoints x (2 | 3)",
                    outputs[0].shape[2], param_.keypoint_num);
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs,
              const preprocess::ImageTransform& transform) -> bool override {
    const auto* data = float_data(*outputs[0], scratch_);
    if (data == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[1]);
    const auto columns = static_cast<std::size_t>(outputs[0]->shape()[2]);
    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    for (std::size_t i = 0; i < count; ++i) {
      const auto* row = data + i * columns;
      const auto label = row[5];
      // Also skips NaN class ids, scores and boxes.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      if (!(row[4] > param_.confidence_threshold[cls]) || !(row[2] - row[0] > 0.0F && row[3] - row[1] > 0.0F))
        continue;
      const auto box =
          core::intersect(to_source(core::Rect2f::from_points({row[0], row[1]}, {row[2], row[3]}), transform), bounds);
      if (box.empty())
        continue;
      auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
      object.box = {box, row[4], static_cast<std::int64_t>(cls), -1};
      object.keypoints.reserve(param_.keypoint_num);
      for (const auto* point = row + 6; point < row + columns; point += dims_) {
        const auto score = dims_ == 3 ? point[2] : 1.0F;
        object.keypoints.push_back(
            {to_source(core::Point2f{point[0], point[1]}, transform),
             score > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible, score});
      }
    }
    return true;
  }

private:
  E2EKeypointParam param_;
  std::size_t dims_ = 3;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_e2e_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<E2EKeypoint>();
}

} // namespace hrt::postprocess
