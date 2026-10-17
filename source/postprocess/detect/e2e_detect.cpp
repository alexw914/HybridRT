// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/e2e_detect.cpp
// Description: Implements decoding of Ultralytics end-to-end detection
//              exports, [1, N, 6] without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct E2EDetectParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2EDetectParam, class_num, confidence_threshold)

// One output [1, N, 6]: per row x1, y1, x2, y2 in network-input pixels, score
// and class id, as YOLOv10/YOLO26 (and YOLOv8 with nms=True) export them.
// Rows are already deduplicated; unused rows score 0.
class E2EDetect final : public DetectPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<E2EDetectParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<E2EDetectParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.input_protos().size() == 1 && outputs.size() == 1 && outputs[0].shape.size() == 3 &&
                        outputs[0].shape[0] == 1 && outputs[0].shape[1] > 0 && outputs[0].shape[2] == 6,
                    false, "e2e_detect: needs one input and one [1, N, 6] output");
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs,
              const preprocess::ImageTransform& transform) -> bool override {
    const auto* data = float_data(*outputs[0], scratch_);
    if (data == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[0]->shape()[1]);
    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    for (std::size_t i = 0; i < count; ++i) {
      const auto* row = data + i * 6;
      const auto score = row[4];
      const auto label = row[5];
      // Also skips NaN scores and class ids.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      if (!(score > param_.confidence_threshold[cls]) || !(row[2] - row[0] > 0.0F && row[3] - row[1] > 0.0F))
        continue;
      const auto rect =
          core::intersect(to_source(core::Rect2f::from_points({row[0], row[1]}, {row[2], row[3]}), transform), bounds);
      if (!rect.empty())
        result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
    }
    return true;
  }

private:
  E2EDetectParam param_;
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_e2e_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<E2EDetect>();
}

} // namespace hrt::postprocess
