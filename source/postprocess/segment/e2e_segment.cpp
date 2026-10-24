// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/e2e_segment.cpp
// Description: Implements decoding of Ultralytics end-to-end segment exports,
//              [1, N, 6 + masks] plus prototypes, without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/mask_decoder.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

struct E2ESegmentParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2ESegmentParam, class_num, confidence_threshold, mask_threshold)

// One detection output [1, N, 6 + masks]: per row x1, y1, x2, y2 in
// network-input pixels, score, class id and mask coefficients, as YOLO26
// (nms=False) and YOLOv8 (nms=True) segment exports write them, plus the
// prototypes [batch, masks, mh, mw]. Rows are already deduplicated; unused rows
// score 0.
class E2ESegment final : public SegmentPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "e2e_segment"))
      return false;
    param_ = config.get<E2ESegmentParam>();
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.proto().output_layout == engine::DataLayout::Nchw && outputs.size() == 2, false,
                    "e2e_segment: needs two NCHW outputs");
    detections_ = outputs[0].shape.size() == 3 ? 0 : 1;
    prototypes_ = 1 - detections_;
    const auto& det = outputs[detections_].shape;
    const auto& proto = outputs[prototypes_].shape;
    KZ_CHECK_RETURN(det.size() == 3 && proto.size() == 4 && det[0] >= 1 && proto[0] == det[0] && det[1] > 0 &&
                        det[2] == 6 + proto[1],
                    false, "e2e_segment: outputs must be [batch, N, 6 + masks] and [batch, masks, h, w]");
    mask_logit_ = core::logit(param_.mask_threshold);
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<E2ESegmentParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView det(*outputs[detections_], std::nullopt);
    const HeadGridView prototypes(*outputs[prototypes_], engine::DataLayout::Nchw);
    const auto count = static_cast<std::size_t>(outputs[detections_]->shape()[1]);
    const auto columns = outputs[detections_]->shape()[2];
    for (std::size_t i = 0; i < count; ++i) {
      const auto label = det.at(i, 5);
      // Also skips NaN class ids, scores and boxes.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      const auto score = det.at(i, 4);
      if (!(score > param_.confidence_threshold[cls]))
        continue;
      const core::Point2f tl{det.at(i, 0), det.at(i, 1)};
      const core::Point2f br{det.at(i, 2), det.at(i, 3)};
      if (!(br.x - tl.x > 0.0F && br.y - tl.y > 0.0F))
        continue;
      const auto net_box = core::Rect2f(tl, br);
      const auto box = transform.clip_reverse(net_box);
      if (box.empty())
        continue;
      coefficients_.clear();
      for (std::int64_t c = 6; c < columns; ++c)
        coefficients_.push_back(det.at(i, c));
      auto& objects = result[static_cast<std::int64_t>(cls)];
      objects.emplace_back().box = {box, score, static_cast<std::int64_t>(cls), -1};
      masks_.add(coefficients_, net_box, box, static_cast<std::int64_t>(cls), objects.size() - 1);
    }
    masks_.decode(prototypes, transform, mask_logit_, result);
    return true;
  }

private:
  E2ESegmentParam param_;
  std::size_t detections_ = 0;
  std::size_t prototypes_ = 1;
  float mask_logit_ = 0.0F;
  Vector<float> coefficients_;
  MaskBatch masks_;
};

} // namespace

auto create_e2e_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<E2ESegment>();
}

} // namespace hrt::postprocess
