// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/general_yolo_segment.cpp
// Description: Implements decoding of single-output YOLO instance
//              segmentation, [1, 4 + classes + masks, N] plus prototypes.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/mask_decoder.hpp"
#include "include/postprocess/nms.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

struct GeneralYoloSegmentParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralYoloSegmentParam, class_num, confidence_threshold, iou_threshold,
                                                mask_threshold)

// Outputs: detections [1, 4 + classes + masks, N] and prototypes
// [batch, masks, mh, mw], identified by rank.
class GeneralYoloSegment final : public SegmentPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "general_yolo_segment"))
      return false;
    param_ = config.get<GeneralYoloSegmentParam>();
    if (!expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || outputs.size() != 2) {
      KZ_LOG_ERROR("general_yolo_segment: needs two NCHW outputs");
      return false;
    }
    detections_ = outputs[0].shape.size() == 3 ? 0 : 1;
    prototypes_ = 1 - detections_;
    const auto& det = outputs[detections_].shape;
    const auto& proto = outputs[prototypes_].shape;
    if (det.size() != 3 || proto.size() != 4 || det[0] < 1 || proto[0] != det[0] ||
        det[1] != static_cast<std::int64_t>(param_.class_num) + 4 + proto[1]) {
      KZ_LOG_ERROR("general_yolo_segment: outputs must be [batch, 4 + class_num + masks, N] and [batch, masks, h, w]");
      return false;
    }
    mask_logit_ = core::logit(param_.mask_threshold);
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<GeneralYoloSegmentParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, std::size_t index,
              const preprocess::ImageTransform& transform) -> bool override {
    const HeadGridView prototypes(*outputs[prototypes_], index, engine::DataLayout::Nchw);
    const HeadGridView det(*outputs[detections_], index, std::nullopt);
    decode_rows(result, det, transform, prototypes, static_cast<std::size_t>(outputs[prototypes_]->shape()[1]));
    return true;
  }

private:
  auto decode_rows(SegmentResultMap& result, const HeadGridView& det, const preprocess::ImageTransform& transform,
                   const HeadGridView& prototypes, std::size_t masks) -> void {
    const auto at = [&](std::size_t row, std::uint32_t i) -> float { return det.at(0, i, row); };
    const auto classes = param_.class_num;
    rects_.resize(classes);
    scores_.resize(classes);
    anchors_.resize(classes);
    // Class rows are read in order; walking anchors first strides every row
    // per anchor. Boxes are only built for scores above the threshold.
    for (std::size_t cls = 0; cls < classes; ++cls) {
      rects_[cls].clear();
      scores_[cls].clear();
      anchors_[cls].clear();
      for (const auto i : scan_.scan_channel(det, 4 + cls, param_.confidence_threshold[cls])) {
        const auto rect = core::center_box(at(0, i), at(1, i), at(2, i), at(3, i));
        if (!core::valid_box(rect))
          continue;
        rects_[cls].push_back(rect);
        scores_[cls].push_back(at(4 + cls, i));
        anchors_[cls].push_back(i);
      }
    }

    Vector<float> coefficients(masks);
    for (std::size_t cls = 0; cls < classes; ++cls) {
      for (auto k : nms_indices(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = transform.clip_reverse(rects_[cls][k]);
        if (box.empty())
          continue;
        for (std::size_t j = 0; j < coefficients.size(); ++j)
          coefficients[j] = at(4 + classes + j, anchors_[cls][k]);
        auto& objects = result[static_cast<std::int64_t>(cls)];
        objects.emplace_back().box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        masks_.add(coefficients, rects_[cls][k], box, static_cast<std::int64_t>(cls), objects.size() - 1);
      }
    }
    masks_.decode(prototypes, transform, mask_logit_, result);
  }

  GeneralYoloSegmentParam param_;
  CellScanner scan_;
  Vector<Vector<core::Rect2f>> rects_;
  Vector<Vector<float>> scores_;
  Vector<Vector<std::uint32_t>> anchors_;
  std::size_t detections_ = 0;
  std::size_t prototypes_ = 1;
  float mask_logit_ = 0.0F;
  MaskBatch masks_;
};

} // namespace

auto create_general_yolo_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<GeneralYoloSegment>();
}

} // namespace hrt::postprocess
