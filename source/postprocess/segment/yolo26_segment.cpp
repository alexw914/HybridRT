// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/yolo26_segment.cpp
// Description: Implements decoding of raw YOLO26 segmentation heads, one
//              output per stride plus the mask prototypes.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>

#include "include/postprocess/common.hpp"
#include "include/postprocess/mask_decoder.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct Yolo26SegmentParam {
  unsigned int class_num = 0;
  Vector<core::Point2f> strides; // Inferred in init from the input and head sizes; not a config key.
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Yolo26SegmentParam, class_num, confidence_threshold, iou_threshold,
                                                mask_threshold)

// One output per stride with channels left, top, right and bottom distances
// in cells (no distance bins), class logits and mask coefficients, as the
// one-to-many head of a YOLO26 segment model emits them, plus the prototypes
// [1, masks, mh, mw] in any output position. NMS still runs.
class Yolo26Segment final : public SegmentPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<Yolo26SegmentParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "yolo26_segment"))
      return false;
    param_ = config.get<Yolo26SegmentParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    std::size_t box_channels = 4;
    const auto outputs =
        find_raw_segment_outputs(engine, param_.strides, box_channels, param_.class_num, "yolo26_segment");
    if (!outputs)
      return false;
    outputs_ = *outputs;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   core::logit);
    mask_logit_ = core::logit(param_.mask_threshold);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const HeadGridView prototypes(*outputs[outputs_.prototypes], layout_);
    rects_.resize(param_.class_num);
    scores_.resize(param_.class_num);
    coefficients_.resize(param_.class_num);
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      rects_[cls].clear();
      scores_[cls].clear();
      coefficients_[cls].clear();
    }
    for (std::size_t k = 0; k < outputs_.heads.size(); ++k) {
      const HeadGridView grid(*outputs[outputs_.heads[k]], layout_);
      decode_head(grid, param_.strides[k]);
    }
    const auto masks = outputs_.masks;
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      for (auto k : nms(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = transform.clip_to_source(rects_[cls][k]);
        if (box.empty())
          continue;
        const auto* first = coefficients_[cls].data() + k * masks;
        mask_coefficients_.assign(first, first + masks);
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        object.mask = masks_.decode(prototypes, mask_coefficients_, rects_[cls][k], box, transform, mask_logit_);
      }
    }
    return true;
  }

private:
  auto decode_head(const HeadGridView& grid, core::Point2f stride) -> void {
    const auto first_mask = 4 + param_.class_num;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan_max_score(grid, 4, param_.class_num, min_score)) {
      const std::int64_t y = cell / grid.cols();
      const std::int64_t x = cell % grid.cols();
      for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
        if (grid.raw(y, x, 4 + cls) < min_score)
          continue;
        const auto score = grid.at(y, x, 4 + cls);
        if (score < logit_threshold_[cls])
          continue;
        const auto cx = static_cast<float>(x) + 0.5F;
        const auto cy = static_cast<float>(y) + 0.5F;
        rects_[cls].push_back(core::Rect2f({(cx - grid.at(y, x, 0)) * stride.x, (cy - grid.at(y, x, 1)) * stride.y},
                                           {(cx + grid.at(y, x, 2)) * stride.x, (cy + grid.at(y, x, 3)) * stride.y}));
        scores_[cls].push_back(core::sigmoid(score));
        for (std::size_t j = 0; j < outputs_.masks; ++j)
          coefficients_[cls].push_back(grid.at(y, x, first_mask + j));
      }
    }
  }

  Yolo26SegmentParam param_;
  RawSegmentOutputs outputs_;
  CellScanner scan_;
  Vector<float> logit_threshold_;
  float mask_logit_ = 0.0F;
  engine::DataLayout layout_ = engine::DataLayout::Nchw;
  Vector<Vector<core::Rect2f>> rects_;
  Vector<Vector<float>> scores_;
  Vector<Vector<float>> coefficients_;
  Vector<float> mask_coefficients_;
  MaskDecoder masks_;
};

} // namespace

auto make_yolo26_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<Yolo26Segment>();
}

} // namespace hrt::postprocess
