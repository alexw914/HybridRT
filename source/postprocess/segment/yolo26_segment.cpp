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
  Vector<float> strides;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Yolo26SegmentParam, class_num, strides, confidence_threshold,
                                                iou_threshold, mask_threshold)

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
    param_ = config.get<Yolo26SegmentParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    std::size_t box_channels = 4;
    const auto outputs = raw_segment_outputs(engine, param_.strides, box_channels, param_.class_num, "yolo26_segment");
    if (!outputs)
      return false;
    outputs_ = *outputs;
    // Scores are compared as logits so most cells skip the sigmoid.
    logit_threshold_.resize(param_.class_num);
    std::transform(param_.confidence_threshold.begin(), param_.confidence_threshold.end(), logit_threshold_.begin(),
                   logit);
    mask_logit_ = logit(param_.mask_threshold);
    layout_ = engine.proto().output_layout;
    return true;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto* proto = float_data(*outputs[outputs_.prototypes], scratch_);
    if (proto == nullptr)
      return false;
    const auto& proto_shape = outputs[outputs_.prototypes]->shape();
    const MaskDecoder::Prototypes prototypes{proto, static_cast<int>(proto_shape[1]), static_cast<int>(proto_shape[2]),
                                             static_cast<int>(proto_shape[3])};
    rects_.resize(param_.class_num);
    scores_.resize(param_.class_num);
    coefficients_.resize(param_.class_num);
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      rects_[cls].clear();
      scores_[cls].clear();
      coefficients_[cls].clear();
    }
    for (std::size_t k = 0; k < outputs_.heads.size(); ++k) {
      const auto ok = HeadGrid<>::visit(*outputs[outputs_.heads[k]], layout_,
                                        [&](const auto& grid) -> void { decode_head(grid, param_.strides[k]); });
      if (!ok)
        return false;
    }
    const auto masks = outputs_.masks;
    for (std::size_t cls = 0; cls < param_.class_num; ++cls) {
      for (auto k : nms_keep(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = clip_to_source(rects_[cls][k], transform);
        if (box.empty())
          continue;
        const auto* first = coefficients_[cls].data() + k * masks;
        mask_coefficients_.assign(first, first + masks);
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        object.mask = masks_.make(prototypes, mask_coefficients_, rects_[cls][k], box, transform, mask_logit_);
      }
    }
    return true;
  }

private:
  template <typename Grid>
  auto decode_head(const Grid& grid, float stride) -> void {
    const auto first_mask = 4 + param_.class_num;
    const auto min_score = grid.raw_bound(*std::min_element(logit_threshold_.begin(), logit_threshold_.end()));
    for (const auto cell : scan_.scan(grid, 4, param_.class_num, min_score)) {
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
        rects_[cls].push_back(
            core::Rect2f::from_points({(cx - grid.at(y, x, 0)) * stride, (cy - grid.at(y, x, 1)) * stride},
                                      {(cx + grid.at(y, x, 2)) * stride, (cy + grid.at(y, x, 3)) * stride}));
        scores_[cls].push_back(sigmoid(score));
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
  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
  MaskDecoder masks_;
};

} // namespace

auto make_yolo26_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<Yolo26Segment>();
}

} // namespace hrt::postprocess
