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
#include "include/postprocess/params.hpp"

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
// prototypes [1, masks, mh, mw]. Rows are already deduplicated; unused rows
// score 0.
class E2ESegment final : public SegmentPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<E2ESegmentParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<E2ESegmentParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(engine.proto().output_layout == engine::DataLayout::Nchw && outputs.size() == 2 &&
                        engine.input_size(),
                    false, "e2e_segment: needs one image input and two NCHW outputs");
    detections_ = outputs[0].shape.size() == 3 ? 0 : 1;
    prototypes_ = 1 - detections_;
    const auto& det = outputs[detections_].shape;
    const auto& proto = outputs[prototypes_].shape;
    KZ_CHECK_RETURN(det.size() == 3 && proto.size() == 4 && det[0] == 1 && proto[0] == 1 && det[1] > 0 &&
                        det[2] == 6 + proto[1],
                    false, "e2e_segment: outputs must be [1, N, 6 + masks] and [1, masks, h, w]");
    mask_logit_ = logit(param_.mask_threshold);
    return true;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto* det = float_data(*outputs[detections_], scratch_[0]);
    const auto* proto = float_data(*outputs[prototypes_], scratch_[1]);
    if (det == nullptr || proto == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(outputs[detections_]->shape()[1]);
    const auto columns = static_cast<std::size_t>(outputs[detections_]->shape()[2]);
    const auto& proto_shape = outputs[prototypes_]->shape();
    const MaskDecoder::Prototypes prototypes{proto, static_cast<int>(proto_shape[1]), static_cast<int>(proto_shape[2]),
                                             static_cast<int>(proto_shape[3])};
    for (std::size_t i = 0; i < count; ++i) {
      const auto* row = det + i * columns;
      const auto label = row[5];
      // Also skips NaN class ids, scores and boxes.
      if (!(label >= 0.0F && label < static_cast<float>(param_.class_num)))
        continue;
      const auto cls = static_cast<std::size_t>(label);
      if (!(row[4] > param_.confidence_threshold[cls]) || !(row[2] - row[0] > 0.0F && row[3] - row[1] > 0.0F))
        continue;
      const auto net_box = core::Rect2f::from_points({row[0], row[1]}, {row[2], row[3]});
      const auto box = clip_to_source(net_box, transform);
      if (box.empty())
        continue;
      coefficients_.assign(row + 6, row + columns);
      auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
      object.box = {box, row[4], static_cast<std::int64_t>(cls), -1};
      object.mask = masks_.make(prototypes, coefficients_, net_box, box, transform, mask_logit_);
    }
    return true;
  }

private:
  E2ESegmentParam param_;
  std::size_t detections_ = 0;
  std::size_t prototypes_ = 1;
  float mask_logit_ = 0.0F;
  Vector<float> coefficients_;
  Array<core::Buffer, 2> scratch_{core::Buffer(core::MemoryType::Cpu, core::DataType::Float32),
                                  core::Buffer(core::MemoryType::Cpu, core::DataType::Float32)};
  MaskDecoder masks_;
};

} // namespace

auto make_e2e_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<E2ESegment>();
}

} // namespace hrt::postprocess
