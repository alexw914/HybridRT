// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/instance_segment.cpp
// Description: Implements instance segmentation decoding: boxes with mask
//              coefficients plus mask prototypes.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <opencv2/imgproc.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

struct InstanceSegmentParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(InstanceSegmentParam, class_num, confidence_threshold, iou_threshold,
                                                mask_threshold)

// Outputs: detections [1, 4 + classes + masks, N] and prototypes
// [1, masks, mh, mw], identified by rank.
class InstanceSegment final : public SegmentPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<InstanceSegmentParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<InstanceSegmentParam>();
    if (param_.class_num == 0 ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold") ||
        !expand_per_class(param_.iou_threshold, param_.class_num, "iou_threshold"))
      return false;
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || outputs.size() != 2 || !engine_input_size(engine)) {
      KZ_LOG_ERROR("instance_segment: needs one image input and two NCHW outputs");
      return false;
    }
    detections_ = outputs[0].shape.size() == 3 ? 0 : 1;
    prototypes_ = 1 - detections_;
    const auto& det = outputs[detections_].shape;
    const auto& proto = outputs[prototypes_].shape;
    if (det.size() != 3 || proto.size() != 4 || det[0] != 1 || proto[0] != 1 ||
        det[1] != static_cast<std::int64_t>(param_.class_num) + 4 + proto[1]) {
      KZ_LOG_ERROR("instance_segment: outputs must be [1, 4 + class_num + masks, N] and [1, masks, h, w]");
      return false;
    }
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
    const auto count = static_cast<std::size_t>(outputs[detections_]->shape()[2]);
    const auto& proto_shape = outputs[prototypes_]->shape();
    const Prototypes prototypes{proto, static_cast<int>(proto_shape[1]), static_cast<int>(proto_shape[2]),
                                static_cast<int>(proto_shape[3])};

    const auto classes = param_.class_num;
    Vector<Vector<core::Rect2f>> rects(classes);
    Vector<Vector<float>> scores(classes);
    Vector<Vector<std::size_t>> anchors(classes);
    for (std::size_t i = 0; i < count; ++i) {
      const auto [l, t, r, b] = center_box(det[i], det[count + i], det[count * 2 + i], det[count * 3 + i]);
      if (!(r - l > 0.0F && b - t > 0.0F))
        continue;
      for (std::size_t cls = 0; cls < classes; ++cls) {
        const auto score = det[(4 + cls) * count + i];
        if (!(score > param_.confidence_threshold[cls]))
          continue;
        rects[cls].push_back(core::Rect2f::from_points({l, t}, {r, b}));
        scores[cls].push_back(score);
        anchors[cls].push_back(i);
      }
    }

    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    Vector<float> coefficients(static_cast<std::size_t>(prototypes.channels));
    for (std::size_t cls = 0; cls < classes; ++cls) {
      for (auto k : nms_keep(rects[cls], scores[cls], param_.iou_threshold[cls])) {
        const auto box = core::intersect(to_source(rects[cls][k], transform), bounds);
        if (box.empty())
          continue;
        for (std::size_t j = 0; j < coefficients.size(); ++j)
          coefficients[j] = det[(4 + classes + j) * count + anchors[cls][k]];
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores[cls][k], static_cast<std::int64_t>(cls), -1};
        object.mask = make_mask(prototypes, coefficients, rects[cls][k], box, transform);
      }
    }
    return true;
  }

private:
  struct Prototypes {
    const float* data;
    int channels;
    int height;
    int width;
  };

  // Mask = coefficients . prototypes inside the box, sampled bilinearly into
  // the box's source pixels and thresholded as a logit.
  auto make_mask(const Prototypes& prototypes, const Vector<float>& coefficients, const core::Rect2f& net_box,
                 const core::Rect2f& source_box, const preprocess::ImageTransform& transform) -> Mask {
    Mask mask;
    const auto x0 = static_cast<int>(std::floor(source_box.x));
    const auto y0 = static_cast<int>(std::floor(source_box.y));
    mask.rect = core::intersect(core::Rect{x0, y0, static_cast<int>(std::ceil(source_box.x + source_box.width)) - x0,
                                           static_cast<int>(std::ceil(source_box.y + source_box.height)) - y0},
                                transform.src_roi);
    if (mask.rect.empty())
      return mask;

    // Prototype pixels per network pixel, and the box region (plus a margin
    // for interpolation) in prototype space.
    const auto sx = static_cast<double>(prototypes.width) / transform.input_size.width;
    const auto sy = static_cast<double>(prototypes.height) / transform.input_size.height;
    const auto bx = static_cast<double>(net_box.x);
    const auto by = static_cast<double>(net_box.y);
    const auto bw = static_cast<double>(net_box.width);
    const auto bh = static_cast<double>(net_box.height);
    const auto region =
        core::intersect(core::Rect{static_cast<int>(std::floor(bx * sx)) - 1, static_cast<int>(std::floor(by * sy)) - 1,
                                   static_cast<int>(std::ceil(bw * sx)) + 3, static_cast<int>(std::ceil(bh * sy)) + 3},
                        core::Rect{0, 0, prototypes.width, prototypes.height});
    if (region.empty())
      return mask;
    logits_.create(region.height, region.width, CV_32F);
    logits_.setTo(0.0);
    const auto plane = static_cast<std::size_t>(prototypes.width) * static_cast<std::size_t>(prototypes.height);
    for (std::size_t j = 0; j < coefficients.size(); ++j) {
      const cv::Mat channel(prototypes.height, prototypes.width, CV_32F,
                            const_cast<float*>(prototypes.data + j * plane));
      cv::scaleAdd(channel(cv::Rect(region.x, region.y, region.width, region.height)),
                   static_cast<double>(coefficients[j]), logits_, logits_);
    }

    // Source pixel center -> network coordinate -> prototype pixel in region.
    const auto& dst = transform.dst_roi;
    const auto& src = transform.src_roi;
    const auto rw = static_cast<double>(dst.width) / src.width;
    const auto rh = static_cast<double>(dst.height) / src.height;
    const cv::Matx23d inverse(rw * sx, 0.0, ((mask.rect.x + 0.5 - src.x) * rw + dst.x) * sx - 0.5 - region.x, 0.0,
                              rh * sy, ((mask.rect.y + 0.5 - src.y) * rh + dst.y) * sy - 0.5 - region.y);
    cv::warpAffine(logits_, sampled_, inverse, {mask.rect.width, mask.rect.height},
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    mask.data.resize(static_cast<std::size_t>(mask.rect.width) * static_cast<std::size_t>(mask.rect.height));
    cv::Mat bits(mask.rect.height, mask.rect.width, CV_8U, mask.data.data());
    cv::compare(sampled_, static_cast<double>(mask_logit_), bits, cv::CMP_GT);
    cv::bitwise_and(bits, cv::Scalar(1), bits);
    return mask;
  }

  InstanceSegmentParam param_;
  std::size_t detections_ = 0;
  std::size_t prototypes_ = 1;
  float mask_logit_ = 0.0F;
  Array<core::Buffer, 2> scratch_{core::Buffer(core::MemoryType::Cpu, core::DataType::Float32),
                                  core::Buffer(core::MemoryType::Cpu, core::DataType::Float32)};
  cv::Mat logits_;
  cv::Mat sampled_;
};

} // namespace

auto make_instance_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<InstanceSegment>();
}

} // namespace hrt::postprocess
