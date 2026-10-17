// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/instance_segment.cpp
// Description: Implements instance segmentation decoding, boxes with mask
//              coefficients plus mask prototypes, for raw heads and
//              Ultralytics end-to-end exports.
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

// Mask decoding shared by both formats; keeps its scratch images.
class MaskDecoder {
public:
  struct Prototypes {
    const float* data;
    int channels;
    int height;
    int width;
  };

  // Mask = coefficients . prototypes inside the box, sampled bilinearly into
  // the box's source pixels and thresholded as a logit.
  auto make(const Prototypes& prototypes, const Vector<float>& coefficients, const core::Rect2f& net_box,
            const core::Rect2f& source_box, const preprocess::ImageTransform& transform, float mask_logit) -> Mask {
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

    // Source pixel center -> network coordinate -> prototype pixel in region:
    // p = a * index + b per axis. Bilinear sampling is separable since the map
    // only scales and shifts: each region row is interpolated across once,
    // then each mask row blends two of them and thresholds in the same pass.
    // Coordinates outside the region repeat its edge.
    const auto& dst = transform.dst_roi;
    const auto& src = transform.src_roi;
    const auto rw = static_cast<double>(dst.width) / src.width;
    const auto rh = static_cast<double>(dst.height) / src.height;
    const auto width = mask.rect.width;
    const auto height = mask.rect.height;
    taps(columns_, width, rw * sx, ((mask.rect.x + 0.5 - src.x) * rw + dst.x) * sx - 0.5 - region.x, region.width);
    taps(rows_, height, rh * sy, ((mask.rect.y + 0.5 - src.y) * rh + dst.y) * sy - 0.5 - region.y, region.height);
    across_.resize(static_cast<std::size_t>(region.height) * static_cast<std::size_t>(width));
    for (int y = 0; y < region.height; ++y) {
      const auto* in = logits_.ptr<float>(y);
      auto* out = across_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      for (int x = 0; x < width; ++x) {
        const auto& tap = columns_[static_cast<std::size_t>(x)];
        out[x] = in[tap.first] + (in[tap.second] - in[tap.first]) * tap.weight;
      }
    }
    mask.data.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
      const auto& tap = rows_[static_cast<std::size_t>(y)];
      const auto* top = across_.data() + static_cast<std::size_t>(tap.first) * static_cast<std::size_t>(width);
      const auto* bottom = across_.data() + static_cast<std::size_t>(tap.second) * static_cast<std::size_t>(width);
      auto* bits = mask.data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
      for (int x = 0; x < width; ++x)
        bits[x] = top[x] + (bottom[x] - top[x]) * tap.weight > mask_logit ? 1 : 0;
    }
    return mask;
  }

private:
  // Bilinear neighbors and the weight of the second along one axis.
  struct Tap {
    int first = 0;
    int second = 0;
    float weight = 0.0F;
  };

  static auto taps(Vector<Tap>& out, int count, double scale, double offset, int size) -> void {
    out.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      const auto position = scale * i + offset;
      const auto first = static_cast<int>(std::floor(position));
      auto& tap = out[static_cast<std::size_t>(i)];
      if (first < 0 || size == 1)
        tap = {0, 0, 0.0F};
      else if (first >= size - 1)
        tap = {size - 1, size - 1, 0.0F};
      else
        tap = {first, first + 1, static_cast<float>(position - first)};
    }
  }

  cv::Mat logits_;
  Vector<Tap> columns_;
  Vector<Tap> rows_;
  Vector<float> across_;
};

struct E2ESegmentParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  float mask_threshold = 0.5F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(E2ESegmentParam, class_num, confidence_threshold, mask_threshold)

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
    const RowView det(*outputs[detections_], scratch_[0]);
    const auto* proto = float_data(*outputs[prototypes_], scratch_[1]);
    if (!det.valid() || proto == nullptr)
      return false;
    const auto& proto_shape = outputs[prototypes_]->shape();
    const MaskDecoder::Prototypes prototypes{proto, static_cast<int>(proto_shape[1]), static_cast<int>(proto_shape[2]),
                                             static_cast<int>(proto_shape[3])};

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
      candidates_.clear();
      det.above(4 + cls, param_.confidence_threshold[cls], candidates_);
      for (const auto i : candidates_) {
        const auto [l, t, r, b] = center_box(det.at(0, i), det.at(1, i), det.at(2, i), det.at(3, i));
        if (!(r - l > 0.0F && b - t > 0.0F))
          continue;
        rects_[cls].push_back(core::Rect2f::from_points({l, t}, {r, b}));
        scores_[cls].push_back(det.at(4 + cls, i));
        anchors_[cls].push_back(i);
      }
    }

    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    Vector<float> coefficients(static_cast<std::size_t>(prototypes.channels));
    for (std::size_t cls = 0; cls < classes; ++cls) {
      for (auto k : nms_keep(rects_[cls], scores_[cls], param_.iou_threshold[cls])) {
        const auto box = core::intersect(to_source(rects_[cls][k], transform), bounds);
        if (box.empty())
          continue;
        for (std::size_t j = 0; j < coefficients.size(); ++j)
          coefficients[j] = det.at(4 + classes + j, anchors_[cls][k]);
        auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
        object.box = {box, scores_[cls][k], static_cast<std::int64_t>(cls), -1};
        object.mask = masks_.make(prototypes, coefficients, rects_[cls][k], box, transform, mask_logit_);
      }
    }
    return true;
  }

private:
  InstanceSegmentParam param_;
  Vector<std::uint32_t> candidates_;
  Vector<Vector<core::Rect2f>> rects_;
  Vector<Vector<float>> scores_;
  Vector<Vector<std::uint32_t>> anchors_;
  std::size_t detections_ = 0;
  std::size_t prototypes_ = 1;
  float mask_logit_ = 0.0F;
  Array<core::Buffer, 2> scratch_{core::Buffer(core::MemoryType::Cpu, core::DataType::Float32),
                                  core::Buffer(core::MemoryType::Cpu, core::DataType::Float32)};
  MaskDecoder masks_;
};

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
                        engine_input_size(engine),
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
    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
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
      const auto box = core::intersect(to_source(net_box, transform), bounds);
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

auto make_instance_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<InstanceSegment>();
}

} // namespace hrt::postprocess
