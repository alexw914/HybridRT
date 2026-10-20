// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/mask_decoder.cpp
// Description: Implements instance mask decoding from coefficients and
//              prototypes.
// -----------------------------------------------------------------------------

#include "include/postprocess/mask_decoder.hpp"

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <opencv2/imgproc.hpp>

#include "include/postprocess/common.hpp"

namespace hrt::postprocess {

namespace {

// Adds weights[j] * channel j of the region, read as T, to `logits`.
template <typename T>
auto accumulate(const HeadGridView& grid, const Vector<float>& weights, const core::Rect& region, cv::Mat& logits)
    -> void {
  const auto step = grid.col_step();
  for (std::size_t j = 0; j < weights.size(); ++j) {
    const auto weight = weights[j];
    for (int y = 0; y < region.height; ++y) {
      const auto* in = grid.row<T>(region.y + y, j) + static_cast<std::size_t>(region.x) * step;
      auto* out = logits.ptr<float>(y);
      // Contiguous NCHW rows vectorize; NHWC strides over the channels.
      if (step == 1) {
        for (int x = 0; x < region.width; ++x)
          out[x] += weight * static_cast<float>(in[x]);
      } else {
        for (int x = 0; x < region.width; ++x)
          out[x] += weight * static_cast<float>(in[static_cast<std::size_t>(x) * step]);
      }
    }
  }
}

} // namespace

auto MaskDecoder::decode(const HeadGridView& prototypes, const Vector<float>& coefficients, const core::Rect2f& net_box,
                         const core::Rect2f& source_box, const preprocess::ImageTransform& transform, float mask_logit)
    -> Mask {
  Mask mask;
  mask.rect = core::intersect(core::cover_pixels(source_box), transform.src_roi);
  if (mask.rect.empty())
    return mask;

  // Prototype pixels per network pixel, and the box region (plus a margin
  // for interpolation) in prototype space.
  const auto proto_width = static_cast<int>(prototypes.cols());
  const auto proto_height = static_cast<int>(prototypes.rows());
  const auto sx = static_cast<double>(proto_width) / transform.input_size.width;
  const auto sy = static_cast<double>(proto_height) / transform.input_size.height;
  const auto bx = static_cast<double>(net_box.x);
  const auto by = static_cast<double>(net_box.y);
  const auto bw = static_cast<double>(net_box.width);
  const auto bh = static_cast<double>(net_box.height);
  const auto region =
      core::intersect(core::Rect{static_cast<int>(std::floor(bx * sx)) - 1, static_cast<int>(std::floor(by * sy)) - 1,
                                 static_cast<int>(std::ceil(bw * sx)) + 3, static_cast<int>(std::ceil(bh * sy)) + 3},
                      core::Rect{0, 0, proto_width, proto_height});
  if (region.empty())
    return mask;
  logits_.create(region.height, region.width, CV_32F);
  if (prototypes.int8()) {
    // sum_j c_j * scale * (q_j - zero_point): the scale folds into the
    // weights and the zero point into the starting value.
    weights_.resize(coefficients.size());
    float sum = 0.0F;
    for (std::size_t j = 0; j < coefficients.size(); ++j) {
      weights_[j] = coefficients[j] * prototypes.scale();
      sum += weights_[j];
    }
    logits_.setTo(-static_cast<double>(sum) * prototypes.zero_point());
    accumulate<std::int8_t>(prototypes, weights_, region, logits_);
  } else {
    logits_.setTo(0.0);
    accumulate<float>(prototypes, coefficients, region, logits_);
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
  bilinear_taps(columns_, width, rw * sx, ((mask.rect.x + 0.5 - src.x) * rw + dst.x) * sx - 0.5 - region.x,
                region.width);
  bilinear_taps(rows_, height, rh * sy, ((mask.rect.y + 0.5 - src.y) * rh + dst.y) * sy - 0.5 - region.y,
                region.height);
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

auto MaskDecoder::bilinear_taps(Vector<Tap>& out, int count, double scale, double offset, int size) -> void {
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

auto find_raw_segment_outputs(const engine::Engine& engine, Vector<core::Point2f>& strides, std::size_t& box_channels,
                              std::size_t classes, const char* name) -> std::optional<RawSegmentOutputs> {
  const auto& outputs = engine.output_protos();
  const auto layout = engine.proto().output_layout;
  const auto input = *engine.input_size();
  KZ_CHECK_RETURN(outputs.size() >= 2 && is_4d_outputs(engine, name), std::nullopt,
                  "{}: needs 4-D raw heads plus the prototypes", name);
  std::optional<RawSegmentOutputs> selected;
  Vector<core::Point2f> inferred;
  std::size_t selected_channels = 0;
  for (std::size_t candidate = 0; candidate < outputs.size(); ++candidate) {
    const auto proto = HeadGridShape::from_proto(outputs[candidate], layout);
    if (proto.batch != 1 || proto.channels <= 0 || proto.rows <= 0 || proto.cols <= 0)
      continue;
    RawSegmentOutputs found{{}, candidate, static_cast<std::size_t>(proto.channels)};
    Vector<core::Point2f> candidate_strides;
    bool fits = true;
    auto candidate_channels = box_channels;
    for (std::size_t i = 0; i < outputs.size() && fits; ++i) {
      if (i == candidate)
        continue;
      const auto head = HeadGridShape::from_proto(outputs[i], layout);
      const auto channels = head.channels - static_cast<std::int64_t>(classes) - static_cast<std::int64_t>(found.masks);
      fits = channels > 0 && channels % 4 == 0;
      if (fits && candidate_channels == 0)
        candidate_channels = static_cast<std::size_t>(channels);
      fits = fits && static_cast<std::size_t>(channels) == candidate_channels;
      if (fits) {
        found.heads.push_back(i);
        candidate_strides.push_back(head.stride(input));
      }
    }
    if (!fits)
      continue;
    KZ_CHECK_RETURN(!selected, std::nullopt, "{}: ambiguous raw heads and prototypes", name);
    selected = std::move(found);
    inferred = std::move(candidate_strides);
    selected_channels = candidate_channels;
  }
  KZ_CHECK_RETURN(selected, std::nullopt, "{}: outputs must be raw heads matching the input size plus mask prototypes",
                  name);
  strides = std::move(inferred);
  box_channels = selected_channels;
  return selected;
}

} // namespace hrt::postprocess
