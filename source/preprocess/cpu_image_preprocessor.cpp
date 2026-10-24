// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/cpu_image_preprocessor.cpp
// Description: Implements the OpenCV-based host image operators.
// -----------------------------------------------------------------------------

#include "include/preprocess/cpu_image_preprocessor.hpp"

#include <algorithm>
#include <cstdint>
#include <opencv2/imgproc.hpp>

namespace hrt::preprocess {

namespace {

enum class Layout {
  Unsupported,
  Inter3,
  Inter4,
  Planar3,
  Gray,
  Yuv,
};

auto layout_of(core::ImageFormat format) -> Layout {
  switch (format) {
  case core::ImageFormat::InterBgr:
  case core::ImageFormat::InterRgb:
    return Layout::Inter3;
  case core::ImageFormat::InterRgba:
    return Layout::Inter4;
  case core::ImageFormat::PlanarBgr:
  case core::ImageFormat::PlanarRgb:
    return Layout::Planar3;
  case core::ImageFormat::Gray:
    return Layout::Gray;
  case core::ImageFormat::Nv12:
  case core::ImageFormat::Nv21:
  case core::ImageFormat::Yu12:
  case core::ImageFormat::Yv12:
    return Layout::Yuv;
  case core::ImageFormat::Unknown:
    break;
  }
  return Layout::Unsupported;
}

auto rgb_order(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterRgb || format == core::ImageFormat::PlanarRgb ||
         format == core::ImageFormat::InterRgba;
}

auto depth_of(core::DataType type) -> int {
  switch (type) {
  case core::DataType::Uint8:
    return CV_8U;
  case core::DataType::Float32:
    return CV_32F;
  default:
    return -1;
  }
}

auto scalar(const Array<float, 3>& v) -> cv::Scalar {
  return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

auto inverse(const Array<float, 3>& v) -> cv::Scalar {
  return {1.0 / static_cast<double>(v[0]), 1.0 / static_cast<double>(v[1]), 1.0 / static_cast<double>(v[2])};
}

auto cv_rect(const core::Rect& rect) -> cv::Rect {
  return {rect.x, rect.y, rect.width, rect.height};
}

auto interpolation(InterpolationMode mode) -> int {
  switch (mode) {
  case InterpolationMode::Nearest:
    return cv::INTER_NEAREST;
  case InterpolationMode::Cubic:
    return cv::INTER_CUBIC;
  case InterpolationMode::Linear:
    break;
  }
  return cv::INTER_LINEAR;
}

// cv::Mat has no const view; sources are only read through these pointers.
auto host(const core::Image& image, core::Stream* stream) -> std::byte* {
  return const_cast<std::byte*>(static_cast<const std::byte*>(image.memory()->cpu(stream)));
}

auto mutable_host(core::Image& image, core::Stream* stream) -> std::byte* {
  return static_cast<std::byte*>(image.memory()->mutable_cpu(stream));
}

// Whole interleaved or YUV image; byte stride[1] is the row pitch.
auto inter(const core::Image& image, std::byte* base) -> cv::Mat {
  const auto layout = layout_of(image.format());
  const auto rows = layout == Layout::Yuv ? image.size().height * 3 / 2 : image.size().height;
  const auto channels = layout == Layout::Inter3 ? 3 : (layout == Layout::Inter4 ? 4 : 1);
  return {rows, image.size().width, CV_MAKETYPE(depth_of(image.data_type()), channels), base, image.stride()[1]};
}

// One plane of a planar or gray image; stride[1] steps planes, stride[2] rows.
auto plane(const core::Image& image, std::byte* base, int index) -> cv::Mat {
  const auto& stride = image.stride();
  return {image.size().height, image.size().width, CV_MAKETYPE(depth_of(image.data_type()), 1),
          base + static_cast<std::size_t>(index) * stride[1], stride[2]};
}

auto yuv_to_bgr_code(core::ImageFormat format) -> int {
  switch (format) {
  case core::ImageFormat::Nv12:
    return cv::COLOR_YUV2BGR_NV12;
  case core::ImageFormat::Nv21:
    return cv::COLOR_YUV2BGR_NV21;
  case core::ImageFormat::Yu12:
    return cv::COLOR_YUV2BGR_I420;
  default:
    return cv::COLOR_YUV2BGR_YV12;
  }
}

auto is_identity(const Array<float, 3>& mean, const Array<float, 3>& stddev) -> bool {
  return std::all_of(mean.begin(), mean.end(), [](float v) -> bool { return v == 0.0F; }) &&
         std::all_of(stddev.begin(), stddev.end(), [](float v) -> bool { return v == 1.0F; });
}

// Top-left `size` part of `buffer`, which grows to hold it and is otherwise reused.
auto area(cv::Mat& buffer, int type, core::Size size) -> cv::Mat {
  if (buffer.type() != type || buffer.cols < size.width || buffer.rows < size.height)
    buffer.create(std::max(buffer.rows, size.height), std::max(buffer.cols, size.width), type);
  return buffer(cv::Rect(0, 0, size.width, size.height));
}

// The parts of `rect` outside `roi`, which lies inside it.
auto borders(const cv::Rect& rect, const cv::Rect& roi) -> Array<cv::Rect, 4> {
  return {cv::Rect(rect.x, rect.y, rect.width, roi.y - rect.y),
          cv::Rect(rect.x, roi.y + roi.height, rect.width, rect.y + rect.height - roi.y - roi.height),
          cv::Rect(rect.x, roi.y, roi.x - rect.x, roi.height),
          cv::Rect(roi.x + roi.width, roi.y, rect.x + rect.width - roi.x - roi.width, roi.height)};
}

} // namespace

auto CpuImagePreprocessor::accepts(core::MemoryType type) const -> bool {
  return type == core::MemoryType::Cpu || type == core::MemoryType::DmaBuffer;
}

auto CpuImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  const auto dst_layout = layout_of(dst.format());
  const auto src_layout = layout_of(src.format());
  if (depth_of(dst.data_type()) < 0)
    return false;
  switch (op) {
  case ImageOperation::Fill:
    return dst_layout != Layout::Unsupported && dst_layout != Layout::Yuv;
  case ImageOperation::Resize:
    return dst.data_type() == src.data_type() && dst_layout == src_layout && dst_layout != Layout::Unsupported &&
           dst_layout != Layout::Yuv;
  case ImageOperation::Normalize:
    return dst.data_type() == core::DataType::Float32 &&
           (dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray);
  case ImageOperation::Convert: {
    if (dst.size() != src.size())
      return false;
    if (dst.format() == src.format() && dst.data_type() == src.data_type())
      return true;
    if (src.data_type() != core::DataType::Uint8)
      return false;
    switch (src_layout) {
    case Layout::Inter3:
    case Layout::Inter4:
    case Layout::Yuv:
      return dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray;
    case Layout::Planar3:
      return dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3;
    case Layout::Gray:
      return dst_layout == Layout::Gray;
    case Layout::Unsupported:
      break;
    }
    return false;
  }
  case ImageOperation::Fused:
    break;
  }
  return false;
}

auto CpuImagePreprocessor::fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color,
                                core::Stream* stream) -> bool {
  auto* base = mutable_host(image, stream);
  if (base == nullptr)
    return false;
  const auto rect = cv_rect(roi);
  switch (layout_of(image.format())) {
  case Layout::Planar3:
    for (int i = 0; i < 3; ++i)
      plane(image, base, i)(rect).setTo(scalar(color)[i]);
    return true;
  case Layout::Gray:
    plane(image, base, 0)(rect).setTo(scalar(color)[0]);
    return true;
  case Layout::Inter3:
    inter(image, base)(rect).setTo(scalar(color));
    return true;
  case Layout::Inter4: {
    auto rgba = scalar(color);
    rgba[3] = 255.0;
    inter(image, base)(rect).setTo(rgba);
    return true;
  }
  default:
    return false;
  }
}

auto CpuImagePreprocessor::resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src,
                                  const core::Rect& src_roi, InterpolationMode mode, core::Stream* stream) -> bool {
  auto* dst_base = mutable_host(dst, stream);
  auto* src_base = host(src, stream);
  if (dst_base == nullptr || src_base == nullptr || dst_roi.empty() || src_roi.empty())
    return false;
  const auto to = cv_rect(dst_roi);
  const auto from = cv_rect(src_roi);
  const auto method = interpolation(mode);
  const auto swap = rgb_order(dst.format()) != rgb_order(src.format());
  switch (layout_of(dst.format())) {
  case Layout::Planar3:
    for (int i = 0; i < 3; ++i) {
      cv::Mat out = plane(dst, dst_base, swap ? 2 - i : i)(to);
      cv::resize(plane(src, src_base, i)(from), out, out.size(), 0, 0, method);
    }
    return true;
  case Layout::Gray: {
    cv::Mat out = plane(dst, dst_base, 0)(to);
    cv::resize(plane(src, src_base, 0)(from), out, out.size(), 0, 0, method);
    return true;
  }
  case Layout::Inter3:
  case Layout::Inter4: {
    cv::Mat out = inter(dst, dst_base)(to);
    cv::resize(inter(src, src_base)(from), out, out.size(), 0, 0, method);
    if (swap)
      cv::cvtColor(out, out, out.channels() == 4 ? cv::COLOR_BGRA2RGBA : cv::COLOR_BGR2RGB);
    return true;
  }
  default:
    return false;
  }
}

auto CpuImagePreprocessor::convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  if (dst.format() == src.format() && dst.data_type() == src.data_type())
    return core::Buffer::copy(*dst.buffer(), *src.buffer(), core::CopyPolicy::DirectCopy, stream);
  auto* dst_base = mutable_host(dst, stream);
  auto* src_base = host(src, stream);
  if (dst_base == nullptr || src_base == nullptr)
    return false;
  const auto depth = depth_of(dst.data_type());
  const auto dst_layout = layout_of(dst.format());
  const auto src_layout = layout_of(src.format());

  if (src_layout == Layout::Gray) {
    plane(src, src_base, 0).convertTo(plane(dst, dst_base, 0), depth);
    return true;
  }

  if (src_layout == Layout::Planar3) {
    const auto swap = rgb_order(dst.format()) != rgb_order(src.format());
    if (dst_layout == Layout::Planar3) {
      for (int i = 0; i < 3; ++i)
        plane(src, src_base, i).convertTo(plane(dst, dst_base, swap ? 2 - i : i), depth);
      return true;
    }
    const Array<cv::Mat, 3> ordered{plane(src, src_base, swap ? 2 : 0), plane(src, src_base, 1),
                                    plane(src, src_base, swap ? 0 : 2)};
    cv::merge(ordered.data(), ordered.size(), bgr_);
    bgr_.convertTo(inter(dst, dst_base), depth);
    return true;
  }

  // Interleaved sources: bring YUV and RGBA to 3-channel BGR first.
  cv::Mat color = inter(src, src_base);
  auto rgb = rgb_order(src.format());
  if (src_layout == Layout::Yuv) {
    cv::cvtColor(color, bgr_, yuv_to_bgr_code(src.format()));
    color = bgr_;
    rgb = false;
  } else if (src_layout == Layout::Inter4) {
    cv::cvtColor(color, bgr_, cv::COLOR_RGBA2BGR);
    color = bgr_;
    rgb = false;
  }

  if (dst_layout == Layout::Gray) {
    cv::cvtColor(color, planes_[0], rgb ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
    planes_[0].convertTo(plane(dst, dst_base, 0), depth);
    return true;
  }
  const auto swap = rgb != rgb_order(dst.format());
  if (dst_layout == Layout::Inter3) {
    if (swap) {
      cv::cvtColor(color, bgr_, cv::COLOR_BGR2RGB);
      color = bgr_;
    }
    color.convertTo(inter(dst, dst_base), depth);
    return true;
  }
  cv::split(color, planes_.data());
  for (int i = 0; i < 3; ++i)
    planes_[static_cast<std::size_t>(swap ? 2 - i : i)].convertTo(plane(dst, dst_base, i), depth);
  return true;
}

auto CpuImagePreprocessor::convert_normalize(core::Image& dst, const core::Image& src, const core::Rect& roi,
                                             const Array<float, 3>& mean, const Array<float, 3>& stddev,
                                             core::Stream* stream) -> bool {
  if (layout_of(src.format()) != Layout::Inter3 || src.data_type() != core::DataType::Uint8 ||
      layout_of(dst.format()) != Layout::Planar3 || dst.data_type() != core::DataType::Float32)
    return convert(dst, src, stream) && normalize(dst, roi, mean, stddev, stream);

  auto* dst_base = mutable_host(dst, stream);
  auto* src_base = host(src, stream);
  if (dst_base == nullptr || src_base == nullptr)
    return false;
  cv::split(inter(src, src_base), planes_.data());
  const auto swap = rgb_order(src.format()) != rgb_order(dst.format());
  const auto rect = cv_rect(roi);
  for (int i = 0; i < 3; ++i) {
    const auto c = static_cast<std::size_t>(i);
    const auto& input = planes_[static_cast<std::size_t>(swap ? 2 - i : i)];
    auto output = plane(dst, dst_base, i);
    // Padding outside a normalized ROI must retain its unnormalized value.
    if (roi != dst.rect())
      input.convertTo(output, CV_32F);
    const auto alpha = 1.0 / static_cast<double>(stddev[c]);
    input(rect).convertTo(output(rect), CV_32F, alpha, -static_cast<double>(mean[c]) * alpha);
  }
  return true;
}

auto CpuImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                          const ImagePreprocessorParam& /*param*/) const -> bool {
  return layout_of(src.format()) == Layout::Inter3 && src.data_type() == core::DataType::Uint8 &&
         layout_of(dst.format()) == Layout::Planar3 && dst.data_type() == core::DataType::Float32;
}

auto CpuImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                 const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  auto* dst_base = mutable_host(dst, stream);
  auto* src_base = host(src, stream);
  const auto& roi = transform.dst_roi;
  if (dst_base == nullptr || src_base == nullptr || roi.empty() || transform.src_roi.empty())
    return false;
  // Same results as filling, resizing into the region and converting the whole input, without touching the padding
  // twice: cv::resize depends only on the source region and the region size.
  auto pixels = area(region_, CV_8UC3, roi.size());
  cv::resize(inter(src, src_base)(cv_rect(transform.src_roi)), pixels, pixels.size(), 0, 0,
             interpolation(param.interpolation_mode));
  Array<cv::Mat, 3> planes;
  for (std::size_t i = 0; i < planes.size(); ++i)
    planes[i] = area(region_planes_[i], CV_8UC1, roi.size());
  cv::split(pixels, planes.data());

  const bool normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  const auto swap = rgb_order(src.format()) != rgb_order(dst.format());
  const auto whole = cv_rect(dst.rect());
  const auto region = cv_rect(roi);
  for (int i = 0; i < 3; ++i) {
    const auto c = static_cast<std::size_t>(i);
    const auto input = static_cast<std::size_t>(swap ? 2 - i : i);
    const auto alpha = normalize ? 1.0 / static_cast<double>(param.std[c]) : 1.0;
    const auto beta = normalize ? -static_cast<double>(param.mean[c]) * alpha : 0.0;
    const auto output = plane(dst, dst_base, i);
    planes[input].convertTo(output(region), CV_32F, alpha, beta);
    if (region == whole)
      continue;
    // Padding is the default color, in input channel order, as a uint8 pixel; a normalized ROI leaves it as is.
    const auto pad = static_cast<double>(cv::saturate_cast<std::uint8_t>(param.default_color[c]));
    const auto value = param.normalize_type == NormalizeType::Roi ? pad : pad * alpha + beta;
    for (const auto& border : borders(whole, region)) {
      if (!border.empty())
        output(border).setTo(value);
    }
  }
  return true;
}

auto CpuImagePreprocessor::normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean,
                                     const Array<float, 3>& stddev, core::Stream* stream) -> bool {
  auto* base = mutable_host(image, stream);
  if (base == nullptr)
    return false;
  const auto rect = cv_rect(roi);
  const auto layout = layout_of(image.format());
  if (layout == Layout::Inter3) {
    cv::Mat region = inter(image, base)(rect);
    cv::subtract(region, scalar(mean), region);
    cv::multiply(region, inverse(stddev), region);
    return true;
  }
  const int planes = layout == Layout::Planar3 ? 3 : 1;
  for (int i = 0; i < planes; ++i) {
    const auto c = static_cast<std::size_t>(i);
    if (mean[c] == 0.0F && stddev[c] == 1.0F)
      continue;
    cv::Mat region = plane(image, base, i)(rect);
    cv::subtract(region, cv::Scalar(scalar(mean)[i]), region);
    cv::multiply(region, cv::Scalar(inverse(stddev)[i]), region);
  }
  return true;
}

} // namespace hrt::preprocess
