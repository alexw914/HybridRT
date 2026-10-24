// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/cuda_image_preprocessor.cpp
// Description: Implements the CUDA image preprocessor, a fused kernel.
// -----------------------------------------------------------------------------

#include "include/preprocess/cuda_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>

#include "include/backend/cuda/cuda_common.hpp"
#include "include/backend/cuda/kernels/image_kernels.hpp"

namespace hrt::preprocess {

namespace {

auto is_rgb(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterRgb || format == core::ImageFormat::PlanarRgb;
}

auto is_interleaved(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterBgr || format == core::ImageFormat::InterRgb;
}

auto is_planar(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::PlanarBgr || format == core::ImageFormat::PlanarRgb;
}

auto input_format(core::ImageFormat format) -> std::optional<backend::InputFormat> {
  switch (format) {
  case core::ImageFormat::InterBgr:
    return backend::InputFormat::Bgr;
  case core::ImageFormat::InterRgb:
    return backend::InputFormat::Rgb;
  case core::ImageFormat::Nv12:
    return backend::InputFormat::Nv12;
  case core::ImageFormat::Nv21:
    return backend::InputFormat::Nv21;
  case core::ImageFormat::Yu12:
    return backend::InputFormat::I420;
  case core::ImageFormat::Yv12:
    return backend::InputFormat::Yv12;
  default:
    return std::nullopt;
  }
}

auto is_identity(const Array<float, 3>& mean, const Array<float, 3>& stddev) -> bool {
  return std::all_of(mean.begin(), mean.end(), [](float v) -> bool { return v == 0.0F; }) &&
         std::all_of(stddev.begin(), stddev.end(), [](float v) -> bool { return v == 1.0F; });
}

// Bytes between rows: stride[2] for planar images, stride[1] for interleaved and YUV ones.
auto pitch(const core::Image& image) -> std::int64_t {
  return static_cast<std::int64_t>(image.stride()[is_planar(image.format()) ? 2 : 1]);
}

auto at(const core::Image& image, std::initializer_list<std::int64_t> index, core::Stream* stream)
    -> const std::uint8_t* {
  const auto base = image.memory()->device(stream);
  return base == 0 ? nullptr : reinterpret_cast<const std::uint8_t*>(base + image.buffer()->offset(index));
}

// Kernel parameters for writing `dst` from `src` as `transform` places it; empty if an address is missing.
auto create_preprocess_param(const core::Image& dst, const ImageTransform& transform, const core::Image& src,
                             const ImagePreprocessorParam& param, core::Stream* stream)
    -> std::optional<backend::PreprocessParam> {
  const auto& roi = transform.dst_roi;
  const auto& from = transform.src_roi;
  const auto source = input_format(src.format());
  const auto* region = at(src, {from.y, from.x}, stream);
  const auto base = dst.memory()->mutable_device(stream);
  if (!source || region == nullptr || base == 0 || roi.empty() || from.empty())
    return std::nullopt;
  backend::PreprocessParam preprocess_param;
  preprocess_param.source = *source;
  preprocess_param.src = region;
  preprocess_param.src_pitch = pitch(src);
  if (*source != backend::InputFormat::Bgr && *source != backend::InputFormat::Rgb) {
    preprocess_param.chroma = at(src, {src.size().height}, stream);
    preprocess_param.origin_x = from.x;
    preprocess_param.origin_y = from.y;
    preprocess_param.image_width = src.size().width;
    preprocess_param.image_height = src.size().height;
  }
  preprocess_param.src_width = from.width;
  preprocess_param.src_height = from.height;
  preprocess_param.scale_x = 1.0 / (static_cast<double>(roi.width) / from.width);
  preprocess_param.scale_y = 1.0 / (static_cast<double>(roi.height) / from.height);
  preprocess_param.dst = reinterpret_cast<std::uint8_t*>(base + dst.buffer()->offset({0}));
  preprocess_param.dst_pitch = pitch(dst);
  preprocess_param.dst_plane = is_planar(dst.format()) ? static_cast<std::int64_t>(dst.stride()[1]) : 0;
  preprocess_param.dst_width = dst.size().width;
  preprocess_param.dst_height = dst.size().height;
  preprocess_param.roi_x = roi.x;
  preprocess_param.roi_y = roi.y;
  preprocess_param.roi_width = roi.width;
  preprocess_param.roi_height = roi.height;
  preprocess_param.float_output = dst.data_type() == core::DataType::Float32;
  preprocess_param.interpolation =
      param.interpolation_mode == InterpolationMode::Nearest ? backend::ResizeInterpolation::Nearest
      : param.interpolation_mode == InterpolationMode::Cubic ? backend::ResizeInterpolation::Cubic
                                                             : backend::ResizeInterpolation::Linear;
  // As the CPU flow: YUV converts to BGR, and padding is the default color, in input channel order, as a uint8
  // pixel, normalized unless only the ROI is.
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  const auto swap = is_rgb(src.format()) != is_rgb(dst.format());
  for (int i = 0; i < 3; ++i) {
    const auto c = static_cast<std::size_t>(i);
    const auto alpha = normalize ? 1.0 / static_cast<double>(param.std[c]) : 1.0;
    const auto beta = normalize ? -static_cast<double>(param.mean[c]) * alpha : 0.0;
    const auto pad = std::clamp(std::nearbyint(static_cast<double>(param.default_color[c])), 0.0, 255.0);
    preprocess_param.channel[c] = swap ? 2 - i : i;
    preprocess_param.alpha[c] = static_cast<float>(alpha);
    preprocess_param.beta[c] = static_cast<float>(beta);
    preprocess_param.pad[c] = static_cast<float>(param.normalize_type == NormalizeType::Roi ? pad : pad * alpha + beta);
  }
  return preprocess_param;
}

} // namespace

auto CudaImagePreprocessor::supports(ImageOperation /*op*/, const core::Image& /*dst*/,
                                     const core::Image& /*src*/) const -> bool {
  return false;
}

auto CudaImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                           const ImagePreprocessorParam& param) const -> bool {
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  return input_format(src.format()).has_value() && src.data_type() == core::DataType::Uint8 &&
         (is_interleaved(dst.format()) || is_planar(dst.format())) &&
         (dst.data_type() == core::DataType::Float32 || (dst.data_type() == core::DataType::Uint8 && !normalize));
}

auto CudaImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                  const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  const auto preprocess_param = create_preprocess_param(dst, transform, src, param, stream);
  if (!preprocess_param)
    return false;
  const auto status = backend::launch_cuda_preprocess(&*preprocess_param, 1, backend::cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: preprocess kernel failed: {}", cudaGetErrorString(status));
  return true;
}

auto CudaImagePreprocessor::fused_batch(std::span<core::Image* const> dsts, std::span<const ImageTransform> transforms,
                                        std::span<const core::Image* const> srcs, const ImagePreprocessorParam& param,
                                        core::Stream* stream) -> bool {
  params_.clear();
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    const auto preprocess_param = create_preprocess_param(*dsts[i], transforms[i], *srcs[i], param, stream);
    if (!preprocess_param)
      return false;
    params_.push_back(*preprocess_param);
  }
  const auto status = backend::launch_cuda_preprocess(params_.data(), params_.size(), backend::cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: preprocess kernel failed: {}", cudaGetErrorString(status));
  return true;
}

} // namespace hrt::preprocess
