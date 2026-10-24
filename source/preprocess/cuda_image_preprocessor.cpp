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

auto rgb_order(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterRgb || format == core::ImageFormat::PlanarRgb;
}

auto is_interleaved(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterBgr || format == core::ImageFormat::InterRgb;
}

auto is_planar(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::PlanarBgr || format == core::ImageFormat::PlanarRgb;
}

auto source_of(core::ImageFormat format) -> std::optional<backend::CudaResizeSource> {
  switch (format) {
  case core::ImageFormat::InterBgr:
    return backend::CudaResizeSource::Bgr;
  case core::ImageFormat::InterRgb:
    return backend::CudaResizeSource::Rgb;
  case core::ImageFormat::Nv12:
    return backend::CudaResizeSource::Nv12;
  case core::ImageFormat::Nv21:
    return backend::CudaResizeSource::Nv21;
  case core::ImageFormat::Yu12:
    return backend::CudaResizeSource::I420;
  case core::ImageFormat::Yv12:
    return backend::CudaResizeSource::Yv12;
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

// The kernel task writing `dst` from `src` as `transform` places it; nothing when an address is missing.
auto make_task(const core::Image& dst, const ImageTransform& transform, const core::Image& src,
               const ImagePreprocessorParam& param, core::Stream* stream) -> std::optional<backend::CudaResizeTask> {
  const auto& roi = transform.dst_roi;
  const auto& from = transform.src_roi;
  const auto source = source_of(src.format());
  const auto* region = at(src, {from.y, from.x}, stream);
  const auto base = dst.memory()->mutable_device(stream);
  if (!source || region == nullptr || base == 0 || roi.empty() || from.empty())
    return std::nullopt;
  backend::CudaResizeTask task;
  task.source = *source;
  task.src = region;
  task.src_pitch = pitch(src);
  if (*source != backend::CudaResizeSource::Bgr && *source != backend::CudaResizeSource::Rgb) {
    task.chroma = at(src, {src.size().height}, stream);
    task.origin_x = from.x;
    task.origin_y = from.y;
    task.image_width = src.size().width;
    task.image_height = src.size().height;
  }
  task.src_width = from.width;
  task.src_height = from.height;
  task.scale_x = 1.0 / (static_cast<double>(roi.width) / from.width);
  task.scale_y = 1.0 / (static_cast<double>(roi.height) / from.height);
  task.dst = reinterpret_cast<std::uint8_t*>(base + dst.buffer()->offset({0}));
  task.dst_pitch = pitch(dst);
  task.dst_plane = is_planar(dst.format()) ? static_cast<std::int64_t>(dst.stride()[1]) : 0;
  task.dst_width = dst.size().width;
  task.dst_height = dst.size().height;
  task.roi_x = roi.x;
  task.roi_y = roi.y;
  task.roi_width = roi.width;
  task.roi_height = roi.height;
  task.float_output = dst.data_type() == core::DataType::Float32;
  task.filter = param.interpolation_mode == InterpolationMode::Nearest ? backend::CudaResizeFilter::Nearest
                : param.interpolation_mode == InterpolationMode::Cubic ? backend::CudaResizeFilter::Cubic
                                                                       : backend::CudaResizeFilter::Linear;
  // As the CPU flow: YUV converts to BGR, and padding is the default color, in input channel order, as a uint8
  // pixel, normalized unless only the ROI is.
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  const auto swap = rgb_order(src.format()) != rgb_order(dst.format());
  for (int i = 0; i < 3; ++i) {
    const auto c = static_cast<std::size_t>(i);
    const auto alpha = normalize ? 1.0 / static_cast<double>(param.std[c]) : 1.0;
    const auto beta = normalize ? -static_cast<double>(param.mean[c]) * alpha : 0.0;
    const auto pad = std::clamp(std::nearbyint(static_cast<double>(param.default_color[c])), 0.0, 255.0);
    task.channel[c] = swap ? 2 - i : i;
    task.alpha[c] = static_cast<float>(alpha);
    task.beta[c] = static_cast<float>(beta);
    task.pad[c] = static_cast<float>(param.normalize_type == NormalizeType::Roi ? pad : pad * alpha + beta);
  }
  return task;
}

} // namespace

auto CudaImagePreprocessor::supports(ImageOperation /*op*/, const core::Image& /*dst*/,
                                     const core::Image& /*src*/) const -> bool {
  return false;
}

auto CudaImagePreprocessor::fill(core::Image& /*image*/, const core::Rect& /*roi*/, const Array<float, 3>& /*color*/,
                                 core::Stream* /*stream*/) -> bool {
  return false;
}

auto CudaImagePreprocessor::resize(core::Image& /*dst*/, const core::Rect& /*dst_roi*/, const core::Image& /*src*/,
                                   const core::Rect& /*src_roi*/, InterpolationMode /*mode*/, core::Stream* /*stream*/)
    -> bool {
  return false;
}

auto CudaImagePreprocessor::convert(core::Image& /*dst*/, const core::Image& /*src*/, core::Stream* /*stream*/)
    -> bool {
  return false;
}

auto CudaImagePreprocessor::normalize(core::Image& /*image*/, const core::Rect& /*roi*/,
                                      const Array<float, 3>& /*mean*/, const Array<float, 3>& /*stddev*/,
                                      core::Stream* /*stream*/) -> bool {
  return false;
}

auto CudaImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                           const ImagePreprocessorParam& param) const -> bool {
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  return source_of(src.format()).has_value() && src.data_type() == core::DataType::Uint8 &&
         (is_interleaved(dst.format()) || is_planar(dst.format())) &&
         (dst.data_type() == core::DataType::Float32 || (dst.data_type() == core::DataType::Uint8 && !normalize));
}

auto CudaImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                  const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  const auto task = make_task(dst, transform, src, param, stream);
  if (!task)
    return false;
  const auto status = backend::launch_cuda_resize(&*task, 1, backend::cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: resize kernel failed: {}", cudaGetErrorString(status));
  return true;
}

auto CudaImagePreprocessor::fused_batch(std::span<core::Image* const> dsts, std::span<const ImageTransform> transforms,
                                        std::span<const core::Image* const> srcs, const ImagePreprocessorParam& param,
                                        core::Stream* stream) -> bool {
  tasks_.clear();
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    const auto task = make_task(*dsts[i], transforms[i], *srcs[i], param, stream);
    if (!task)
      return false;
    tasks_.push_back(*task);
  }
  const auto status = backend::launch_cuda_resize(tasks_.data(), tasks_.size(), backend::cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: resize kernel failed: {}", cudaGetErrorString(status));
  return true;
}

} // namespace hrt::preprocess
