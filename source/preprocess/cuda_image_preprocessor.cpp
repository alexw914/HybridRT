// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/cuda_image_preprocessor.cpp
// Description: Implements the CUDA image operators: a fused kernel and NPP steps.
// -----------------------------------------------------------------------------

#include "include/preprocess/cuda_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>

#include "include/backend/cuda/cuda_common.hpp"
#include "include/preprocess/cuda_image_kernels.hpp"

namespace hrt::preprocess {

namespace {

enum class Layout {
  Unsupported,
  Inter3,
  Planar3,
  Gray,
  Nv12,
};

auto layout_of(core::ImageFormat format) -> Layout {
  switch (format) {
  case core::ImageFormat::InterBgr:
  case core::ImageFormat::InterRgb:
    return Layout::Inter3;
  case core::ImageFormat::PlanarBgr:
  case core::ImageFormat::PlanarRgb:
    return Layout::Planar3;
  case core::ImageFormat::Gray:
    return Layout::Gray;
  case core::ImageFormat::Nv12:
    return Layout::Nv12;
  default:
    return Layout::Unsupported;
  }
}

auto rgb_order(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterRgb || format == core::ImageFormat::PlanarRgb;
}

auto is_u8_or_f32(core::DataType type) -> bool {
  return type == core::DataType::Uint8 || type == core::DataType::Float32;
}

auto npp_size(core::Size size) -> NppiSize {
  return {size.width, size.height};
}

auto npp_rect(const core::Rect& rect) -> NppiRect {
  return {rect.x, rect.y, rect.width, rect.height};
}

auto npp_interpolation(InterpolationMode mode) -> int {
  switch (mode) {
  case InterpolationMode::Nearest:
    return NPPI_INTER_NN;
  case InterpolationMode::Cubic:
    return NPPI_INTER_CUBIC;
  case InterpolationMode::Linear:
    break;
  }
  return NPPI_INTER_LINEAR;
}

// Bytes between rows: stride[2] for planar and gray, stride[1] otherwise.
auto pitch(const core::Image& image) -> int {
  const auto layout = layout_of(image.format());
  return static_cast<int>(image.stride()[layout == Layout::Planar3 || layout == Layout::Gray ? 2 : 1]);
}

template <typename T>
auto at(const core::Image& image, std::initializer_list<std::int64_t> index, core::Stream* stream) -> const T* {
  const auto base = image.memory()->device(stream);
  return base == 0 ? nullptr : reinterpret_cast<const T*>(base + image.buffer()->offset(index));
}

template <typename T>
auto at_mut(const core::Image& image, std::initializer_list<std::int64_t> index, core::Stream* stream) -> T* {
  const auto base = image.memory()->mutable_device(stream);
  return base == 0 ? nullptr : reinterpret_cast<T*>(base + image.buffer()->offset(index));
}

template <typename T>
auto planes(const core::Image& image, bool swap, core::Stream* stream) -> Array<const T*, 3> {
  Array<const T*, 3> result{at<T>(image, {0}, stream), at<T>(image, {1}, stream), at<T>(image, {2}, stream)};
  if (swap)
    std::swap(result[0], result[2]);
  return result;
}

template <typename T>
auto planes_mut(const core::Image& image, bool swap, core::Stream* stream) -> Array<T*, 3> {
  Array<T*, 3> result{at_mut<T>(image, {0}, stream), at_mut<T>(image, {1}, stream), at_mut<T>(image, {2}, stream)};
  if (swap)
    std::swap(result[0], result[2]);
  return result;
}

constexpr int kSwapOrder[3] = {2, 1, 0};

auto is_identity(const Array<float, 3>& mean, const Array<float, 3>& stddev) -> bool {
  return std::all_of(mean.begin(), mean.end(), [](float v) -> bool { return v == 0.0F; }) &&
         std::all_of(stddev.begin(), stddev.end(), [](float v) -> bool { return v == 1.0F; });
}

} // namespace

auto CudaImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  const auto dst_layout = layout_of(dst.format());
  const auto src_layout = layout_of(src.format());
  if (!is_u8_or_f32(dst.data_type()))
    return false;
  switch (op) {
  case ImageOperation::Fill:
    return dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray;
  case ImageOperation::Resize:
    return dst.data_type() == src.data_type() && dst_layout == src_layout &&
           (dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray);
  case ImageOperation::Normalize:
    return dst.data_type() == core::DataType::Float32 &&
           (dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray);
  case ImageOperation::Convert:
    if (dst.size() != src.size() || !is_u8_or_f32(src.data_type()))
      return false;
    switch (src_layout) {
    case Layout::Nv12:
      return src.data_type() == core::DataType::Uint8 &&
             (dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray);
    case Layout::Inter3:
      return dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3 || dst_layout == Layout::Gray;
    case Layout::Planar3:
      return dst_layout == Layout::Inter3 || dst_layout == Layout::Planar3;
    case Layout::Gray:
      return dst_layout == Layout::Gray;
    case Layout::Unsupported:
      break;
    }
    return false;
  case ImageOperation::Fused:
    break;
  }
  return false;
}

auto CudaImagePreprocessor::fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color,
                                 core::Stream* stream) -> bool {
  const auto ctx = backend::npp_context(stream);
  const auto size = npp_size(roi.size());
  const auto step = pitch(image);
  const auto u8 = image.data_type() == core::DataType::Uint8;
  switch (layout_of(image.format())) {
  case Layout::Planar3:
  case Layout::Gray: {
    const int count = layout_of(image.format()) == Layout::Gray ? 1 : 3;
    for (int i = 0; i < count; ++i) {
      const auto value = color[static_cast<std::size_t>(i)];
      const auto status =
          u8 ? nppiSet_8u_C1R_Ctx(static_cast<Npp8u>(value), at_mut<Npp8u>(image, {i, roi.y, roi.x}, stream), step,
                                  size, ctx)
             : nppiSet_32f_C1R_Ctx(value, at_mut<Npp32f>(image, {i, roi.y, roi.x}, stream), step, size, ctx);
      KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSet failed: {}", static_cast<int>(status));
    }
    return true;
  }
  case Layout::Inter3: {
    if (u8) {
      const Npp8u value[3] = {static_cast<Npp8u>(color[0]), static_cast<Npp8u>(color[1]), static_cast<Npp8u>(color[2])};
      const auto status = nppiSet_8u_C3R_Ctx(value, at_mut<Npp8u>(image, {roi.y, roi.x}, stream), step, size, ctx);
      KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSet_8u_C3R failed: {}", static_cast<int>(status));
      return true;
    }
    const auto status =
        nppiSet_32f_C3R_Ctx(color.data(), at_mut<Npp32f>(image, {roi.y, roi.x}, stream), step, size, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSet_32f_C3R failed: {}", static_cast<int>(status));
    return true;
  }
  default:
    return false;
  }
}

auto CudaImagePreprocessor::resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src,
                                   const core::Rect& src_roi, InterpolationMode mode, core::Stream* stream) -> bool {
  const auto ctx = backend::npp_context(stream);
  const auto interpolation = npp_interpolation(mode);
  const auto u8 = dst.data_type() == core::DataType::Uint8;
  const auto swap = rgb_order(dst.format()) != rgb_order(src.format());
  const auto src_size = npp_size(src.size());
  const auto dst_size = npp_size(dst.size());
  const auto from = npp_rect(src_roi);
  const auto to = npp_rect(dst_roi);
  NppStatus status = NPP_NO_ERROR;
  switch (layout_of(dst.format())) {
  case Layout::Planar3:
    if (u8) {
      auto in = planes<Npp8u>(src, swap, stream);
      auto out = planes_mut<Npp8u>(dst, false, stream);
      status = nppiResize_8u_P3R_Ctx(in.data(), pitch(src), src_size, from, out.data(), pitch(dst), dst_size, to,
                                     interpolation, ctx);
    } else {
      auto in = planes<Npp32f>(src, swap, stream);
      auto out = planes_mut<Npp32f>(dst, false, stream);
      status = nppiResize_32f_P3R_Ctx(in.data(), pitch(src), src_size, from, out.data(), pitch(dst), dst_size, to,
                                      interpolation, ctx);
    }
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiResize_P3R failed: {}", static_cast<int>(status));
    return true;
  case Layout::Gray:
    status =
        u8 ? nppiResize_8u_C1R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), src_size, from,
                                   at_mut<Npp8u>(dst, {0}, stream), pitch(dst), dst_size, to, interpolation, ctx)
           : nppiResize_32f_C1R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src), src_size, from,
                                    at_mut<Npp32f>(dst, {0}, stream), pitch(dst), dst_size, to, interpolation, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiResize_C1R failed: {}", static_cast<int>(status));
    return true;
  case Layout::Inter3: {
    status =
        u8 ? nppiResize_8u_C3R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), src_size, from,
                                   at_mut<Npp8u>(dst, {0}, stream), pitch(dst), dst_size, to, interpolation, ctx)
           : nppiResize_32f_C3R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src), src_size, from,
                                    at_mut<Npp32f>(dst, {0}, stream), pitch(dst), dst_size, to, interpolation, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiResize_C3R failed: {}", static_cast<int>(status));
    if (!swap)
      return true;
    // Swap only the resized region so the padding keeps its color order.
    const auto roi_size = npp_size(dst_roi.size());
    status = u8 ? nppiSwapChannels_8u_C3IR_Ctx(at_mut<Npp8u>(dst, {dst_roi.y, dst_roi.x}, stream), pitch(dst), roi_size,
                                               kSwapOrder, ctx)
                : nppiSwapChannels_32f_C3IR_Ctx(at_mut<Npp32f>(dst, {dst_roi.y, dst_roi.x}, stream), pitch(dst),
                                                roi_size, kSwapOrder, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSwapChannels_C3IR failed: {}", static_cast<int>(status));
    return true;
  }
  default:
    return false;
  }
}

auto CudaImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                           const ImagePreprocessorParam& param) const -> bool {
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  return layout_of(src.format()) == Layout::Inter3 && src.data_type() == core::DataType::Uint8 &&
         (layout_of(dst.format()) == Layout::Inter3 || layout_of(dst.format()) == Layout::Planar3) &&
         (dst.data_type() == core::DataType::Float32 || (dst.data_type() == core::DataType::Uint8 && !normalize)) &&
         param.interpolation_mode != InterpolationMode::Cubic;
}

auto CudaImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                  const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  const auto& roi = transform.dst_roi;
  const auto& from = transform.src_roi;
  const auto* source = at<std::uint8_t>(src, {from.y, from.x}, stream);
  auto* target = at_mut<std::uint8_t>(dst, {0}, stream);
  if (source == nullptr || target == nullptr || roi.empty() || from.empty())
    return false;
  const auto planar = layout_of(dst.format()) == Layout::Planar3;
  CudaResizeTask task;
  task.src = source;
  task.src_pitch = pitch(src);
  task.src_width = from.width;
  task.src_height = from.height;
  task.scale_x = 1.0 / (static_cast<double>(roi.width) / from.width);
  task.scale_y = 1.0 / (static_cast<double>(roi.height) / from.height);
  task.dst = target;
  task.dst_pitch = pitch(dst);
  task.dst_plane = planar ? static_cast<std::int64_t>(dst.stride()[1]) : 0;
  task.dst_width = dst.size().width;
  task.dst_height = dst.size().height;
  task.roi_x = roi.x;
  task.roi_y = roi.y;
  task.roi_width = roi.width;
  task.roi_height = roi.height;
  task.float_output = dst.data_type() == core::DataType::Float32;
  task.nearest = param.interpolation_mode == InterpolationMode::Nearest;
  // As the CPU flow: padding is the default color, in input channel order, as a uint8 pixel, normalized unless only
  // the ROI is.
  const auto normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  const auto swap = rgb_order(src.format()) != rgb_order(dst.format());
  for (int i = 0; i < 3; ++i) {
    const auto c = static_cast<std::size_t>(i);
    const auto input = swap ? 2 - i : i;
    const auto alpha = normalize ? 1.0 / static_cast<double>(param.std[c]) : 1.0;
    const auto beta = normalize ? -static_cast<double>(param.mean[c]) * alpha : 0.0;
    const auto pad = std::clamp(std::nearbyint(static_cast<double>(param.default_color[c])), 0.0, 255.0);
    task.channel[c] = input;
    task.alpha[c] = static_cast<float>(alpha);
    task.beta[c] = static_cast<float>(beta);
    task.pad[c] = static_cast<float>(param.normalize_type == NormalizeType::Roi ? pad : pad * alpha + beta);
  }
  const auto status = launch_cuda_resize(&task, 1, backend::cuda_stream(stream));
  KZ_CHECK_RETURN(status == cudaSuccess, false, "cuda: resize kernel failed: {}", cudaGetErrorString(status));
  return true;
}

auto CudaImagePreprocessor::scratch(UniquePtr<core::Image>& slot, core::ImageFormat format, core::DataType data_type,
                                    core::Size size) -> core::Image* {
  if (slot == nullptr)
    slot = std::make_unique<core::Image>(core::MemoryType::Cuda, format, data_type, size);
  else if (!slot->reshape(format, size, data_type))
    return nullptr;
  return slot->valid() ? slot.get() : nullptr;
}

auto CudaImagePreprocessor::convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  const core::Image* current = &src;
  if (layout_of(src.format()) == Layout::Nv12) {
    const auto rgb = rgb_order(dst.format());
    auto* color = scratch(color_, rgb ? core::ImageFormat::InterRgb : core::ImageFormat::InterBgr,
                          core::DataType::Uint8, src.size());
    if (color == nullptr)
      return false;
    const auto& stride = src.stride();
    const Npp8u* yuv[2] = {at<Npp8u>(src, {0}, stream), at<Npp8u>(src, {src.size().height}, stream)};
    const auto ctx = backend::npp_context(stream);
    const auto status =
        rgb ? nppiNV12ToRGB_8u_P2C3R_Ctx(yuv, static_cast<int>(stride[1]), at_mut<Npp8u>(*color, {0}, stream),
                                         pitch(*color), npp_size(src.size()), ctx)
            : nppiNV12ToBGR_8u_P2C3R_Ctx(yuv, static_cast<int>(stride[1]), at_mut<Npp8u>(*color, {0}, stream),
                                         pitch(*color), npp_size(src.size()), ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiNV12ToRGB failed: {}", static_cast<int>(status));
    current = color;
  }
  if (current->data_type() != dst.data_type()) {
    if (current->format() == dst.format())
      return convert_type(dst, *current, stream);
    auto* typed = scratch(typed_, current->format(), dst.data_type(), current->size());
    if (typed == nullptr || !convert_type(*typed, *current, stream))
      return false;
    current = typed;
  }
  if (current->format() == dst.format())
    return core::Buffer::copy(*dst.buffer(), *current->buffer(), core::CopyPolicy::DirectCopy, stream);
  return convert_layout(dst, *current, stream);
}

auto CudaImagePreprocessor::convert_type(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  const auto ctx = backend::npp_context(stream);
  const auto layout = layout_of(src.format());
  const auto to_float = dst.data_type() == core::DataType::Float32;
  if (layout == Layout::Inter3) {
    const auto size = npp_size(src.size());
    const auto status =
        to_float ? nppiConvert_8u32f_C3R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), at_mut<Npp32f>(dst, {0}, stream),
                                             pitch(dst), size, ctx)
                 : nppiConvert_32f8u_C3R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src), at_mut<Npp8u>(dst, {0}, stream),
                                             pitch(dst), size, NPP_RND_NEAR, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiConvert_C3R failed: {}", static_cast<int>(status));
    return true;
  }
  const int count = layout == Layout::Gray ? 1 : 3;
  for (int i = 0; i < count; ++i) {
    const auto size = npp_size(src.size());
    const auto status =
        to_float ? nppiConvert_8u32f_C1R_Ctx(at<Npp8u>(src, {i}, stream), pitch(src), at_mut<Npp32f>(dst, {i}, stream),
                                             pitch(dst), size, ctx)
                 : nppiConvert_32f8u_C1R_Ctx(at<Npp32f>(src, {i}, stream), pitch(src), at_mut<Npp8u>(dst, {i}, stream),
                                             pitch(dst), size, NPP_RND_NEAR, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiConvert_C1R failed: {}", static_cast<int>(status));
  }
  return true;
}

auto CudaImagePreprocessor::convert_layout(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  const auto ctx = backend::npp_context(stream);
  const auto size = npp_size(src.size());
  const auto u8 = src.data_type() == core::DataType::Uint8;
  const auto swap = rgb_order(dst.format()) != rgb_order(src.format());
  const auto from = layout_of(src.format());
  const auto to = layout_of(dst.format());
  NppStatus status = NPP_NO_ERROR;
  if (from == Layout::Planar3 && to == Layout::Planar3) {
    for (int i = 0; i < 3; ++i) {
      const auto source = swap ? 2 - i : i;
      status = u8 ? nppiCopy_8u_C1R_Ctx(at<Npp8u>(src, {source}, stream), pitch(src), at_mut<Npp8u>(dst, {i}, stream),
                                        pitch(dst), size, ctx)
                  : nppiCopy_32f_C1R_Ctx(at<Npp32f>(src, {source}, stream), pitch(src),
                                         at_mut<Npp32f>(dst, {i}, stream), pitch(dst), size, ctx);
      KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiCopy_C1R failed: {}", static_cast<int>(status));
    }
    return true;
  }
  if (from == Layout::Inter3 && to == Layout::Planar3) {
    if (u8) {
      auto out = planes_mut<Npp8u>(dst, swap, stream);
      status = nppiCopy_8u_C3P3R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), out.data(), pitch(dst), size, ctx);
    } else {
      auto out = planes_mut<Npp32f>(dst, swap, stream);
      status = nppiCopy_32f_C3P3R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src), out.data(), pitch(dst), size, ctx);
    }
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiCopy_C3P3R failed: {}", static_cast<int>(status));
    return true;
  }
  if (from == Layout::Planar3 && to == Layout::Inter3) {
    if (u8) {
      auto in = planes<Npp8u>(src, swap, stream);
      status = nppiCopy_8u_P3C3R_Ctx(in.data(), pitch(src), at_mut<Npp8u>(dst, {0}, stream), pitch(dst), size, ctx);
    } else {
      auto in = planes<Npp32f>(src, swap, stream);
      status = nppiCopy_32f_P3C3R_Ctx(in.data(), pitch(src), at_mut<Npp32f>(dst, {0}, stream), pitch(dst), size, ctx);
    }
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiCopy_P3C3R failed: {}", static_cast<int>(status));
    return true;
  }
  if (from == Layout::Inter3 && to == Layout::Inter3) {
    status = u8 ? nppiSwapChannels_8u_C3R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), at_mut<Npp8u>(dst, {0}, stream),
                                              pitch(dst), size, kSwapOrder, ctx)
                : nppiSwapChannels_32f_C3R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src),
                                               at_mut<Npp32f>(dst, {0}, stream), pitch(dst), size, kSwapOrder, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSwapChannels_C3R failed: {}", static_cast<int>(status));
    return true;
  }
  if (from == Layout::Inter3 && to == Layout::Gray) {
    // Luma weights in source channel order.
    const Npp32f bgr[3] = {0.114F, 0.587F, 0.299F};
    const Npp32f rgb[3] = {0.299F, 0.587F, 0.114F};
    const auto* weights = rgb_order(src.format()) ? rgb : bgr;
    status = u8 ? nppiColorToGray_8u_C3C1R_Ctx(at<Npp8u>(src, {0}, stream), pitch(src), at_mut<Npp8u>(dst, {0}, stream),
                                               pitch(dst), size, weights, ctx)
                : nppiColorToGray_32f_C3C1R_Ctx(at<Npp32f>(src, {0}, stream), pitch(src),
                                                at_mut<Npp32f>(dst, {0}, stream), pitch(dst), size, weights, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiColorToGray failed: {}", static_cast<int>(status));
    return true;
  }
  KZ_LOG_ERROR("cuda: unsupported layout conversion");
  return false;
}

auto CudaImagePreprocessor::normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean,
                                      const Array<float, 3>& stddev, core::Stream* stream) -> bool {
  const auto ctx = backend::npp_context(stream);
  const auto size = npp_size(roi.size());
  const auto step = pitch(image);
  if (layout_of(image.format()) == Layout::Inter3) {
    auto* data = at_mut<Npp32f>(image, {roi.y, roi.x}, stream);
    auto status = nppiSubC_32f_C3IR_Ctx(mean.data(), data, step, size, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSubC_32f_C3IR failed: {}", static_cast<int>(status));
    status = nppiDivC_32f_C3IR_Ctx(stddev.data(), data, step, size, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiDivC_32f_C3IR failed: {}", static_cast<int>(status));
    return true;
  }
  const int count = layout_of(image.format()) == Layout::Gray ? 1 : 3;
  for (int i = 0; i < count; ++i) {
    const auto c = static_cast<std::size_t>(i);
    if (mean[c] == 0.0F && stddev[c] == 1.0F)
      continue;
    auto* data = at_mut<Npp32f>(image, {i, roi.y, roi.x}, stream);
    auto status = nppiSubC_32f_C1IR_Ctx(mean[c], data, step, size, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiSubC_32f_C1IR failed: {}", static_cast<int>(status));
    status = nppiDivC_32f_C1IR_Ctx(stddev[c], data, step, size, ctx);
    KZ_CHECK_RETURN(status >= NPP_NO_ERROR, false, "npp: nppiDivC_32f_C1IR failed: {}", static_cast<int>(status));
  }
  return true;
}

} // namespace hrt::preprocess
