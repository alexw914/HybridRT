// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/image_preprocessor.cpp
// Description: Implements resize geometry, preprocessor creation and the
//              shared preprocessing flow.
// -----------------------------------------------------------------------------

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>

#include "include/preprocess/cpu_image_preprocessor.hpp"

#ifdef HRT_USE_METAL
#include "include/preprocess/metal_image_preprocessor.hpp"
#endif

#ifdef HRT_USE_CUDA
#include "include/preprocess/cuda_image_preprocessor.hpp"
#endif
#ifdef HRT_USE_RKNN
#include "include/preprocess/rga_image_preprocessor.hpp"
#endif
#ifdef HRT_USE_CANN
#include "include/preprocess/dvpp_image_preprocessor.hpp"
#endif

namespace hrt::preprocess {

namespace {

auto device_of(core::MemoryType type) -> ImagePreprocessorType {
  switch (type) {
  case core::MemoryType::Cuda:
    return ImagePreprocessorType::Cuda;
  case core::MemoryType::DmaBuffer:
    return ImagePreprocessorType::Rga;
  case core::MemoryType::Acl:
  case core::MemoryType::Dvpp:
    return ImagePreprocessorType::Dvpp;
  default:
    return ImagePreprocessorType::Cpu;
  }
}

// Formats resize can produce from each other.
auto same_layout(core::ImageFormat a, core::ImageFormat b) -> bool {
  auto family = [](core::ImageFormat format) -> int {
    switch (format) {
    case core::ImageFormat::InterBgr:
    case core::ImageFormat::InterRgb:
      return 1;
    case core::ImageFormat::PlanarBgr:
    case core::ImageFormat::PlanarRgb:
      return 2;
    default:
      return 10 + static_cast<int>(format);
    }
  };
  return family(a) == family(b);
}

auto is_identity(const Array<float, 3>& mean, const Array<float, 3>& stddev) -> bool {
  return std::all_of(mean.begin(), mean.end(), [](float v) -> bool { return v == 0.0F; }) &&
         std::all_of(stddev.begin(), stddev.end(), [](float v) -> bool { return v == 1.0F; });
}

auto scaled(core::Size size, double scale) -> core::Size {
  return {static_cast<int>(size.width * scale), static_cast<int>(size.height * scale)};
}

auto pad_to_aspect(const core::Rect& rect, core::Size target, float scale) -> core::Rect {
  const auto aspect = static_cast<double>(target.width) / target.height;
  double width = rect.width;
  double height = rect.height;
  if (width > aspect * height)
    height = width / aspect;
  else
    width = height * aspect;
  width *= static_cast<double>(scale);
  height *= static_cast<double>(scale);
  const auto cx = rect.x + rect.width / 2.0;
  const auto cy = rect.y + rect.height / 2.0;
  return {static_cast<int>(std::lround(cx - width / 2)), static_cast<int>(std::lround(cy - height / 2)),
          static_cast<int>(std::lround(width)), static_cast<int>(std::lround(height))};
}

} // namespace

auto calc_resize_position(core::Rect& dst_roi, core::Rect& src_roi, const core::Rect& dst_rect,
                          const core::Rect& src_rect, core::Size src_size, const ImagePreprocessorParam& param)
    -> bool {
  if (dst_rect.empty() || src_rect.empty())
    return false;
  const auto ratio = std::min(static_cast<double>(dst_rect.width) / src_rect.width,
                              static_cast<double>(dst_rect.height) / src_rect.height);
  switch (param.resize_type) {
  case ResizeType::Normal:
    dst_roi = dst_rect;
    src_roi = src_rect;
    return true;
  case ResizeType::KeepRatio: {
    const auto size = scaled(src_rect.size(), ratio);
    dst_roi = core::intersect({dst_rect.x, dst_rect.y, size.width, size.height}, dst_rect);
    src_roi = src_rect;
    return true;
  }
  case ResizeType::TextLine: {
    const auto width = std::clamp(std::ceil(static_cast<double>(dst_rect.height) * src_rect.width / src_rect.height),
                                  1.0, static_cast<double>(dst_rect.width));
    dst_roi = {dst_rect.x, dst_rect.y, static_cast<int>(width), dst_rect.height};
    src_roi = src_rect;
    return true;
  }
  case ResizeType::LetterBox: {
    const auto size = scaled(src_rect.size(), ratio);
    dst_roi = core::intersect({dst_rect.x + (dst_rect.width - size.width) / 2,
                               dst_rect.y + (dst_rect.height - size.height) / 2, size.width, size.height},
                              dst_rect);
    src_roi = src_rect;
    return true;
  }
  case ResizeType::CenterCrop: {
    if (dst_rect.width != dst_rect.height || param.resize_short_size <= 0) {
      KZ_LOG_ERROR("center_crop needs a square input and resize_short_size > 0");
      return false;
    }
    dst_roi = dst_rect;
    const auto side = std::min(src_rect.width, src_rect.height) * dst_rect.width / param.resize_short_size;
    src_roi = {src_rect.x + std::max(0, (src_rect.width - side) / 2),
               src_rect.y + std::max(0, (src_rect.height - side) / 2), side, side};
    return true;
  }
  case ResizeType::PadOutside: {
    // Map the padded source rect onto the full input, then cut both sides
    // where the padded rect leaves the image.
    const auto padded = pad_to_aspect(src_rect, dst_rect.size(), param.pad_scale);
    const auto scale_x = static_cast<double>(dst_rect.width) / padded.width;
    const auto scale_y = static_cast<double>(dst_rect.height) / padded.height;
    src_roi = core::intersect(padded, {0, 0, src_size.width, src_size.height});
    if (src_roi.empty())
      return false;
    const auto x1 = dst_rect.x + static_cast<int>((src_roi.x - padded.x) * scale_x);
    const auto y1 = dst_rect.y + static_cast<int>((src_roi.y - padded.y) * scale_y);
    const auto x2 = dst_rect.x + static_cast<int>((src_roi.x + src_roi.width - padded.x) * scale_x);
    const auto y2 = dst_rect.y + static_cast<int>((src_roi.y + src_roi.height - padded.y) * scale_y);
    dst_roi = core::intersect({x1, y1, x2 - x1, y2 - y1}, dst_rect);
    return true;
  }
  }
  return false;
}

ImagePreprocessor::ImagePreprocessor() = default;

ImagePreprocessor::~ImagePreprocessor() = default;

auto ImagePreprocessor::create(ImagePreprocessorType type, core::MemoryType target) -> UniquePtr<ImagePreprocessor> {
  if (type == ImagePreprocessorType::Auto) {
    auto preprocessor = create(device_of(target));
    return preprocessor != nullptr ? std::move(preprocessor) : create(ImagePreprocessorType::Cpu);
  }
  switch (type) {
  case ImagePreprocessorType::Cpu:
    return std::make_unique<CpuImagePreprocessor>();
#ifdef HRT_USE_METAL
  case ImagePreprocessorType::Metal:
    return MetalImagePreprocessor::create();
#endif
#ifdef HRT_USE_CUDA
  case ImagePreprocessorType::Cuda:
    return std::make_unique<CudaImagePreprocessor>();
#endif
#ifdef HRT_USE_RKNN
  case ImagePreprocessorType::Rga:
    return std::make_unique<RgaImagePreprocessor>();
#endif
#ifdef HRT_USE_CANN
  case ImagePreprocessorType::Dvpp:
    return DvppImagePreprocessor::create();
#endif
  default:
    return nullptr;
  }
}

auto ImagePreprocessor::adjust_roi(core::Rect& /*dst_roi*/, core::Rect& /*src_roi*/, const core::Image& /*dst*/,
                                   const core::Image& /*src*/) const -> void {}

auto ImagePreprocessor::fused(core::Image& /*dst*/, const ImageTransform& /*transform*/, const core::Image& /*src*/,
                              const ImagePreprocessorParam& /*param*/, core::Stream* /*stream*/) -> bool {
  return false;
}

auto ImagePreprocessor::convert_normalize(core::Image& dst, const core::Image& src, const core::Rect& roi,
                                          const Array<float, 3>& mean, const Array<float, 3>& stddev,
                                          core::Stream* stream) -> bool {
  return convert(dst, src, stream) && normalize(dst, roi, mean, stddev, stream);
}

auto ImagePreprocessor::cpu() -> ImagePreprocessor* {
  if (type() == ImagePreprocessorType::Cpu)
    return this;
  if (cpu_ == nullptr)
    cpu_ = create(ImagePreprocessorType::Cpu);
  return cpu_.get();
}

auto ImagePreprocessor::scratch(UniquePtr<core::Image>& slot, core::MemoryType memory_type, core::ImageFormat format,
                                core::DataType data_type, core::Size size) -> core::Image* {
  if (slot == nullptr || slot->memory()->memory_type() != memory_type)
    slot = std::make_unique<core::Image>(memory_type, format, data_type, size);
  else if (!slot->reshape(format, size, data_type))
    return nullptr;
  return slot->valid() ? slot.get() : nullptr;
}

auto ImagePreprocessor::process(core::Image& dst, const core::Image& src, const core::Rect& src_rect,
                                const ImagePreprocessorParam& param, core::Stream* stream)
    -> std::optional<ImageTransform> {
  ImageTransform transform{.dst_roi = {}, .src_roi = {}, .input_size = dst.size()};
  const auto rect = src_rect.empty() ? src.rect() : src_rect;
  if (!calc_resize_position(transform.dst_roi, transform.src_roi, dst.rect(), rect, src.size(), param))
    return std::nullopt;
  // Only pad_outside clips by itself; reading past the image would abort in OpenCV.
  const auto& read = transform.src_roi;
  KZ_CHECK_RETURN(core::intersect(read, src.rect()) == read, std::nullopt,
                  "ImagePreprocessor: source region ({}, {}, {}x{}) leaves the {}x{} image; clip it with "
                  "core::pad_scale",
                  read.x, read.y, read.width, read.height, src.size().width, src.size().height);
  adjust_roi(transform.dst_roi, transform.src_roi, dst, src);

  const core::Image* input = &src;
  auto local = transform;
  if (!accepts(src.memory()->memory_type())) {
    // Host interleaved sources stage only what resize reads, as in cascaded crops: src_roi plus a margin for the
    // interpolation taps. Other layouts stage whole.
    const core::Image* from = &src;
    std::optional<core::Image> region;
    const auto format = src.format();
    if (src.memory()->memory_type() == core::MemoryType::Cpu &&
        (format == core::ImageFormat::InterBgr || format == core::ImageFormat::InterRgb ||
         format == core::ImageFormat::InterRgba)) {
      constexpr int kMargin = 2;
      const auto& roi = transform.src_roi;
      const auto area = core::intersect(
          {roi.x - kMargin, roi.y - kMargin, roi.width + 2 * kMargin, roi.height + 2 * kMargin}, src.rect());
      const auto& stride = src.stride();
      auto* base = static_cast<std::byte*>(const_cast<void*>(src.memory()->cpu(stream)));
      if (base != nullptr && area != src.rect())
        region = core::Image::wrap(core::MemoryType::Cpu,
                                   base + static_cast<std::size_t>(area.y) * stride[1] +
                                       static_cast<std::size_t>(area.x) * stride[2],
                                   format, {area.width, area.height}, src.data_type(), stride[1]);
      if (region && region->valid()) {
        from = &*region;
        local.src_roi.x -= area.x;
        local.src_roi.y -= area.y;
      }
    }
    auto* staged = scratch(staged_src_, memory_type(), format, from->data_type(), from->size());
    if (staged == nullptr ||
        !core::Buffer::copy(*staged->buffer(), *from->buffer(), core::CopyPolicy::AutoDstStride, stream)) {
      // Device memory can run out (e.g. a small shared CMA heap); the CPU then runs the whole call.
      auto* fallback = cpu();
      if (fallback == nullptr || fallback == this)
        return std::nullopt;
      return fallback->process(dst, src, src_rect, param, stream);
    }
    input = staged;
  }

  core::Image* output = &dst;
  const auto direct = accepts(dst.memory()->memory_type());
  if (!direct) {
    output = scratch(staged_dst_, memory_type(), dst.format(), dst.data_type(), dst.size());
    if (output == nullptr)
      return std::nullopt;
  }

  if (!run(*output, *input, local, param, stream))
    return std::nullopt;
  if (!direct && !core::Buffer::copy(*dst.buffer(), *output->buffer(), core::CopyPolicy::DirectCopy, stream))
    return std::nullopt;
  return transform;
}

auto ImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                       const ImagePreprocessorParam& /*param*/) const -> bool {
  return supports(ImageOperation::Fused, dst, src);
}

auto ImagePreprocessor::run(core::Image& dst, const core::Image& src, const ImageTransform& transform,
                            const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  if (supports_fused(dst, src, param))
    return fused(dst, transform, src, param, stream);

  const bool normalize = param.normalize_type != NormalizeType::None && !is_identity(param.mean, param.std);
  if (normalize && dst.data_type() != core::DataType::Float32) {
    KZ_LOG_ERROR("image_preprocessor: normalization needs a float32 input");
    return false;
  }
  auto* fallback = cpu();
  // The device operator for a step when it has one, else the CPU operator.
  auto pick = [&](ImageOperation op, const core::Image& to, const core::Image& from) -> ImagePreprocessor* {
    if (supports(op, to, from))
      return this;
    if (fallback != nullptr && fallback->supports(op, to, from))
      return fallback;
    return nullptr;
  };

  // Formats resize cannot handle (YUV, RGBA) are converted first, at source
  // size and in the source data type.
  const core::Image* source = &src;
  core::Image* converted = nullptr;
  if (pick(ImageOperation::Resize, src, src) == nullptr) {
    converted = scratch(converted_, memory_type(), dst.format(), src.data_type(), src.size());
    if (converted == nullptr)
      return false;
    source = converted;
  }
  // Resize in the source data type, then convert, unless resize can produce
  // the input format directly.
  core::Image* resized = &dst;
  if (source->data_type() != dst.data_type() || !same_layout(dst.format(), source->format())) {
    const auto format = same_layout(dst.format(), source->format()) ? dst.format() : source->format();
    resized = scratch(resized_, memory_type(), format, source->data_type(), dst.size());
    if (resized == nullptr)
      return false;
  }
  const bool pad = transform.dst_roi != dst.rect();
  // Steps that are not needed count as run by this preprocessor.
  const Array<ImageOperation, 5> ops{ImageOperation::Convert, ImageOperation::Fill, ImageOperation::Resize,
                                     ImageOperation::Convert, ImageOperation::Normalize};
  const Array<ImagePreprocessor*, 5> steps{
      converted != nullptr ? pick(ImageOperation::Convert, *converted, src) : this,
      pad ? pick(ImageOperation::Fill, *resized, *resized) : this,
      pick(ImageOperation::Resize, *resized, *source),
      resized != &dst ? pick(ImageOperation::Convert, dst, *resized) : this,
      normalize ? pick(ImageOperation::Normalize, dst, dst) : this,
  };

  // CPU operators only address host memory: unless the device runs every step, a device-only
  // buffer sends the whole call through the CPU, downloading the source and uploading the result.
  const auto on_device = std::all_of(steps.begin(), steps.end(), [this](auto* step) -> bool { return step == this; });
  if (fallback != nullptr && type() != ImagePreprocessorType::Cpu &&
      !(dst.memory()->have_cpu() && src.memory()->have_cpu()) && !on_device) {
    const core::Image* host_src = &src;
    if (!src.memory()->have_cpu()) {
      auto* staged = scratch(host_src_, core::MemoryType::Cpu, src.format(), src.data_type(), src.size());
      if (staged == nullptr ||
          !core::Buffer::copy(*staged->buffer(), *src.buffer(), core::CopyPolicy::AutoDstStride, stream))
        return false;
      if (stream != nullptr)
        stream->synchronize();
      host_src = staged;
    }
    core::Image* host_dst = &dst;
    if (!dst.memory()->have_cpu()) {
      host_dst = scratch(host_dst_, core::MemoryType::Cpu, dst.format(), dst.data_type(), dst.size());
      if (host_dst == nullptr)
        return false;
    }
    if (!fallback->run(*host_dst, *host_src, transform, param, stream))
      return false;
    return host_dst == &dst ||
           core::Buffer::copy(*dst.buffer(), *host_dst->buffer(), core::CopyPolicy::DirectCopy, stream);
  }
  for (std::size_t i = 0; i < steps.size(); ++i) {
    if (steps[i] == nullptr) {
      KZ_LOG_ERROR("image_preprocessor: no operator for step {} ({} <- {})", static_cast<int>(ops[i]),
                   kitzoo::json::Json(dst.format()).dump(), kitzoo::json::Json(src.format()).dump());
      return false;
    }
  }

  const auto& [to_convert, to_fill, to_resize, to_cast, to_normalize] = steps;
  if (converted != nullptr && !to_convert->convert(*converted, src, stream))
    return false;
  if (pad && !to_fill->fill(*resized, resized->rect(), param.default_color, stream))
    return false;
  if (!to_resize->resize(*resized, transform.dst_roi, *source, transform.src_roi, param.interpolation_mode, stream))
    return false;
  const auto roi = param.normalize_type == NormalizeType::Roi ? transform.dst_roi : dst.rect();
  if (resized != &dst) {
    if (normalize && to_cast->supports(ImageOperation::Normalize, dst, dst))
      return to_cast->convert_normalize(dst, *resized, roi, param.mean, param.std, stream);
    if (!to_cast->convert(dst, *resized, stream))
      return false;
  }
  return !normalize || to_normalize->normalize(dst, roi, param.mean, param.std, stream);
}

} // namespace hrt::preprocess
