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
#include <opencv2/core/utility.hpp>

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

auto rgb_order(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::InterRgb || format == core::ImageFormat::PlanarRgb ||
         format == core::ImageFormat::InterRgba;
}

// `color`, given in the channel order of the input `dst`, in the channel order of `image`.
auto ordered_color(const Array<float, 3>& color, const core::Image& image, const core::Image& dst) -> Array<float, 3> {
  if (rgb_order(image.format()) == rgb_order(dst.format()))
    return color;
  return {color[2], color[1], color[0]};
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
  const auto padded = core::pad_scale(core::pad_aspect_ratio(core::Rect2d(rect), aspect), static_cast<double>(scale));
  return {static_cast<int>(std::lround(padded.x)), static_cast<int>(std::lround(padded.y)),
          static_cast<int>(std::lround(padded.width)), static_cast<int>(std::lround(padded.height))};
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
    // As Ultralytics' LetterBox: the scaled size rounds half to even, as Python's round(), to at least 1, and the
    // offset rounds down.
    const core::Size size{std::max(1, static_cast<int>(std::nearbyint(src_rect.width * ratio))),
                          std::max(1, static_cast<int>(std::nearbyint(src_rect.height * ratio)))};
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

auto ImagePreprocessor::stage(core::Image& dst, const core::Image& src, const core::Rect& src_rect,
                              const ImagePreprocessorParam& param, core::Stream* stream,
                              UniquePtr<core::Image>& src_slot, UniquePtr<core::Image>& dst_slot, bool& to_cpu)
    -> std::optional<Staged> {
  to_cpu = false;
  Staged staged{.transform = {.dst_roi = {}, .src_roi = {}, .input_size = dst.size()},
                .local = {},
                .input = &src,
                .output = &dst};
  auto& transform = staged.transform;
  const auto rect = src_rect.empty() ? src.rect() : src_rect;
  if (!calc_resize_position(transform.dst_roi, transform.src_roi, dst.rect(), rect, src.size(), param))
    return std::nullopt;
  // Only pad_outside clips by itself; reading past the image would abort in OpenCV.
  const auto& read = transform.src_roi;
  KZ_CHECK_RETURN(core::intersect(read, src.rect()) == read, std::nullopt,
                  "ImagePreprocessor: source region ({}, {}, {}x{}) leaves the {}x{} image; clip it with "
                  "core::intersect",
                  read.x, read.y, read.width, read.height, src.size().width, src.size().height);
  adjust_roi(transform.dst_roi, transform.src_roi, dst, src);

  staged.local = transform;
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
        staged.local.src_roi.x -= area.x;
        staged.local.src_roi.y -= area.y;
      }
    }
    const auto* input = scratch(src_slot, memory_type(), format, from->data_type(), from->size());
    if (input == nullptr ||
        !core::Buffer::copy(*input->buffer(), *from->buffer(), core::CopyPolicy::AutoDstStride, stream)) {
      // Device memory can run out (e.g. a small shared CMA heap); the CPU then runs the whole call.
      to_cpu = true;
      return std::nullopt;
    }
    staged.input = input;
  }

  if (!accepts(dst.memory()->memory_type())) {
    staged.output = scratch(dst_slot, memory_type(), dst.format(), dst.data_type(), dst.size());
    if (staged.output == nullptr)
      return std::nullopt;
  }
  return staged;
}

auto ImagePreprocessor::process(core::Image& dst, const core::Image& src, const core::Rect& src_rect,
                                const ImagePreprocessorParam& param, core::Stream* stream)
    -> std::optional<ImageTransform> {
  bool to_cpu = false;
  const auto staged = stage(dst, src, src_rect, param, stream, staged_src_, staged_dst_, to_cpu);
  if (to_cpu) {
    auto* fallback = cpu();
    if (fallback == nullptr || fallback == this)
      return std::nullopt;
    return fallback->process(dst, src, src_rect, param, stream);
  }
  if (!staged || !run(*staged->output, *staged->input, staged->local, param, stream))
    return std::nullopt;
  if (staged->output != &dst &&
      !core::Buffer::copy(*dst.buffer(), *staged->output->buffer(), core::CopyPolicy::DirectCopy, stream))
    return std::nullopt;
  return staged->transform;
}

auto ImagePreprocessor::process(std::span<core::Image> dsts, std::span<const core::Image> srcs,
                                std::span<const core::Rect> src_rects, const ImagePreprocessorParam& param,
                                core::Stream* stream) -> std::optional<Vector<ImageTransform>> {
  KZ_CHECK_RETURN(!srcs.empty() && dsts.size() == srcs.size() && (src_rects.empty() || src_rects.size() == srcs.size()),
                  std::nullopt, "ImagePreprocessor: got {} targets and {} source rects for {} sources", dsts.size(),
                  src_rects.size(), srcs.size());
  Vector<std::optional<ImageTransform>> done(srcs.size());
  auto run_sample = [&](ImagePreprocessor& processor, std::size_t i) -> void {
    done[i] = processor.process(dsts[i], srcs[i], src_rects.empty() ? core::Rect{} : src_rects[i], param, stream);
  };
  // Staging keeps the formats, so the fused check holds before it.
  const auto fuses_all = [&]() -> bool {
    for (std::size_t i = 0; i < srcs.size(); ++i) {
      if (!supports_fused(dsts[i], srcs[i], param))
        return false;
    }
    return true;
  };
  if (type() == ImagePreprocessorType::Cpu && srcs.size() > 1) {
    while (lanes_.size() + 1 < srcs.size())
      lanes_.push_back(create(ImagePreprocessorType::Cpu));
    cv::parallel_for_(cv::Range(0, static_cast<int>(srcs.size())), [&](const cv::Range& range) -> void {
      for (auto i = static_cast<std::size_t>(range.start); i < static_cast<std::size_t>(range.end); ++i)
        run_sample(i == 0 ? *this : *lanes_[i - 1], i);
    });
  } else if (srcs.size() > 1 && fuses_all()) {
    return process_fused(dsts, srcs, src_rects, param, stream);
  } else {
    // Device samples run in stream order and reuse one set of scratch images.
    for (std::size_t i = 0; i < srcs.size(); ++i)
      run_sample(*this, i);
  }
  Vector<ImageTransform> transforms;
  transforms.reserve(srcs.size());
  for (const auto& transform : done) {
    if (!transform)
      return std::nullopt;
    transforms.push_back(*transform);
  }
  return transforms;
}

auto ImagePreprocessor::process_fused(std::span<core::Image> dsts, std::span<const core::Image> srcs,
                                      std::span<const core::Rect> src_rects, const ImagePreprocessorParam& param,
                                      core::Stream* stream) -> std::optional<Vector<ImageTransform>> {
  // Every sample is staged before any operator runs, so each needs its own staging images.
  batch_srcs_.resize(std::max(batch_srcs_.size(), srcs.size()));
  batch_dsts_.resize(std::max(batch_dsts_.size(), srcs.size()));
  Vector<ImageTransform> transforms;
  Vector<ImageTransform> locals;
  Vector<const core::Image*> inputs;
  Vector<core::Image*> outputs;
  for (std::size_t i = 0; i < srcs.size(); ++i) {
    bool to_cpu = false;
    const auto rect = src_rects.empty() ? core::Rect{} : src_rects[i];
    const auto staged = stage(dsts[i], srcs[i], rect, param, stream, batch_srcs_[i], batch_dsts_[i], to_cpu);
    if (!staged) {
      auto* fallback = to_cpu ? cpu() : nullptr;
      if (fallback == nullptr || fallback == this)
        return std::nullopt;
      return fallback->process(dsts, srcs, src_rects, param, stream);
    }
    transforms.push_back(staged->transform);
    locals.push_back(staged->local);
    inputs.push_back(staged->input);
    outputs.push_back(staged->output);
  }
  if (!fused_batch(outputs, locals, inputs, param, stream))
    return std::nullopt;
  for (std::size_t i = 0; i < srcs.size(); ++i) {
    if (outputs[i] != &dsts[i] &&
        !core::Buffer::copy(*dsts[i].buffer(), *outputs[i]->buffer(), core::CopyPolicy::DirectCopy, stream))
      return std::nullopt;
  }
  return transforms;
}

auto ImagePreprocessor::fused_batch(std::span<core::Image* const> dsts, std::span<const ImageTransform> transforms,
                                    std::span<const core::Image* const> srcs, const ImagePreprocessorParam& param,
                                    core::Stream* stream) -> bool {
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    if (!fused(*dsts[i], transforms[i], *srcs[i], param, stream))
      return false;
  }
  return true;
}

auto ImageTransform::to_source(const core::Rect2f& rect) const -> core::Rect2f {
  const auto scale_x = static_cast<float>(src_roi.width) / static_cast<float>(dst_roi.width);
  const auto scale_y = static_cast<float>(src_roi.height) / static_cast<float>(dst_roi.height);
  return {(rect.x - static_cast<float>(dst_roi.x)) * scale_x + static_cast<float>(src_roi.x),
          (rect.y - static_cast<float>(dst_roi.y)) * scale_y + static_cast<float>(src_roi.y), rect.width * scale_x,
          rect.height * scale_y};
}

auto ImageTransform::to_source(const core::Point2f& point) const -> core::Point2f {
  const auto mapped = to_source(core::Rect2f{point.x, point.y, 0.0F, 0.0F});
  return {mapped.x, mapped.y};
}

auto ImageTransform::clip_to_source(const core::Rect2f& rect) const -> core::Rect2f {
  return core::intersect(to_source(rect), core::Rect2f(src_roi));
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
  if (pad && !to_fill->fill(*resized, resized->rect(), ordered_color(param.default_color, *resized, dst), stream))
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
