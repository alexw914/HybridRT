// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/common.cpp
// Description: Implements shared postprocessing helpers and host staging of
//              engine outputs.
// -----------------------------------------------------------------------------

#include "include/postprocess/common.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <numeric>

#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

PostprocessorBase::~PostprocessorBase() = default;

auto PostprocessorBase::to_host(const core::Buffers& outputs, core::Stream* stream) -> const core::Buffers* {
  view_.resize(outputs.size());
  host_.resize(outputs.size());
  bool copied = false;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    if (output->memory()->have_cpu()) {
      view_[i] = output;
      continue;
    }
    if (host_[i] == nullptr)
      host_[i] = std::make_shared<core::Buffer>(core::MemoryType::Cpu);
    if (!core::Buffer::copy(*host_[i], *output, core::CopyPolicy::AutoDstStride, stream))
      return nullptr;
    host_[i]->scale = output->scale;
    host_[i]->zero_point = output->zero_point;
    view_[i] = host_[i];
    copied = true;
  }
  if (copied && stream != nullptr)
    stream->synchronize();
  return &view_;
}

auto expand_per_class(Vector<float>& values, std::size_t class_num, const char* name) -> bool {
  if (values.size() == 1 && class_num > 1)
    values.resize(class_num, values.front());
  if (values.size() != class_num) {
    KZ_LOG_ERROR("postprocess: {} needs 1 or {} values, got {}", name, class_num, values.size());
    return false;
  }
  if (!std::all_of(values.begin(), values.end(), [](float v) -> bool { return v >= 0.0F && v <= 1.0F; })) {
    KZ_LOG_ERROR("postprocess: {} values must be in [0, 1]", name);
    return false;
  }
  return true;
}

auto engine_input_size(const engine::Engine& engine) -> std::optional<core::Size> {
  const auto& inputs = engine.input_protos();
  if (inputs.size() != 1 || inputs[0].shape.size() != 4 || inputs[0].shape[0] != 1) {
    KZ_LOG_ERROR("postprocess: expected one 4-D batch-1 input");
    return std::nullopt;
  }
  const auto& shape = inputs[0].shape;
  if (engine.proto().input_layout == engine::DataLayout::Nchw)
    return core::Size{static_cast<int>(shape[3]), static_cast<int>(shape[2])};
  return core::Size{static_cast<int>(shape[2]), static_cast<int>(shape[1])};
}

auto head_shape(const core::BufferProto& proto, engine::DataLayout layout) -> std::optional<HeadShape> {
  const auto& shape = proto.shape;
  if (shape.size() != 4)
    return std::nullopt;
  if (layout == engine::DataLayout::Nchw)
    return HeadShape{shape[1], shape[2], shape[3]};
  return HeadShape{shape[3], shape[1], shape[2]};
}

auto float_data(const core::Buffer& output, core::Buffer& scratch) -> const float* {
  if (output.data_type() == core::DataType::Float32) {
    if (output.is_contiguous())
      return static_cast<const float*>(output.memory()->cpu());
    auto& source = const_cast<core::Buffer&>(output);
    if (!core::Buffer::copy(scratch, source, core::CopyPolicy::AutoDstStride))
      return nullptr;
    return static_cast<const float*>(scratch.memory()->cpu());
  }
  if (output.data_type() == core::DataType::Int8) {
    // Strided outputs are packed first; engines rarely produce them.
    const core::Buffer* source = &output;
    core::Buffer packed(core::MemoryType::Cpu, core::DataType::Int8);
    if (!output.is_contiguous()) {
      if (!core::Buffer::copy(packed, const_cast<core::Buffer&>(output), core::CopyPolicy::AutoDstStride))
        return nullptr;
      source = &packed;
    }
    if (!scratch.reshape(core::DataType::Float32, output.shape()))
      return nullptr;
    const auto* in = static_cast<const std::int8_t*>(source->memory()->cpu());
    auto* out = static_cast<float*>(scratch.memory()->mutable_cpu());
    const auto count = static_cast<std::size_t>(output.volume());
    for (std::size_t i = 0; i < count; ++i)
      out[i] = dequantize(in[i], output.scale, output.zero_point);
    return out;
  }
  KZ_LOG_ERROR("postprocess: unsupported output type {}", to_string(output.data_type()));
  return nullptr;
}

RowView::RowView(const core::Buffer& output, core::Buffer& scratch) {
  const auto& shape = output.shape();
  if (shape.empty())
    return;
  count_ = static_cast<std::size_t>(shape.back());
  if (output.data_type() == core::DataType::Int8 && output.is_contiguous()) {
    data_ = output.memory()->cpu();
    int8_ = true;
    scale_ = output.scale;
    zero_point_ = output.zero_point;
    return;
  }
  data_ = float_data(output, scratch);
}

auto RowView::above(std::size_t row, float threshold, Vector<std::uint32_t>& out) const -> void {
  if (!int8_) {
    const auto* values = static_cast<const float*>(data_) + row * count_;
    for (std::size_t i = 0; i < count_; ++i) {
      if (values[i] > threshold)
        out.push_back(static_cast<std::uint32_t>(i));
    }
    return;
  }
  const auto* values = static_cast<const std::int8_t*>(data_) + row * count_;
  // value > threshold needs q > threshold / scale + zero_point; one integer
  // below that bound guards float rounding, and at() decides exactly.
  const auto bound = scale_ > 0.0F
                         ? std::floor(static_cast<double>(threshold) / static_cast<double>(scale_) + zero_point_) - 1.0
                         : -129.0;
  const auto integer = static_cast<std::int32_t>(std::clamp(bound, -129.0, 127.0));
  for (std::size_t i = 0; i < count_; ++i) {
    if (values[i] > integer && dequantize(values[i], scale_, zero_point_) > threshold)
      out.push_back(static_cast<std::uint32_t>(i));
  }
}

auto nms_keep(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t> {
  Vector<std::size_t> order(rects.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) -> bool { return scores[a] > scores[b]; });
  Vector<std::size_t> kept;
  for (auto index : order) {
    const auto& rect = rects[index];
    const auto overlaps = std::any_of(kept.begin(), kept.end(), [&](std::size_t other) -> bool {
      const auto inter = core::intersect(rect, rects[other]).area();
      return inter / (rect.area() + rects[other].area() - inter) >= iou_threshold;
    });
    if (!overlaps)
      kept.push_back(index);
  }
  return kept;
}

auto to_source(const core::Rect2f& rect, const preprocess::ImageTransform& transform) -> core::Rect2f {
  const auto& dst = transform.dst_roi;
  const auto& src = transform.src_roi;
  const auto scale_x = static_cast<float>(src.width) / static_cast<float>(dst.width);
  const auto scale_y = static_cast<float>(src.height) / static_cast<float>(dst.height);
  return {(rect.x - static_cast<float>(dst.x)) * scale_x + static_cast<float>(src.x),
          (rect.y - static_cast<float>(dst.y)) * scale_y + static_cast<float>(src.y), rect.width * scale_x,
          rect.height * scale_y};
}

auto to_source(const core::Point2f& point, const preprocess::ImageTransform& transform) -> core::Point2f {
  const auto mapped = to_source(core::Rect2f{point.x, point.y, 0.0F, 0.0F}, transform);
  return {mapped.x, mapped.y};
}

auto check_config(std::string_view type, const kitzoo::json::Json& config) -> Vector<String> {
  UniquePtr<PostprocessorBase> postprocessor = make_detect(type);
  if (postprocessor == nullptr)
    postprocessor = make_rotated(type);
  if (postprocessor == nullptr)
    postprocessor = make_classify(type);
  if (postprocessor == nullptr)
    postprocessor = make_keypoint(type);
  if (postprocessor == nullptr)
    postprocessor = make_segment(type);
  if (postprocessor == nullptr)
    return {"postprocess.type: unknown \"" + String(type) + "\""};
  return postprocessor->check_config(config);
}

} // namespace hrt::postprocess
