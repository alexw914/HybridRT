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

auto head_stride(core::Size input, const core::BufferProto& proto, engine::DataLayout layout) -> std::optional<float> {
  const auto head = head_shape(proto, layout);
  if (!head || proto.shape[0] != 1 || input.width <= 0 || input.height <= 0 || head->rows <= 0 || head->cols <= 0 ||
      head->channels <= 0)
    return std::nullopt;
  const auto x = static_cast<double>(input.width) / static_cast<double>(head->cols);
  const auto y = static_cast<double>(input.height) / static_cast<double>(head->rows);
  if (x != y)
    return std::nullopt;
  return static_cast<float>(x);
}

auto resolve_strides(Vector<float>& strides, const Vector<float>& inferred) -> bool {
  KZ_CHECK_RETURN(strides.empty() || strides == inferred, false,
                  "postprocess: configured strides do not match the input/output shapes");
  strides = inferred;
  return true;
}

auto keypoint_shape(std::int64_t channels, unsigned int count, std::optional<std::size_t> reg_max)
    -> std::optional<KeypointShape> {
  KZ_CHECK_RETURN(channels > 0, std::nullopt, "postprocess: keypoint channels must be positive");
  std::optional<KeypointShape> found;
  for (const std::size_t dims : {2U, 3U}) {
    auto minimum = reg_max ? (*reg_max == 0 ? 1 : *reg_max) : 0;
    auto maximum = reg_max ? (*reg_max == 0 ? static_cast<std::size_t>(channels / 4) : *reg_max) : 0;
    if (reg_max && *reg_max == 0 && count != 0) {
      const auto box_channels = channels - static_cast<std::int64_t>(count) * static_cast<std::int64_t>(dims);
      if (box_channels <= 0 || box_channels % 4 != 0)
        continue;
      minimum = maximum = static_cast<std::size_t>(box_channels / 4);
    }
    for (auto bins = minimum; bins <= maximum; ++bins) {
      if (bins > static_cast<std::size_t>(channels / 4))
        break;
      const auto extra = channels - static_cast<std::int64_t>(bins * 4);
      if (extra <= 0 || extra % static_cast<std::int64_t>(dims) != 0)
        continue;
      const auto inferred = static_cast<std::uint64_t>(extra / static_cast<std::int64_t>(dims));
      if (inferred > std::numeric_limits<unsigned int>::max() || (count != 0 && count != inferred))
        continue;
      KZ_CHECK_RETURN(!found, std::nullopt, "postprocess: ambiguous keypoint channels; supply keypoint_num or reg_max");
      found = KeypointShape{static_cast<unsigned int>(inferred), dims, bins};
    }
  }
  KZ_CHECK_RETURN(found, std::nullopt, "postprocess: channels do not match keypoint_num or reg_max");
  return found;
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
    postprocessor = make_embedding(type);
  if (postprocessor == nullptr)
    postprocessor = make_keypoint(type);
  if (postprocessor == nullptr)
    postprocessor = make_segment(type);
  if (postprocessor == nullptr)
    postprocessor = make_ocr(type);
  if (postprocessor == nullptr)
    return {"postprocess.type: unknown \"" + String(type) + "\""};
  return postprocessor->check_config(config);
}

} // namespace hrt::postprocess
