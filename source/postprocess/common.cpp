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

auto is_float_outputs(const engine::Engine& engine, std::string_view name) -> bool {
  for (const auto& output : engine.output_protos()) {
    KZ_CHECK_RETURN(output.data_type == core::DataType::Float32 || output.data_type == core::DataType::Int8, false,
                    "{}: outputs must be float32 or int8, got {}", name, to_string(output.data_type));
  }
  return true;
}

auto is_4d_outputs(const engine::Engine& engine, std::string_view name) -> bool {
  const auto& outputs = engine.output_protos();
  KZ_CHECK_RETURN(!outputs.empty() &&
                      std::all_of(outputs.begin(), outputs.end(),
                                  [](const core::BufferProto& output) -> bool { return output.shape.size() == 4; }),
                  false, "{}: needs 4-D head outputs", name);
  return true;
}

auto HeadGridShape::from_proto(const core::BufferProto& proto, engine::DataLayout layout) -> HeadGridShape {
  const auto& shape = proto.shape;
  if (layout == engine::DataLayout::Nchw)
    return HeadGridShape{shape[0], shape[1], shape[2], shape[3]};
  return HeadGridShape{shape[0], shape[3], shape[1], shape[2]};
}

auto HeadGridShape::stride(core::Size input) const -> core::Point2f {
  return {static_cast<float>(static_cast<double>(input.width) / static_cast<double>(cols)),
          static_cast<float>(static_cast<double>(input.height) / static_cast<double>(rows))};
}

HeadGridView::HeadGridView(const core::Buffer& buffer, std::optional<engine::DataLayout> layout)
    : int8_(buffer.data_type() == core::DataType::Int8), scale_(buffer.scale), zero_point_(buffer.zero_point) {
  const auto& shape = buffer.shape();
  const auto& stride = buffer.stride();
  const auto dims = shape.size();
  const auto item = int8_ ? sizeof(std::int8_t) : sizeof(float);
  if (!layout) {
    rows_ = 1;
    cols_ = shape[dims - 1];
    col_step_ = 1;
    channel_step_ = stride[dims - 1] / item;
  } else if (*layout == engine::DataLayout::Nchw) {
    rows_ = shape[dims - 2];
    cols_ = shape[dims - 1];
    row_step_ = stride[dims - 1] / item;
    col_step_ = 1;
    channel_step_ = stride[dims - 2] / item;
  } else {
    rows_ = shape[dims - 3];
    cols_ = shape[dims - 2];
    row_step_ = stride[dims - 2] / item;
    col_step_ = stride[dims - 1] / item;
    channel_step_ = 1;
  }
  data_ = buffer.memory()->cpu();
}

auto HeadGridView::raw_bound(float threshold) const -> float {
  if (!int8_)
    return threshold;
  if (!(scale_ > 0.0F))
    return std::numeric_limits<float>::lowest();
  const auto bound = std::floor(static_cast<double>(threshold) / static_cast<double>(scale_) + zero_point_);
  return static_cast<float>(std::clamp(bound, -129.0, 128.0));
}

namespace {

// raw_bound() in the element type: every int8 reaches -129 and none reaches 128.
template <typename T>
auto typed_bound(float bound) {
  if constexpr (std::is_same_v<T, float>)
    return bound;
  else
    return static_cast<std::int32_t>(std::clamp(bound, -129.0F, 128.0F));
}

template <typename T>
auto scan_max_cells(const HeadGridView& grid, std::size_t first, std::size_t count, float raw_bound, Vector<T>& best,
                    Vector<std::uint32_t>& cells) -> void {
  const auto rows = grid.rows();
  const auto cols = grid.cols();
  const auto bound = typed_bound<T>(raw_bound);
  cells.clear();
  if (count == 1) {
    const auto step = grid.col_step();
    for (std::int64_t y = 0; y < rows; ++y) {
      const auto* in = grid.row<T>(y, first);
      for (std::int64_t x = 0; x < cols; ++x) {
        if (in[static_cast<std::size_t>(x) * step] >= bound)
          cells.push_back(static_cast<std::uint32_t>(y * cols + x));
      }
    }
    return;
  }
  best.assign(static_cast<std::size_t>(rows * cols), std::numeric_limits<T>::lowest());
  if (grid.col_step() == 1) {
    for (std::size_t c = first; c < first + count; ++c) {
      for (std::int64_t y = 0; y < rows; ++y) {
        const auto* in = grid.row<T>(y, c);
        auto* out = best.data() + y * cols;
        for (std::int64_t x = 0; x < cols; ++x)
          out[x] = std::max(out[x], in[x]);
      }
    }
  } else {
    const auto step = grid.col_step();
    for (std::int64_t y = 0; y < rows; ++y) {
      for (std::int64_t x = 0; x < cols; ++x) {
        auto value = std::numeric_limits<T>::lowest();
        for (std::size_t c = first; c < first + count; ++c)
          value = std::max(value, grid.row<T>(y, c)[static_cast<std::size_t>(x) * step]);
        best[static_cast<std::size_t>(y * cols + x)] = value;
      }
    }
  }
  for (std::size_t i = 0; i < best.size(); ++i) {
    if (best[i] >= bound)
      cells.push_back(static_cast<std::uint32_t>(i));
  }
}

template <typename T>
auto scan_channel_cells(const HeadGridView& grid, std::size_t channel, float threshold, Vector<std::uint32_t>& cells)
    -> void {
  const auto* values = grid.row<T>(0, channel);
  const auto step = grid.col_step();
  const auto bound = typed_bound<T>(grid.raw_bound(threshold));
  cells.clear();
  for (std::int64_t x = 0; x < grid.cols(); ++x) {
    if (values[static_cast<std::size_t>(x) * step] >= bound && grid.at(0, x, channel) > threshold)
      cells.push_back(static_cast<std::uint32_t>(x));
  }
}

} // namespace

auto CellScanner::scan_max_score(const HeadGridView& grid, std::size_t first, std::size_t count, float bound)
    -> const Vector<std::uint32_t>& {
  if (grid.int8())
    scan_max_cells(grid, first, count, bound, int8_best_, cells_);
  else
    scan_max_cells(grid, first, count, bound, float_best_, cells_);
  return cells_;
}

auto CellScanner::scan_channel(const HeadGridView& grid, std::size_t channel, float threshold)
    -> const Vector<std::uint32_t>& {
  if (grid.int8())
    scan_channel_cells<std::int8_t>(grid, channel, threshold, cells_);
  else
    scan_channel_cells<float>(grid, channel, threshold, cells_);
  return cells_;
}

auto make_scratch(const engine::Engine& engine) -> Vector<core::Buffer> {
  Vector<core::Buffer> scratch;
  for (const auto& output : engine.output_protos()) {
    auto& buffer = scratch.emplace_back(core::MemoryType::Cpu, core::DataType::Float32);
    if (output.data_type == core::DataType::Int8)
      buffer.reshape(output.shape);
  }
  return scratch;
}

auto dequantize_buffer(const core::Buffer& output, core::Buffer& scratch) -> void {
  const auto* base = static_cast<const std::uint8_t*>(output.memory()->cpu());
  auto* out = static_cast<float*>(scratch.memory()->mutable_cpu());
  const auto convert = [&](const std::uint8_t* in, std::size_t count) -> void {
    const auto* values = static_cast<const std::int8_t*>(static_cast<const void*>(in));
    for (std::size_t i = 0; i < count; ++i)
      out[i] = core::dequantize(values[i], output.scale, output.zero_point);
    out += count;
  };
  const auto count = static_cast<std::size_t>(output.volume());
  if (output.is_contiguous()) {
    convert(base, count);
    return;
  }
  // Rows of the last dimension are contiguous; the outer index of a row is
  // decomposed against the shape, dimension k - 1 stepping stride[k] bytes.
  const auto& shape = output.shape();
  const auto& stride = output.stride();
  const auto cols = static_cast<std::size_t>(shape.back());
  for (std::size_t row = 0; cols != 0 && row < count / cols; ++row) {
    std::size_t offset = 0;
    auto rest = row;
    for (auto k = shape.size() - 1; k > 0; --k) {
      const auto dim = static_cast<std::size_t>(shape[k - 1]);
      offset += rest % dim * stride[k];
      rest /= dim;
    }
    convert(base + offset, cols);
  }
}

auto nms_indices(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t> {
  Vector<std::size_t> order(rects.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) -> bool { return scores[a] > scores[b]; });
  Vector<std::size_t> kept;
  for (auto index : order) {
    const auto& rect = rects[index];
    const auto overlaps = std::any_of(kept.begin(), kept.end(), [&](std::size_t other) -> bool {
      return core::iou(rect, rects[other]) >= iou_threshold;
    });
    if (!overlaps)
      kept.push_back(index);
  }
  return kept;
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
