// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/common.hpp
// Description: Declares helpers shared by postprocessors: head grid views, int8
//              lookup tables, output validation and float32 copies of outputs.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_COMMON_HPP
#define HRT_POSTPROCESS_COMMON_HPP

#include <hrt/engine/engine.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "include/core/config.hpp"
#include "include/core/math.hpp"
#include "include/postprocess/postprocessor_params.hpp"

namespace hrt::postprocess {

// Values per pose keypoint: x, y and visibility.
inline constexpr std::size_t kKeypointValues = 3;

// Broadcasts one value to `class_num` and checks each is in [0, 1]; fails on
// a zero class_num.
auto expand_per_class(Vector<float>& values, std::size_t class_num, const char* name) -> bool;

// Whether every engine output is float32 or int8, the types HeadGridView and
// sample_floats() read. Checked once in init, so decode() relies on it.
auto is_float_outputs(const engine::Engine& engine, std::string_view name) -> bool;

// Whether the engine has outputs and every one is 4-D, as head_channels() and
// head_stride() expect.
auto is_4d_outputs(const engine::Engine& engine, std::string_view name) -> bool;

// Channels of a 4-D head output under the engine layout; see is_4d_outputs().
auto head_channels(const core::BufferProto& proto, engine::DataLayout layout) -> std::int64_t;

// Input-to-head scale along x and y of a 4-D head output.
auto head_stride(const core::BufferProto& proto, engine::DataLayout layout, core::Size input) -> core::Point2f;

// Validate head channels and fill inferred strides and per-class thresholds.
auto check_yolov5_heads(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

auto check_yolov8_heads(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

auto check_yolo26_heads(Yolo26Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

// Bytes between consecutive samples of a batched host output.
auto sample_bytes(const core::Buffer& output) -> std::size_t;

// Host address of sample `index` of a batched host output, read in place.
auto sample_data(const core::Buffer& output, std::size_t index) -> const void*;

// Elements of one sample of a batched output.
auto sample_volume(const core::Buffer& output) -> std::size_t;

// Element access into one sample of an engine output as a grid of rows x cols cells with
// channels, NCHW or NHWC, float32 or quantized int8, honoring buffer strides. Without a layout, a decoded
// [B, rows, N] output is one grid row of N cells whose channels are its rows.
// Other data types are rejected in init by is_float_outputs().
class HeadGridView {
public:
  HeadGridView(const core::Buffer& buffer, std::size_t index, std::optional<engine::DataLayout> layout);

  KZ_NODISCARD auto int8() const -> bool { return int8_; }

  KZ_NODISCARD auto rows() const -> std::int64_t { return rows_; }

  KZ_NODISCARD auto cols() const -> std::int64_t { return cols_; }

  KZ_NODISCARD auto scale() const -> float { return scale_; }

  KZ_NODISCARD auto zero_point() const -> std::int32_t { return zero_point_; }

  // Cell (y, 0) of `channel` as T, float or std::int8_t to match int8(); cells
  // of a row are col_step() elements apart.
  template <typename T>
  KZ_NODISCARD auto row(std::int64_t y, std::size_t channel) const -> const T* {
    return static_cast<const T*>(data_) + static_cast<std::size_t>(y) * row_step_ + channel * channel_step_;
  }

  KZ_NODISCARD auto col_step() const -> std::size_t { return col_step_; }

  // The int8 integer of an int8 grid.
  KZ_NODISCARD auto quantized(std::int64_t y, std::int64_t x, std::size_t channel) const -> std::int8_t {
    return static_cast<const std::int8_t*>(data_)[offset(y, x, channel)];
  }

  // The stored value: the float, or the int8 integer before dequantization.
  KZ_NODISCARD auto raw(std::int64_t y, std::int64_t x, std::size_t channel) const -> float {
    const auto index = offset(y, x, channel);
    return int8_ ? static_cast<float>(static_cast<const std::int8_t*>(data_)[index])
                 : static_cast<const float*>(data_)[index];
  }

  KZ_NODISCARD auto at(std::int64_t y, std::int64_t x, std::size_t channel) const -> float {
    const auto index = offset(y, x, channel);
    return int8_ ? core::dequantize(static_cast<const std::int8_t*>(data_)[index], scale_, zero_point_)
                 : static_cast<const float*>(data_)[index];
  }

  // Element (index, column) of a layout-less [B, N, C] output.
  KZ_NODISCARD auto at(std::size_t index, std::int64_t column) const -> float { return at(0, column, index); }

  // Bound in the raw domain: at() >= threshold implies raw() >= bound, so
  // int8 cells are filtered without dequantizing.
  KZ_NODISCARD auto raw_bound(float threshold) const -> float;

private:
  KZ_NODISCARD auto offset(std::int64_t y, std::int64_t x, std::size_t channel) const -> std::size_t {
    return static_cast<std::size_t>(y) * row_step_ + static_cast<std::size_t>(x) * col_step_ + channel * channel_step_;
  }

  const void* data_ = nullptr;
  bool int8_ = false;
  float scale_ = 1.0F;
  std::int32_t zero_point_ = 0;
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  std::size_t row_step_ = 0;
  std::size_t col_step_ = 0;
  std::size_t channel_step_ = 0;
};

// Cells of a head whose scores pass a threshold, collected into buffers
// reused across calls. Each call overwrites the previous result.
class CellScanner {
public:
  // Cells (y * cols + x) whose largest raw score over channels [first, first +
  // count) is at least `bound` (a HeadGridView::raw_bound()), in row-major order.
  // With NCHW heads the class planes are read one after another into a
  // running maximum, a sequential pass the compiler vectorizes, instead of
  // striding every plane for every cell; int8 scores stay int8.
  auto scan_max_score(const HeadGridView& grid, std::size_t first, std::size_t count, float bound)
      -> const Vector<std::uint32_t>&;

  // Cells x of row 0, in order, whose `channel` value exceeds `threshold`.
  // int8 values are compared as integers against the bound and dequantized
  // only when they reach it.
  auto scan_channel(const HeadGridView& grid, std::size_t channel, float threshold) -> const Vector<std::uint32_t>&;

private:
  Vector<float> float_best_;
  Vector<std::int8_t> int8_best_;
  Vector<std::uint32_t> cells_;
};

// Per-value results for the 256 integers of one int8 output, so decoding reads
// a table instead of dequantizing and calling exp: the sigmoid of each value,
// and exp(scale * -d) for a value d steps below a maximum (softmax weights).
// Also per-class HeadGridView::raw_bound()s of score thresholds, so int8
// scores are filtered as integers. bind() rebuilds them only when the grid's
// type or quantization changes; float grids use only the bounds.
class QuantTables {
public:
  auto bind(const HeadGridView& grid, const Vector<float>& thresholds = {}) -> void;

  // raw_bound() of thresholds[cls]: raw() >= bound(cls) exactly when at() >= thresholds[cls].
  KZ_NODISCARD auto bound(std::size_t cls) const -> float { return bounds_[cls]; }

  KZ_NODISCARD auto sigmoid(std::int8_t value) const -> float {
    return sigmoid_[static_cast<std::size_t>(value + 128)];
  }

  // exp(scale * -steps), steps in [0, 255].
  KZ_NODISCARD auto exp_below(std::int32_t steps) const -> float { return exp_below_[static_cast<std::size_t>(steps)]; }

private:
  bool ready_ = false;
  bool int8_ = false;
  float scale_ = 0.0F;
  std::int32_t zero_point_ = 0;
  Vector<float> bounds_;
  Array<float, 256> sigmoid_{};
  Array<float, 256> exp_below_{};
};

// core::sigmoid(grid.at(y, x, channel)), from `tables` (bound to `grid`) for int8 grids.
inline auto sigmoid_at(const HeadGridView& grid, const QuantTables& tables, std::int64_t y, std::int64_t x,
                       std::size_t channel) -> float {
  return grid.int8() ? tables.sigmoid(grid.quantized(y, x, channel)) : core::sigmoid(grid.at(y, x, channel));
}

// Turns the logit scores of decoded boxes into probabilities. NMS ranks the
// logits (the sigmoid keeps their order), so only kept boxes pay for it.
template <typename ResultMap>
auto sigmoid_scores(ResultMap& result) -> void {
  for (auto& [cls, boxes] : result) {
    for (auto& box : boxes)
      box.score = core::sigmoid(box.score);
  }
}

// Expected value of the softmax over `bins` consecutive channels starting at
// `first` (YOLOv8 distribution focal loss decoding). int8 grids take their
// weights from `tables`, bound to `grid`.
inline auto dfl(const HeadGridView& grid, const QuantTables& tables, std::int64_t y, std::int64_t x, std::size_t first,
                std::size_t bins) -> float {
  float sum = 0.0F;
  float weighted = 0.0F;
  if (grid.int8()) {
    auto max = grid.quantized(y, x, first);
    for (std::size_t i = 1; i < bins; ++i)
      max = std::max(max, grid.quantized(y, x, first + i));
    for (std::size_t i = 0; i < bins; ++i) {
      const auto e = tables.exp_below(max - grid.quantized(y, x, first + i));
      sum += e;
      weighted += e * static_cast<float>(i);
    }
    return weighted / sum;
  }
  auto max = grid.at(y, x, first);
  for (std::size_t i = 1; i < bins; ++i)
    max = std::max(max, grid.at(y, x, first + i));
  for (std::size_t i = 0; i < bins; ++i) {
    const auto e = std::exp(grid.at(y, x, first + i) - max);
    sum += e;
    weighted += e * static_cast<float>(i);
  }
  return weighted / sum;
}

// Index of the first maximum, as std::max_element. The maximum is found by a
// reduction the compiler vectorizes, then its first position, since
// std::max_element on int8 runs one value at a time.
inline auto first_max(const std::int8_t* values, std::size_t count) -> std::size_t {
  std::int8_t max = std::numeric_limits<std::int8_t>::lowest();
  for (std::size_t i = 0; i < count; ++i)
    max = std::max(max, values[i]);
  return static_cast<std::size_t>(std::find(values, values + count, max) - values);
}

// float32 values of sample `index` of a host output: float32 outputs in
// place, int8 ones dequantized into `scratch` (create_scratch()), honoring
// strides. Decoders that read a whole sample use this; decoders reading part
// of one use HeadGridView instead.
auto sample_floats(const core::Buffer& output, std::size_t index, core::Buffer& scratch) -> const float*;

// sample_floats() over the whole of an output without a batch dimension.
auto output_floats(const core::Buffer& output, core::Buffer& scratch) -> const float*;

// One sample_floats() buffer per engine output, holding one sample of the int8
// ones; output_floats() grows it on first use.
auto create_scratch(const engine::Engine& engine) -> Vector<core::Buffer>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_COMMON_HPP
