// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/common.hpp
// Description: Declares helpers shared by postprocessors: head grid views, output
//              validation and float32 copies of outputs.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_COMMON_HPP
#define HRT_POSTPROCESS_COMMON_HPP

#include <hrt/engine/engine.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "include/core/config.hpp"
#include "include/core/math.hpp"

namespace hrt::postprocess {

// Values per pose keypoint: x, y and visibility.
inline constexpr std::size_t kKeypointValues = 3;

// Broadcasts one value to `class_num` and checks each is in [0, 1]; fails on
// a zero class_num.
auto expand_per_class(Vector<float>& values, std::size_t class_num, const char* name) -> bool;

// Whether every engine output is float32 or int8, the types HeadGridView and
// dequantize_buffer() read. Checked once in init, so decode() relies on it.
auto is_float_outputs(const engine::Engine& engine, std::string_view name) -> bool;

// Whether the engine has outputs and every one is 4-D, as head_channels() and
// head_stride() expect.
auto is_4d_outputs(const engine::Engine& engine, std::string_view name) -> bool;

// Channels of a 4-D head output under the engine layout; see is_4d_outputs().
auto head_channels(const core::BufferProto& proto, engine::DataLayout layout) -> std::int64_t;

// Input-to-head scale along x and y of a 4-D head output.
auto head_stride(const core::BufferProto& proto, engine::DataLayout layout, core::Size input) -> core::Point2f;

// Element access into one engine output as a grid of rows x cols cells with
// channels, NCHW or NHWC, float32 or quantized int8, honoring buffer strides. Without a layout, a decoded
// [1, rows, N] output is one grid row of N cells whose channels are its rows.
// Other data types are rejected in init by is_float_outputs().
class HeadGridView {
public:
  HeadGridView(const core::Buffer& buffer, std::optional<engine::DataLayout> layout);

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

  // Element (index, column) of a layout-less [1, N, C] output.
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

// Expected value of the softmax over `bins` consecutive channels starting at
// `first` (YOLOv8 distribution focal loss decoding).
inline auto dfl(const HeadGridView& grid, std::int64_t y, std::int64_t x, std::size_t first, std::size_t bins)
    -> float {
  auto max = grid.at(y, x, first);
  for (std::size_t i = 1; i < bins; ++i)
    max = std::max(max, grid.at(y, x, first + i));
  float sum = 0.0F;
  float weighted = 0.0F;
  for (std::size_t i = 0; i < bins; ++i) {
    const auto e = std::exp(grid.at(y, x, first + i) - max);
    sum += e;
    weighted += e * static_cast<float>(i);
  }
  return weighted / sum;
}

// Dequantizes an int8 output into `scratch`, a float32 host buffer of the
// same shape, honoring strides. Decoders that read a whole output read
// float32 outputs in place and call this for int8 ones; decoders reading part
// of an output use HeadGridView instead.
auto dequantize_buffer(const core::Buffer& output, core::Buffer& scratch) -> void;

// One dequantize_buffer() buffer per engine output, shaped like the int8 ones.
auto create_scratch(const engine::Engine& engine) -> Vector<core::Buffer>;

// Indices of the rects kept by greedy NMS, highest score first.
auto nms_indices(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t>;

// Checks and initializes a postprocessor made for `type`. Aborts on an unknown
// type, a config check_config rejects, an engine without one image input, or
// engine outputs init() rejects. init() may therefore dereference
// engine.input_size().
template <typename Postprocessor>
auto check_init(UniquePtr<Postprocessor> postprocessor, std::string_view type, const kitzoo::json::Json& config,
                const engine::Engine& engine) -> UniquePtr<Postprocessor> {
  KZ_CHECK_MSG(postprocessor != nullptr, "postprocess: unknown type \"{}\"", type);
  const auto problems = postprocessor->check_config(config);
  KZ_CHECK_MSG(problems.empty(), "postprocess: invalid {} config: {}", type, core::join(problems));
  KZ_CHECK_MSG(engine.input_size().has_value(), "postprocess: {} needs one 4-D image input", type);
  KZ_CHECK_MSG(postprocessor->init(config, engine), "postprocess: {} does not fit the engine", type);
  return postprocessor;
}

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_COMMON_HPP
