// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/common.hpp
// Description: Declares helpers shared by postprocessors: activation and
//              quantization math, head views and output validation.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_COMMON_HPP
#define HRT_POSTPROCESS_COMMON_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "include/core/config.hpp"

namespace hrt::postprocess {

inline auto sigmoid(float x) -> float {
  return 1.0F / (1.0F + std::exp(-x));
}

// Corners l, t, r, b of a center x, center y, width, height box.
inline auto center_box(float cx, float cy, float w, float h) -> Array<float, 4> {
  return {cx - w / 2.0F, cy - h / 2.0F, cx + w / 2.0F, cy + h / 2.0F};
}

inline auto logit(float p) -> float {
  return p < 1.0F ? std::log(p / (1.0F - p)) : std::numeric_limits<float>::max();
}

inline auto dequantize(std::int8_t value, float scale, std::int32_t zero_point) -> float {
  return scale * (static_cast<float>(value) - static_cast<float>(zero_point));
}

auto expand_per_class(Vector<float>& values, std::size_t class_num, const char* name) -> bool;

// Batch, channels and grid of one 4-D head output.
struct HeadShape {
  std::int64_t batch = 0;
  std::int64_t channels = 0;
  std::int64_t rows = 0;
  std::int64_t cols = 0;

  // Nullopt unless `proto` is 4-D.
  static auto of(const core::BufferProto& proto, engine::DataLayout layout) -> std::optional<HeadShape>;

  // Uniform input-to-head scale of a positive batch-1 grid.
  KZ_NODISCARD auto stride(core::Size input) const -> std::optional<float>;
};

// Validates an optional configured list and stores the inferred strides.
auto resolve_strides(Vector<float>& strides, const Vector<float>& inferred) -> bool;

struct KeypointShape {
  unsigned int count = 0;
  std::size_t dims = 0;
  std::size_t reg_max = 0;
};

// Resolves a unique channel partition into keypoints and optional DFL bins.
// A zero count or bin count requests inference; nullopt means no DFL channels.
auto keypoint_shape(std::int64_t channels, unsigned int count, std::optional<std::size_t> reg_max = std::nullopt)
    -> std::optional<KeypointShape>;

// Element access into one detection head output, NCHW or NHWC, float32 or
// quantized int8, honoring buffer strides. Without a layout, a decoded
// [1, rows, N] output is one grid row of N cells whose channels are its rows.
// Invalid for other data types.
class HeadGrid {
public:
  HeadGrid(const core::Buffer& buffer, std::optional<engine::DataLayout> layout);

  KZ_NODISCARD auto valid() const -> bool { return data_ != nullptr; }

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
    return int8_ ? dequantize(static_cast<const std::int8_t*>(data_)[index], scale_, zero_point_)
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
  // count) is at least `bound` (a HeadGrid::raw_bound()), in row-major order.
  // With NCHW heads the class planes are read one after another into a
  // running maximum, a sequential pass the compiler vectorizes, instead of
  // striding every plane for every cell; int8 scores stay int8.
  auto scan(const HeadGrid& grid, std::size_t first, std::size_t count, float bound) -> const Vector<std::uint32_t>&;

  // Cells x of row 0, in order, whose `channel` value exceeds `threshold`.
  // int8 values are compared as integers against the bound and dequantized
  // only when they reach it.
  auto above(const HeadGrid& grid, std::size_t channel, float threshold) -> const Vector<std::uint32_t>&;

private:
  Vector<float> float_best_;
  Vector<std::int8_t> int8_best_;
  Vector<std::uint32_t> cells_;
};

// Expected value of the softmax over `bins` consecutive channels starting at
// `first` (YOLOv8 distribution focal loss decoding).
inline auto dfl(const HeadGrid& grid, std::int64_t y, std::int64_t x, std::size_t first, std::size_t bins) -> float {
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

// Contiguous float32 view of an output for decoders that read all of it:
// the output itself when it is contiguous float32, otherwise `scratch`
// holding a dequantized or packed copy, created on first use and resized only
// when the output shape changes. Null for types other than float32 and int8.
// Decoders reading part of an output use HeadGrid instead.
auto float_view(const SharedPtr<core::Buffer>& output, SharedPtr<core::Buffer>& scratch) -> SharedPtr<core::Buffer>;

auto nms_keep(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t>;

// check_config of a decoder reading `Param` from the "postprocess" section.
template <typename Param>
auto check_section(const kitzoo::json::Json& config) -> Vector<String> {
  core::Problems problems;
  core::check_param<Param>(config, "postprocess", {"type"}, problems);
  return problems;
}

// Aborts on an unknown type, a config check_config rejects, or engine
// outputs init() rejects.
template <typename Postprocessor>
auto checked(UniquePtr<Postprocessor> postprocessor, std::string_view type, const kitzoo::json::Json& config,
             const engine::Engine& engine) -> UniquePtr<Postprocessor> {
  KZ_CHECK_MSG(postprocessor != nullptr, "postprocess: unknown type \"{}\"", type);
  const auto problems = postprocessor->check_config(config);
  KZ_CHECK_MSG(problems.empty(), "postprocess: invalid {} config: {}", type, core::join(problems));
  KZ_CHECK_MSG(postprocessor->init(config, engine), "postprocess: {} does not fit the engine", type);
  return postprocessor;
}

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_COMMON_HPP
