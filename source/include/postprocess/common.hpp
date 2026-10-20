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

auto engine_input_size(const engine::Engine& engine) -> std::optional<core::Size>;

struct HeadShape {
  std::int64_t channels = 0;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
};

auto head_shape(const core::BufferProto& proto, engine::DataLayout layout) -> std::optional<HeadShape>;

// Returns the uniform input-to-head scale for a positive batch-1 grid.
auto head_stride(core::Size input, const core::BufferProto& proto, engine::DataLayout layout) -> std::optional<float>;

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
template <typename T>
class HeadGrid {
public:
  HeadGrid(const core::Buffer& buffer, std::optional<engine::DataLayout> layout)
      : data_(static_cast<const T*>(buffer.memory()->cpu())), scale_(buffer.scale), zero_point_(buffer.zero_point) {
    const auto& shape = buffer.shape();
    const auto& stride = buffer.stride();
    const auto dims = shape.size();
    const auto item = sizeof(T);
    if (!layout) {
      if (dims < 2) {
        data_ = nullptr;
        return;
      }
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
  }

  KZ_NODISCARD auto valid() const -> bool { return data_ != nullptr; }

  KZ_NODISCARD auto rows() const -> std::int64_t { return rows_; }

  KZ_NODISCARD auto cols() const -> std::int64_t { return cols_; }

  // Cell (y, 0) of `channel`; cells of a row are col_step() elements apart.
  KZ_NODISCARD auto row(std::int64_t y, std::size_t channel) const -> const T* {
    return data_ + static_cast<std::size_t>(y) * row_step_ + channel * channel_step_;
  }

  KZ_NODISCARD auto col_step() const -> std::size_t { return col_step_; }

  KZ_NODISCARD auto raw(std::int64_t y, std::int64_t x, std::size_t channel) const -> T {
    return data_[static_cast<std::size_t>(y) * row_step_ + static_cast<std::size_t>(x) * col_step_ +
                 channel * channel_step_];
  }

  KZ_NODISCARD auto at(std::int64_t y, std::int64_t x, std::size_t channel) const -> float {
    if constexpr (std::is_same_v<T, float>)
      return raw(y, x, channel);
    else
      return dequantize(raw(y, x, channel), scale_, zero_point_);
  }

  // Bound in the element domain: at() >= threshold implies raw() >= bound, so
  // int8 cells are filtered without dequantizing.
  KZ_NODISCARD auto raw_bound(float threshold) const {
    if constexpr (std::is_same_v<T, float>) {
      return threshold;
    } else {
      if (!(scale_ > 0.0F))
        return std::numeric_limits<std::int32_t>::min();
      const auto bound = std::floor(static_cast<double>(threshold) / static_cast<double>(scale_) + zero_point_);
      return static_cast<std::int32_t>(std::clamp(bound, -129.0, 128.0));
    }
  }

private:
  const T* data_;
  float scale_;
  std::int32_t zero_point_;
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  std::size_t row_step_ = 0;
  std::size_t col_step_ = 0;
  std::size_t channel_step_ = 0;
};

// Reusable buffers of scan_cells().
struct CellScan {
  Vector<float> float_best;
  Vector<std::int8_t> int8_best;
  Vector<std::int32_t> cells;

  template <typename T>
  auto best() -> Vector<T>& {
    if constexpr (std::is_same_v<T, float>)
      return float_best;
    else
      return int8_best;
  }
};

// Cells (y * cols + x) whose largest raw score over channels [first, first +
// count) is at least `bound`, in row-major order. With NCHW heads the class
// planes are read one after another into a running maximum, a sequential
// pass the compiler vectorizes, instead of striding every plane for every
// cell; int8 scores are compared without dequantizing.
template <typename T, typename Bound>
auto scan_cells(const HeadGrid<T>& grid, std::size_t first, std::size_t count, Bound bound, CellScan& scan)
    -> const Vector<std::int32_t>& {
  const auto rows = grid.rows();
  const auto cols = grid.cols();
  auto& best = scan.best<T>();
  best.assign(static_cast<std::size_t>(rows * cols), std::numeric_limits<T>::lowest());
  if (grid.col_step() == 1) {
    for (std::size_t c = first; c < first + count; ++c) {
      for (std::int64_t y = 0; y < rows; ++y) {
        const auto* in = grid.row(y, c);
        auto* out = best.data() + y * cols;
        for (std::int64_t x = 0; x < cols; ++x)
          out[x] = std::max(out[x], in[x]);
      }
    }
  } else {
    for (std::int64_t y = 0; y < rows; ++y) {
      for (std::int64_t x = 0; x < cols; ++x) {
        auto value = std::numeric_limits<T>::lowest();
        for (std::size_t c = first; c < first + count; ++c)
          value = std::max(value, grid.raw(y, x, c));
        best[static_cast<std::size_t>(y * cols + x)] = value;
      }
    }
  }
  scan.cells.clear();
  for (std::size_t i = 0; i < best.size(); ++i) {
    if (best[i] >= bound)
      scan.cells.push_back(static_cast<std::int32_t>(i));
  }
  return scan.cells;
}

// Appends, in order, the cells x of row 0 whose `channel` value exceeds
// `threshold`. int8 values are compared as integers against the bound and
// dequantized only when they reach it.
template <typename T>
auto cells_above(const HeadGrid<T>& grid, std::size_t channel, float threshold, Vector<std::uint32_t>& out) -> void {
  const auto* values = grid.row(0, channel);
  const auto step = grid.col_step();
  const auto bound = grid.raw_bound(threshold);
  for (std::int64_t x = 0; x < grid.cols(); ++x) {
    const auto value = values[static_cast<std::size_t>(x) * step];
    if (value >= bound && grid.at(0, x, channel) > threshold)
      out.push_back(static_cast<std::uint32_t>(x));
  }
}

// Calls `decode(grid)` with a HeadGrid of the buffer's element type; false for
// types other than float32 and int8.
template <typename Fn>
auto visit_head(const core::Buffer& buffer, std::optional<engine::DataLayout> layout, Fn&& decode) -> bool {
  switch (buffer.data_type()) {
  case core::DataType::Float32: {
    HeadGrid<float> grid(buffer, layout);
    return grid.valid() && (decode(grid), true);
  }
  case core::DataType::Int8: {
    HeadGrid<std::int8_t> grid(buffer, layout);
    return grid.valid() && (decode(grid), true);
  }
  default:
    return false;
  }
}

// Expected value of the softmax over `bins` consecutive channels starting at
// `first` (YOLOv8 distribution focal loss decoding).
template <typename Grid>
auto dfl(const Grid& grid, std::int64_t y, std::int64_t x, std::size_t first, std::size_t bins) -> float {
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

// Contiguous float32 view of an output: the output itself, or `scratch`
// holding a dequantized or packed copy. Null for other data types.
auto float_data(const core::Buffer& output, core::Buffer& scratch) -> const float*;

auto nms_keep(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t>;

auto to_source(const core::Rect2f& rect, const preprocess::ImageTransform& transform) -> core::Rect2f;

auto to_source(const core::Point2f& point, const preprocess::ImageTransform& transform) -> core::Point2f;

// to_source() clipped to the source region; empty when the box leaves it.
auto clip_to_source(const core::Rect2f& rect, const preprocess::ImageTransform& transform) -> core::Rect2f;

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
