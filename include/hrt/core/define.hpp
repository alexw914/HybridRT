// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/define.hpp
// Description: Declares element types, shapes, strides and 2D geometry shared
//              by memory, buffers, images and engines.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_DEFINE_HPP
#define HRT_CORE_DEFINE_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/core/macro.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hrt::core {

enum class DataType {
  Unknown,
  Float32,
  Float16,
  Int64,
  Int32,
  Int8,
  Uint8,
  Bool,
};

auto data_type_size(DataType type) -> std::size_t;

auto to_string(DataType type) -> std::string_view;

// Dimensions, outermost first; negative values mark dynamic dimensions.
using Shape = Vector<std::int64_t>;

// Byte distance between consecutive elements of each dimension of a byte
// shape; stride[0] is the total allocation size.
using Stride = Vector<std::size_t>;

auto is_dynamic(const Shape& shape) -> bool;

auto volume(const Shape& shape) -> std::int64_t;

auto contiguous_stride(const Shape& shape, DataType type = DataType::Uint8) -> Stride;

struct Size {
  int width = 0;
  int height = 0;

  auto operator==(const Size&) const -> bool = default;
};

struct Rect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;

  auto operator==(const Rect&) const -> bool = default;

  KZ_NODISCARD auto size() const -> Size { return {width, height}; }

  KZ_NODISCARD auto empty() const -> bool { return width <= 0 || height <= 0; }
};

auto intersect(const Rect& a, const Rect& b) -> Rect;

struct Point2f {
  float x = 0.0F;
  float y = 0.0F;
};

struct Size2f {
  float width = 0.0F;
  float height = 0.0F;

  KZ_NODISCARD auto area() const -> float { return width * height; }

  KZ_NODISCARD auto empty() const -> bool { return width <= 0.0F || height <= 0.0F; }
};

struct Rect2f {
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;

  static auto from_points(Point2f a, Point2f b) -> Rect2f;

  KZ_NODISCARD auto area() const -> float { return width * height; }

  KZ_NODISCARD auto empty() const -> bool { return width <= 0.0F || height <= 0.0F; }
};

auto intersect(const Rect2f& a, const Rect2f& b) -> Rect2f;

// Grows `rect` about its center by scale_w x scale_h, as a cascade grows a
// detection before the next stage, and returns the whole pixels it covers
// inside `bounds` (the image rect), a valid src_rect for any resize_type;
// empty when they do not overlap.
auto pad_scale(const Rect2f& rect, float scale_w, float scale_h, const Rect& bounds) -> Rect;

// Angle in degrees, clockwise in image coordinates (y down).
struct RotatedRect {
  Point2f center;
  Size2f size;
  float angle = 0.0F;
};

} // namespace hrt::core

#endif // HRT_CORE_DEFINE_HPP
