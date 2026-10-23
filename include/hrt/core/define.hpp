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

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

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

enum class MemoryType {
  Unknown,
  Cpu,
  Cuda,
  DmaBuffer, // Linux dma-buf mapped to the host; device() is the fd, shared with media devices and NPUs.
  Acl,       // Ascend device memory from aclrtMalloc, for engine IO.
  Dvpp,      // Ascend DVPP media memory, required by VPC before A2; rows are padded to 16 by default.
};

auto data_type_size(DataType type) -> std::size_t;

// Names for logs and diagnostics.
auto to_string(DataType type) -> std::string_view;
auto to_string(MemoryType type) -> std::string_view;

// Dimensions, outermost first; negative values mark dynamic dimensions.
using Shape = Vector<std::int64_t>;

// Byte distance between consecutive elements of each dimension of a byte
// shape; stride[0] is the total allocation size.
using Stride = Vector<std::size_t>;

auto is_dynamic(const Shape& shape) -> bool;

auto volume(const Shape& shape) -> std::int64_t;

auto contiguous_stride(const Shape& shape, DataType type = DataType::Uint8) -> Stride;

//////////////////////////////// Geometry ////////////////////////////////

template <typename T>
struct BasicPoint {
  T x{};
  T y{};

  auto operator==(const BasicPoint&) const -> bool = default;
};

template <typename T>
struct BasicSize {
  T width{};
  T height{};

  auto operator==(const BasicSize&) const -> bool = default;

  KZ_NODISCARD constexpr auto area() const -> T { return width * height; }

  KZ_NODISCARD constexpr auto empty() const -> bool { return width <= 0 || height <= 0; }
};

// Top-left corner and size.
template <typename T>
struct BasicRect {
  T x{};
  T y{};
  T width{};
  T height{};

  constexpr BasicRect() = default;

  constexpr BasicRect(T left, T top, T w, T h) : x(left), y(top), width(w), height(h) {}

  // Spans two opposite corners given in any order.
  constexpr BasicRect(BasicPoint<T> a, BasicPoint<T> b)
      : x(std::min(a.x, b.x)), y(std::min(a.y, b.y)), width(std::max(a.x, b.x) - x), height(std::max(a.y, b.y) - y) {}

  // Element-wise static_cast; float to int truncates.
  template <typename U>
  constexpr explicit BasicRect(const BasicRect<U>& other)
      : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), width(static_cast<T>(other.width)),
        height(static_cast<T>(other.height)) {}

  auto operator==(const BasicRect&) const -> bool = default;

  KZ_NODISCARD constexpr auto tl() const -> BasicPoint<T> { return {x, y}; }

  KZ_NODISCARD constexpr auto br() const -> BasicPoint<T> { return {x + width, y + height}; }

  KZ_NODISCARD constexpr auto size() const -> BasicSize<T> { return {width, height}; }

  KZ_NODISCARD constexpr auto area() const -> T { return width * height; }

  KZ_NODISCARD constexpr auto empty() const -> bool { return width <= 0 || height <= 0; }
};

using Point2f = BasicPoint<float>;
using Size = BasicSize<int>;
using Size2f = BasicSize<float>;
using Rect = BasicRect<int>;
using Rect2f = BasicRect<float>;
using Rect2d = BasicRect<double>;

// Overlap of two rects; empty when they do not overlap.
template <typename T>
constexpr auto intersect(const BasicRect<T>& a, const BasicRect<T>& b) -> BasicRect<T> {
  const auto x1 = std::max(a.x, b.x);
  const auto y1 = std::max(a.y, b.y);
  const auto x2 = std::min(a.x + a.width, b.x + b.width);
  const auto y2 = std::min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1)
    return {};
  return {x1, y1, x2 - x1, y2 - y1};
}

// Intersection over union.
template <typename T>
constexpr auto iou(const BasicRect<T>& a, const BasicRect<T>& b) -> float {
  const auto inter = static_cast<float>(intersect(a, b).area());
  return inter / (static_cast<float>(a.area()) + static_cast<float>(b.area()) - inter);
}

// Rect of a center x, center y, width, height box.
template <typename T>
constexpr auto center_box(T cx, T cy, T w, T h) -> BasicRect<T> {
  return {cx - w / 2, cy - h / 2, w, h};
}

// Positive size and a finite corner; false for NaN or infinite values.
template <typename T>
auto valid_box(const BasicRect<T>& rect) -> bool {
  return rect.width > 0 && rect.height > 0 && std::isfinite(rect.x + rect.y);
}

// Grows the shorter side about the center so width / height equals `aspect`.
template <std::floating_point T>
constexpr auto pad_aspect_ratio(const BasicRect<T>& rect, std::type_identity_t<T> aspect) -> BasicRect<T> {
  auto width = rect.width;
  auto height = rect.height;
  if (width > aspect * height)
    height = width / aspect;
  else
    width = height * aspect;
  return {rect.x + (rect.width - width) / 2, rect.y + (rect.height - height) / 2, width, height};
}

// Scales width and height about the center.
template <std::floating_point T>
constexpr auto pad_scale(const BasicRect<T>& rect, std::type_identity_t<T> scale_w, std::type_identity_t<T> scale_h)
    -> BasicRect<T> {
  const auto width = rect.width * scale_w;
  const auto height = rect.height * scale_h;
  return {rect.x + (rect.width - width) / 2, rect.y + (rect.height - height) / 2, width, height};
}

template <std::floating_point T>
constexpr auto pad_scale(const BasicRect<T>& rect, std::type_identity_t<T> scale) -> BasicRect<T> {
  return pad_scale(rect, scale, scale);
}

// Smallest rect of whole pixels covering `rect`.
template <std::floating_point T>
auto cover_pixels(const BasicRect<T>& rect) -> Rect {
  const auto x1 = static_cast<int>(std::floor(rect.x));
  const auto y1 = static_cast<int>(std::floor(rect.y));
  return {x1, y1, static_cast<int>(std::ceil(rect.x + rect.width)) - x1,
          static_cast<int>(std::ceil(rect.y + rect.height)) - y1};
}

// A cascade grows a detection for the next stage with
// intersect(cover_pixels(pad_scale(box, scale_w, scale_h)), image.rect()): the
// whole pixels it covers inside the image, a valid src_rect for any
// resize_type, or empty when the box misses the image.

// Angle in degrees, clockwise in image coordinates (y down).
struct RotatedRect {
  Point2f center;
  Size2f size;
  float angle = 0.0F;
};

} // namespace hrt::core

#endif // HRT_CORE_DEFINE_HPP
