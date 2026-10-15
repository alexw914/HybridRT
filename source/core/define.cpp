// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/define.cpp
// Description: Implements data type, shape and geometry helpers.
// -----------------------------------------------------------------------------

#include <hrt/core/define.hpp>

#include <algorithm>
#include <cmath>

namespace hrt::core {

auto data_type_size(DataType type) -> std::size_t {
  switch (type) {
  case DataType::Float32:
  case DataType::Int32:
    return 4;
  case DataType::Float16:
    return 2;
  case DataType::Int64:
    return 8;
  case DataType::Int8:
  case DataType::Uint8:
  case DataType::Bool:
    return 1;
  case DataType::Unknown:
    break;
  }
  return 0;
}

auto to_string(DataType type) -> std::string_view {
  switch (type) {
  case DataType::Float32:
    return "float32";
  case DataType::Float16:
    return "float16";
  case DataType::Int64:
    return "int64";
  case DataType::Int32:
    return "int32";
  case DataType::Int8:
    return "int8";
  case DataType::Uint8:
    return "uint8";
  case DataType::Bool:
    return "bool";
  case DataType::Unknown:
    break;
  }
  return "unknown";
}

auto is_dynamic(const Shape& shape) -> bool {
  return std::any_of(shape.begin(), shape.end(), [](std::int64_t dim) -> bool { return dim < 0; });
}

auto volume(const Shape& shape) -> std::int64_t {
  if (is_dynamic(shape))
    return -1;
  std::int64_t count = 1;
  for (auto dim : shape)
    count *= dim;
  return count;
}

auto contiguous_stride(const Shape& shape, DataType type) -> Stride {
  Stride stride(shape.size());
  std::size_t step = data_type_size(type);
  for (auto i = shape.size(); i > 0; --i) {
    step *= static_cast<std::size_t>(shape[i - 1]);
    stride[i - 1] = step;
  }
  return stride;
}

auto intersect(const Rect& a, const Rect& b) -> Rect {
  const auto x1 = std::max(a.x, b.x);
  const auto y1 = std::max(a.y, b.y);
  const auto x2 = std::min(a.x + a.width, b.x + b.width);
  const auto y2 = std::min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1)
    return {};
  return {x1, y1, x2 - x1, y2 - y1};
}

auto Rect2f::from_points(Point2f a, Point2f b) -> Rect2f {
  const auto x = std::min(a.x, b.x);
  const auto y = std::min(a.y, b.y);
  return {x, y, std::max(a.x, b.x) - x, std::max(a.y, b.y) - y};
}

auto intersect(const Rect2f& a, const Rect2f& b) -> Rect2f {
  const auto x1 = std::max(a.x, b.x);
  const auto y1 = std::max(a.y, b.y);
  const auto x2 = std::min(a.x + a.width, b.x + b.width);
  const auto y2 = std::min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1)
    return {};
  return {x1, y1, x2 - x1, y2 - y1};
}

auto pad_scale(const Rect2f& rect, float scale_w, float scale_h, const Rect& bounds) -> Rect {
  const auto width = static_cast<double>(rect.width);
  const auto height = static_cast<double>(rect.height);
  const auto cx = static_cast<double>(rect.x) + (width / 2.0);
  const auto cy = static_cast<double>(rect.y) + (height / 2.0);
  const auto half_w = width * static_cast<double>(scale_w) / 2.0;
  const auto half_h = height * static_cast<double>(scale_h) / 2.0;
  const auto x1 = static_cast<int>(std::floor(cx - half_w));
  const auto y1 = static_cast<int>(std::floor(cy - half_h));
  const auto x2 = static_cast<int>(std::ceil(cx + half_w));
  const auto y2 = static_cast<int>(std::ceil(cy + half_h));
  return intersect({x1, y1, x2 - x1, y2 - y1}, bounds);
}

} // namespace hrt::core
