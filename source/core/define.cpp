// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/define.cpp
// Description: Implements data type, shape and geometry helpers.
// -----------------------------------------------------------------------------

#include <hrt/core/define.hpp>

#include <algorithm>

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

auto to_string(MemoryType type) -> std::string_view {
  switch (type) {
  case MemoryType::Cpu:
    return "Cpu";
  case MemoryType::Cuda:
    return "Cuda";
  case MemoryType::DmaBuffer:
    return "DmaBuffer";
  case MemoryType::Acl:
    return "Acl";
  case MemoryType::Dvpp:
    return "Dvpp";
  case MemoryType::Unknown:
    break;
  }
  return "Unknown";
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

} // namespace hrt::core
