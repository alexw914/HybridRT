// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/image.cpp
// Description: Implements image plane layouts over buffers.
// -----------------------------------------------------------------------------

#include <hrt/core/image.hpp>

#include <kitzoo/log/logger.hpp>

#include <optional>

namespace hrt::core {

namespace {

auto image_shape(ImageFormat format, Size size) -> std::optional<Shape> {
  const std::int64_t h = size.height;
  const std::int64_t w = size.width;
  switch (format) {
  case ImageFormat::InterBgr:
  case ImageFormat::InterRgb:
    return Shape{h, w, 3};
  case ImageFormat::InterRgba:
    return Shape{h, w, 4};
  case ImageFormat::PlanarBgr:
  case ImageFormat::PlanarRgb:
    return Shape{3, h, w};
  case ImageFormat::Gray:
    return Shape{1, h, w};
  case ImageFormat::Nv12:
  case ImageFormat::Nv21:
  case ImageFormat::Yu12:
  case ImageFormat::Yv12:
    return Shape{h * 3 / 2, w, 1};
  case ImageFormat::Unknown:
    break;
  }
  return std::nullopt;
}

// Index of the row dimension in image_shape(): rows are the second dimension
// of planar layouts and the first otherwise.
auto row_axis(ImageFormat format) -> std::size_t {
  switch (format) {
  case ImageFormat::PlanarBgr:
  case ImageFormat::PlanarRgb:
  case ImageFormat::Gray:
    return 1;
  default:
    return 0;
  }
}

} // namespace

auto Image::wrap(MemoryType memory_type, void* data, ImageFormat format, Size size, DataType data_type,
                 std::size_t row_pitch) -> std::optional<Image> {
  auto shape = image_shape(format, size);
  if (!shape || data == nullptr) {
    KZ_LOG_ERROR("Image::wrap: unsupported format or null data");
    return std::nullopt;
  }
  auto byte_shape = *shape;
  byte_shape.back() *= static_cast<std::int64_t>(data_type_size(data_type));
  auto stride = contiguous_stride(byte_shape);
  if (row_pitch != 0) {
    const auto rows = row_axis(format);
    if (row_pitch < stride[rows + 1]) {
      KZ_LOG_ERROR("Image::wrap: row pitch {} is smaller than a row ({})", row_pitch, stride[rows + 1]);
      return std::nullopt;
    }
    stride[rows + 1] = row_pitch;
    for (auto axis = rows + 1; axis > 0; --axis)
      stride[axis - 1] = static_cast<std::size_t>(byte_shape[axis - 1]) * stride[axis];
  }
  auto* host = memory_type == MemoryType::Cpu ? data : nullptr;
  auto device = memory_type == MemoryType::Cpu ? std::uintptr_t{0} : reinterpret_cast<std::uintptr_t>(data);
  SharedPtr<Memory> memory = Memory::wrap(memory_type, host, device, stride.front(), byte_shape, stride);
  if (memory == nullptr)
    return std::nullopt;
  Image image(std::move(memory), format, data_type, size, stride);
  if (!image.valid())
    return std::nullopt;
  return image;
}

Image::Image(MemoryType memory_type, ImageFormat format, DataType data_type, Size size, const Stride& stride)
    : Image(SharedPtr<Memory>(Memory::create(memory_type)), format, data_type, size, stride) {}

Image::Image(SharedPtr<Memory> memory, ImageFormat format, DataType data_type, Size size, const Stride& stride)
    : format_(format), size_(size) {
  auto shape = image_shape(format, size);
  if (!shape) {
    KZ_LOG_ERROR("Image: unsupported format");
    return;
  }
  buffer_ = std::make_shared<Buffer>(std::move(memory), data_type == DataType::Unknown ? DataType::Uint8 : data_type,
                                     *shape, stride);
  valid_ = buffer_->valid();
}

Image::Image(SharedPtr<Buffer> buffer, ImageFormat format, DataType data_type, Size size, const Stride& stride)
    : buffer_(std::move(buffer)) {
  valid_ = buffer_->valid() && reshape(format, size, data_type, stride);
}

auto Image::reshape(Size size, DataType data_type, const Stride& stride) -> bool {
  return reshape(format_, size, data_type, stride);
}

auto Image::reshape(ImageFormat format, Size size, DataType data_type, const Stride& stride) -> bool {
  auto shape = image_shape(format, size);
  if (!shape) {
    KZ_LOG_ERROR("Image: unsupported format");
    return false;
  }
  if (!buffer_->reshape(data_type == DataType::Unknown ? buffer_->data_type() : data_type, *shape, stride))
    return false;
  format_ = format;
  size_ = size;
  return true;
}

} // namespace hrt::core
