// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/image.hpp
// Description: Declares images as formatted buffers.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_IMAGE_HPP
#define HRT_CORE_IMAGE_HPP

#include <hrt/core/buffer.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>

#include <optional>

namespace hrt::core {

enum class ImageFormat {
  Unknown,
  InterBgr,
  InterRgb,
  PlanarBgr,
  PlanarRgb,
  Nv12,
  Nv21,
  Yu12,
  Yv12,
  InterRgba,
  Gray,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ImageFormat, {
                                              {ImageFormat::Unknown, nullptr},
                                              {ImageFormat::InterBgr, "inter_bgr"},
                                              {ImageFormat::InterRgb, "inter_rgb"},
                                              {ImageFormat::PlanarBgr, "planar_bgr"},
                                              {ImageFormat::PlanarRgb, "planar_rgb"},
                                              {ImageFormat::Nv12, "nv12"},
                                              {ImageFormat::Nv21, "nv21"},
                                              {ImageFormat::Yu12, "yu12"},
                                              {ImageFormat::Yv12, "yv12"},
                                              {ImageFormat::InterRgba, "inter_rgba"},
                                              {ImageFormat::Gray, "gray"},
                                          })

// Planes are laid out as: interleaved {h, w, c}, planar {3, h, w}, gray
// {1, h, w}, and YUV 4:2:0 {h * 3 / 2, w, 1}.
class Image {
public:
  // Views caller-owned pixels without copying: `data` is a host pointer for
  // MemoryType::Cpu, the dma-buf fd cast to a pointer for MemoryType::DmaBuffer,
  // and a device address otherwise. A zero `row_pitch` means packed rows.
  // Empty when the memory type cannot be wrapped.
  static auto wrap(MemoryType memory_type, void* data, ImageFormat format, Size size,
                   DataType data_type = DataType::Uint8, std::size_t row_pitch = 0) -> std::optional<Image>;

  Image(MemoryType memory_type, ImageFormat format, DataType data_type = DataType::Uint8, Size size = {},
        const Stride& stride = {});
  Image(SharedPtr<Memory> memory, ImageFormat format, DataType data_type = DataType::Uint8, Size size = {},
        const Stride& stride = {});
  Image(SharedPtr<Buffer> buffer, ImageFormat format, DataType data_type = DataType::Uint8, Size size = {},
        const Stride& stride = {});

  // Unknown data_type keeps the current one.
  auto reshape(Size size, DataType data_type = DataType::Unknown, const Stride& stride = {}) -> bool;
  auto reshape(ImageFormat format, Size size, DataType data_type = DataType::Unknown, const Stride& stride = {})
      -> bool;

  KZ_NODISCARD auto valid() const -> bool { return valid_ && buffer_->valid(); }

  KZ_NODISCARD auto size() const -> Size { return size_; }

  KZ_NODISCARD auto rect() const -> Rect { return {0, 0, size_.width, size_.height}; }

  KZ_NODISCARD auto format() const -> ImageFormat { return format_; }

  KZ_NODISCARD auto data_type() const -> DataType { return buffer_->data_type(); }

  KZ_NODISCARD auto data_type_size() const -> std::size_t { return buffer_->data_type_size(); }

  KZ_NODISCARD auto stride() const -> const Stride& { return buffer_->stride(); }

  KZ_NODISCARD auto buffer() const -> const SharedPtr<Buffer>& { return buffer_; }

  KZ_NODISCARD auto memory() const -> const SharedPtr<Memory>& { return buffer_->memory(); }

private:
  ImageFormat format_ = ImageFormat::Unknown;
  Size size_;
  SharedPtr<Buffer> buffer_;
  bool valid_ = false;
};

} // namespace hrt::core

#endif // HRT_CORE_IMAGE_HPP
