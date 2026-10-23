// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/buffer.hpp
// Description: Declares typed tensors over memory, used for engine IO and
//              images.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_BUFFER_HPP
#define HRT_CORE_BUFFER_HPP

#include <hrt/core/memory.hpp>

#include <kitzoo/core/macro.hpp>

#include <initializer_list>

namespace hrt::core {

struct BufferProto {
  MemoryType memory_type = MemoryType::Unknown;
  DataType data_type = DataType::Unknown;
  Shape shape;
  Stride stride;
};

// Element shape and type over a byte-shaped Memory: the memory's last
// dimension is the buffer's last dimension times the element size.
class Buffer {
public:
  explicit Buffer(MemoryType memory_type, DataType data_type = DataType::Uint8, const Shape& shape = {},
                  const Stride& stride = {});

  explicit Buffer(SharedPtr<Memory> memory, DataType data_type = DataType::Uint8, const Shape& shape = {},
                  const Stride& stride = {});

  explicit Buffer(const BufferProto& proto);

  Buffer(SharedPtr<Memory> memory, const BufferProto& proto);

  KZ_NODISCARD auto valid() const -> bool { return memory_ != nullptr && valid_; }

  KZ_NODISCARD auto memory() const -> const SharedPtr<Memory>& { return memory_; }

  KZ_NODISCARD auto memory_type() const -> MemoryType { return memory_->memory_type(); }

  KZ_NODISCARD auto data_type() const -> DataType { return data_type_; }

  KZ_NODISCARD auto item_size() const -> std::size_t { return data_type_size(data_type_); }

  KZ_NODISCARD auto dims() const -> std::size_t { return shape_.size(); }

  KZ_NODISCARD auto shape() const -> const Shape& { return shape_; }

  KZ_NODISCARD auto volume() const -> std::int64_t { return hrt::core::volume(shape_); }

  KZ_NODISCARD auto stride() const -> const Stride& { return memory_->stride(); }

  KZ_NODISCARD auto byte_size() const -> std::size_t { return memory_->bytes(); }

  KZ_NODISCARD auto is_contiguous() const -> bool { return memory_->is_contiguous(); }

  KZ_NODISCARD auto proto() const -> BufferProto;

  auto reshape(const Shape& shape, const Stride& stride = {}) -> bool;

  auto reshape(DataType data_type, const Shape& shape, const Stride& stride = {}) -> bool;

  // Byte offset of the element at the leading `index` coordinates.
  KZ_NODISCARD auto offset(std::initializer_list<std::int64_t> index) const -> std::size_t;

  static auto copy(Buffer& dst, Buffer& src, CopyPolicy policy, Stream* stream = nullptr) -> bool;

  // Affine quantization of the elements: real = scale * (q - zero_point).
  KZ_NODISCARD auto scale() const -> float { return scale_; }

  KZ_NODISCARD auto zero_point() const -> std::int32_t { return zero_point_; }

  auto set_quantization(float scale, std::int32_t zero_point) -> void {
    scale_ = scale;
    zero_point_ = zero_point;
  }

private:
  SharedPtr<Memory> memory_;
  DataType data_type_ = DataType::Unknown;
  Shape shape_;
  float scale_ = 1.0F;
  std::int32_t zero_point_ = 0;
  bool valid_ = false;
};

using Buffers = Vector<SharedPtr<Buffer>>;

} // namespace hrt::core

#endif // HRT_CORE_BUFFER_HPP
