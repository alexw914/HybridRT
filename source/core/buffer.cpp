// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/buffer.cpp
// Description: Implements typed buffers over memory.
// -----------------------------------------------------------------------------

#include <hrt/core/buffer.hpp>

namespace hrt::core {

Buffer::Buffer(MemoryType memory_type, DataType data_type, const Shape& shape, const Stride& stride)
    : Buffer(SharedPtr<Memory>(Memory::create(memory_type)), data_type, shape, stride) {}

Buffer::Buffer(SharedPtr<Memory> memory, DataType data_type, const Shape& shape, const Stride& stride)
    : memory_(std::move(memory)) {
  if (memory_ != nullptr)
    valid_ = reshape(data_type, shape, stride);
}

Buffer::Buffer(const BufferProto& proto) : Buffer(proto.memory_type, proto.data_type, proto.shape, proto.stride) {}

Buffer::Buffer(SharedPtr<Memory> memory, const BufferProto& proto)
    : Buffer(std::move(memory), proto.data_type, proto.shape, proto.stride) {}

auto Buffer::proto() const -> BufferProto {
  return {memory_type(), data_type_, shape_, stride()};
}

auto Buffer::reshape(const Shape& shape, const Stride& stride) -> bool {
  return reshape(data_type_, shape, stride);
}

auto Buffer::reshape(DataType data_type, const Shape& shape, const Stride& stride) -> bool {
  auto new_shape = shape.empty() ? Shape{0} : shape;
  auto byte_shape = new_shape;
  byte_shape.back() *= static_cast<std::int64_t>(data_type_size(data_type));
  if (!memory_->resize(byte_shape, stride))
    return false;
  data_type_ = data_type;
  shape_ = std::move(new_shape);
  return true;
}

auto Buffer::offset(std::initializer_list<std::int64_t> index) const -> std::size_t {
  const auto& steps = stride();
  std::size_t result = 0;
  std::size_t axis = 0;
  for (auto value : index) {
    const auto step = axis + 1 == dims() ? item_size() : steps[axis + 1];
    result += static_cast<std::size_t>(value) * step;
    ++axis;
  }
  return result;
}

auto Buffer::copy(Buffer& dst, Buffer& src, CopyPolicy policy, Stream* stream) -> bool {
  switch (policy) {
  case CopyPolicy::KeepSrcStride:
    if (!dst.reshape(src.data_type(), src.shape(), src.stride()))
      return false;
    break;
  case CopyPolicy::AutoDstStride:
    if (!dst.reshape(src.data_type(), src.shape()))
      return false;
    break;
  case CopyPolicy::DirectCopy:
    break;
  }
  return Memory::copy(*dst.memory_, *src.memory_, CopyPolicy::DirectCopy, stream);
}

} // namespace hrt::core
