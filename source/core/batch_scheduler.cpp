// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/batch_scheduler.cpp
// Description: Implements views of single samples of batched buffers.
// -----------------------------------------------------------------------------

#include "include/core/batch_scheduler.hpp"

#include <kitzoo/log/logger.hpp>

namespace hrt::core {

auto batch_sample(const Buffer& buffer, std::size_t index, bool write) -> SharedPtr<Buffer> {
  const auto& shape = buffer.shape();
  KZ_CHECK_RETURN(!shape.empty() && shape[0] > 0 && index < static_cast<std::size_t>(shape[0]), nullptr,
                  "batch: sample {} is outside a batch of shape {}", index, shape.empty() ? 0 : shape[0]);

  // stride[0] spans the whole batch and stride[1] one sample; a 1-D buffer
  // holds one element per sample.
  auto& memory = *buffer.memory();
  const auto batch = static_cast<std::size_t>(shape[0]);
  auto stride = buffer.stride();
  const auto sample_bytes = stride.size() > 1 ? stride[1] : stride[0] / batch;
  const auto offset = index * sample_bytes;
  stride[0] = sample_bytes;
  auto byte_shape = memory.shape();
  byte_shape[0] = 1;

  UniquePtr<Memory> view;
  if (memory.have_cpu()) {
    auto* host = static_cast<std::byte*>(write ? memory.mutable_cpu() : const_cast<void*>(memory.cpu()));
    if (host != nullptr)
      view = Memory::wrap(MemoryType::Cpu, host + offset, 0, sample_bytes, byte_shape, stride);
  } else {
    const auto device = write ? memory.mutable_device() : memory.device();
    if (device != 0)
      view = Memory::wrap(memory.memory_type(), nullptr, device + offset, sample_bytes, byte_shape, stride);
  }
  KZ_CHECK_RETURN(view != nullptr, nullptr, "batch: cannot view sample {} of {} memory", index,
                  to_string(memory.memory_type()));

  auto sample_shape = shape;
  sample_shape[0] = 1;
  auto sample = std::make_shared<Buffer>(SharedPtr<Memory>(std::move(view)), buffer.data_type(), sample_shape, stride);
  KZ_CHECK_RETURN(sample->valid(), nullptr, "batch: cannot view sample {}", index);
  sample->set_quantization(buffer.scale(), buffer.zero_point());
  return sample;
}

} // namespace hrt::core
