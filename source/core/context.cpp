// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/context.cpp
// Description: Implements the host context and context creation.
// -----------------------------------------------------------------------------

#include <hrt/core/context.hpp>

#ifdef HRT_USE_CUDA
#include "include/backend/cuda/cuda_context.hpp"
#endif
#ifdef HRT_USE_CANN
#include "include/backend/ascend/acl_context.hpp"
#endif

namespace hrt::core {

namespace {

class CpuStream final : public Stream {};

class CpuContext final : public Context {
public:
  explicit CpuContext(int device_id) : Context(DeviceType::Cpu, device_id) {}

  auto set_current() -> void override {}

  auto create_stream() -> UniquePtr<Stream> override { return std::make_unique<CpuStream>(); }
};

} // namespace

Stream::~Stream() = default;

auto Stream::synchronize() -> void {}

Context::~Context() = default;

auto Context::create(DeviceType device_type, int device_id) -> UniquePtr<Context> {
  switch (device_type) {
  case DeviceType::Cpu:
    return std::make_unique<CpuContext>(device_id);
#ifdef HRT_USE_CUDA
  case DeviceType::Cuda:
    if (!backend::CudaContext::available(device_id))
      return nullptr;
    return std::make_unique<backend::CudaContext>(device_id);
#endif
#ifdef HRT_USE_CANN
  case DeviceType::Ascend: {
    auto context = std::make_unique<backend::AclContext>(device_id);
    if (context->handle() == nullptr)
      return nullptr;
    return context;
  }
#endif
  default:
    break;
  }
  return nullptr;
}

} // namespace hrt::core
