// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/context.hpp
// Description: Declares devices, device contexts and execution streams.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_CONTEXT_HPP
#define HRT_CORE_CONTEXT_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/core/macro.hpp>

namespace hrt::core {

enum class DeviceType {
  Cpu,
  Cuda,
  Ascend,
};

// An ordered queue of device work, e.g. a CUDA stream. A null Stream* means
// synchronous host execution.
class Stream {
public:
  virtual ~Stream();

  Stream(const Stream&) = delete;
  auto operator=(const Stream&) -> Stream& = delete;
  Stream(Stream&&) = delete;
  auto operator=(Stream&&) -> Stream& = delete;

  virtual auto synchronize() -> void;

  KZ_NODISCARD auto handle() const -> void* { return handle_; }

protected:
  Stream() = default;

  void* handle_ = nullptr;
};

class Context {
public:
  // Returns nullptr when the device is not compiled in or not usable.
  static auto create(DeviceType device_type = DeviceType::Cpu, int device_id = 0) -> UniquePtr<Context>;

  virtual ~Context();

  Context(const Context&) = delete;
  auto operator=(const Context&) -> Context& = delete;
  Context(Context&&) = delete;
  auto operator=(Context&&) -> Context& = delete;

  virtual auto make_current() -> void = 0;

  virtual auto create_stream() -> UniquePtr<Stream> = 0;

  KZ_NODISCARD auto device_type() const -> DeviceType { return device_type_; }

  KZ_NODISCARD auto device_id() const -> int { return device_id_; }

  KZ_NODISCARD auto handle() const -> void* { return handle_; }

protected:
  Context(DeviceType device_type, int device_id) : device_type_(device_type), device_id_(device_id) {}

  DeviceType device_type_;
  int device_id_;
  void* handle_ = nullptr;
};

} // namespace hrt::core

#endif // HRT_CORE_CONTEXT_HPP
