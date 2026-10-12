// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/engine/engine.hpp
// Description: Declares the inference engine interface implemented by each
//              backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_ENGINE_HPP
#define HRT_ENGINE_ENGINE_HPP

#include <hrt/core/buffer.hpp>
#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>

#include <string_view>

namespace hrt::engine {

enum class EngineType {
  Unknown,
  Ncnn,
  Mnn,
  Ort,
  Rknn,
  Cann,
  TensorRt,
};

auto engine_type_from(std::string_view name) -> EngineType;

auto device_type_of(EngineType type) -> core::DeviceType;

enum class DataLayout {
  Nchw,
  Nhwc,
};

struct EngineProto {
  DataLayout input_layout = DataLayout::Nchw;
  DataLayout output_layout = DataLayout::Nchw;
};

// One loaded model. Inputs and outputs are ordered by tensor name so every
// backend presents the same order to postprocessing. Not thread-safe; create
// one engine per worker.
class Engine {
public:
  static auto create(EngineType type) -> UniquePtr<Engine>;

  static auto create(std::string_view name) -> UniquePtr<Engine>;

  virtual ~Engine();

  Engine(const Engine&) = delete;
  auto operator=(const Engine&) -> Engine& = delete;
  Engine(Engine&&) = delete;
  auto operator=(Engine&&) -> Engine& = delete;

  // Releases any previous model first. Aborts when `param.config` is not a JSON
  // object or holds settings the backend does not know.
  auto initialize(const hrtc::EngineParam& param) -> bool;

  virtual auto inference(core::Stream* stream = nullptr) -> bool = 0;

  auto release() -> void;

  KZ_NODISCARD auto proto() const -> const EngineProto& { return proto_; }

  KZ_NODISCARD auto input_names() const -> const Vector<String>& { return input_names_; }

  KZ_NODISCARD auto output_names() const -> const Vector<String>& { return output_names_; }

  KZ_NODISCARD auto input_protos() const -> const Vector<core::BufferProto>& { return input_protos_; }

  KZ_NODISCARD auto output_protos() const -> const Vector<core::BufferProto>& { return output_protos_; }

  KZ_NODISCARD auto inputs() const -> const core::Buffers& { return inputs_; }

  KZ_NODISCARD auto outputs() const -> const core::Buffers& { return outputs_; }

  // Binds buffers matching the protos in count, data type and shape; backends
  // with stricter memory needs override.
  virtual auto set_inputs(const core::Buffers& inputs) -> bool;

  virtual auto set_outputs(const core::Buffers& outputs) -> bool;

  // Empty host memory that this backend copies to and from fastest (e.g.
  // CUDA-pinned), for IO the host writes or reads; null when plain host memory
  // is as good. Sized by the Buffer built over it.
  KZ_NODISCARD virtual auto create_host_memory() const -> SharedPtr<core::Memory>;

protected:
  Engine() = default;

  // `config` is `param.config` parsed, an empty object when absent.
  virtual auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool = 0;

  virtual auto do_release() -> void = 0;

  EngineProto proto_;
  Vector<String> input_names_;
  Vector<String> output_names_;
  Vector<core::BufferProto> input_protos_;
  Vector<core::BufferProto> output_protos_;
  core::Buffers inputs_;
  core::Buffers outputs_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_ENGINE_HPP
