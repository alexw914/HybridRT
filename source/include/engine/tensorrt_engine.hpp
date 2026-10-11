// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/tensorrt_engine.hpp
// Description: Declares the TensorRT backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_TENSORRT_ENGINE_HPP
#define HRT_ENGINE_TENSORRT_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <NvInfer.h>

namespace hrt::engine {

struct TrtDeleter {
  template <typename T>
  auto operator()(T* object) const -> void {
#if HRT_TENSORRT_VERSION >= 80000
    delete object;
#else
    if (object != nullptr)
      object->destroy();
#endif
  }
};

template <typename T>
using TrtPtr = UniquePtr<T, TrtDeleter>;

// EngineParam.data is a serialized engine for the target GPU and TensorRT
// version. Config: {"plugin_libraries": [...]}, loaded before deserializing.
// CUDA buffers bind directly; host buffers are staged through device copies.
class TensorrtEngine final : public Engine {
public:
  TensorrtEngine() = default;

  ~TensorrtEngine() override;

  TensorrtEngine(const TensorrtEngine&) = delete;
  auto operator=(const TensorrtEngine&) -> TensorrtEngine& = delete;
  TensorrtEngine(TensorrtEngine&&) = delete;
  auto operator=(TensorrtEngine&&) -> TensorrtEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  // Points TensorRT at `buffer` or, for host buffers, at a device staging copy.
  auto bind(std::size_t index, bool input, const SharedPtr<core::Buffer>& buffer) -> bool;

  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
#if HRT_TENSORRT_VERSION < 80500
  Vector<int> input_bindings_;
  Vector<int> output_bindings_;
  Vector<void*> bindings_;
#endif
  // Device copies of host-bound buffers; null when bound directly.
  core::Buffers input_staging_;
  core::Buffers output_staging_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_TENSORRT_ENGINE_HPP
