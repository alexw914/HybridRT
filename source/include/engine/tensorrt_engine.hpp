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

// EngineParam.data is a serialized engine for the target GPU and TensorRT
// version (8.5+). Config: {"plugin_libraries": [...]}, loaded before
// deserializing; "input_shapes": {"<input>": [dims]} fixes dynamic dims within
// optimization profile 0, and without it only a dynamic leading batch dim, to 1.
// CUDA buffers bind directly; host buffers are staged through device copies,
// which run fastest from create_host_memory() (CUDA-pinned).
class TensorRTEngine final : public Engine {
public:
  TensorRTEngine() = default;

  ~TensorRTEngine() override;

  TensorRTEngine(const TensorRTEngine&) = delete;
  auto operator=(const TensorRTEngine&) -> TensorRTEngine& = delete;
  TensorRTEngine(TensorRTEngine&&) = delete;
  auto operator=(TensorRTEngine&&) -> TensorRTEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

  KZ_NODISCARD auto create_host_memory() const -> SharedPtr<core::Memory> override;

  KZ_NODISCARD auto clone() const -> UniquePtr<Engine> override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  // Points TensorRT at `buffer` or, for host buffers, at a device staging copy.
  auto bind(std::size_t index, bool input, const SharedPtr<core::Buffer>& buffer) -> bool;

  SharedPtr<nvinfer1::IRuntime> runtime_;
  SharedPtr<nvinfer1::ICudaEngine> engine_;
  UniquePtr<nvinfer1::IExecutionContext> context_;
  // Device copies of host-bound buffers; null when bound directly.
  core::Buffers input_staging_;
  core::Buffers output_staging_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_TENSORRT_ENGINE_HPP
