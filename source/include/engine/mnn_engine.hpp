// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/mnn_engine.hpp
// Description: Declares the MNN backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_MNN_ENGINE_HPP
#define HRT_ENGINE_MNN_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <MNN/Interpreter.hpp>

namespace hrt::engine {

// EngineParam.data holds the .mnn model. Config: {"forward_type": "cpu" |
// "opencl" | ..., "num_thread": 4, "cache_file": .., "cache_dir": ..}, the
// last two for OpenCL tuning caches.
class MnnEngine final : public Engine {
public:
  MnnEngine() = default;

  ~MnnEngine() override;

  MnnEngine(const MnnEngine&) = delete;
  auto operator=(const MnnEngine&) -> MnnEngine& = delete;
  MnnEngine(MnnEngine&&) = delete;
  auto operator=(MnnEngine&&) -> MnnEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  UniquePtr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
  Vector<MNN::Tensor*> input_tensors_;
  Vector<MNN::Tensor*> output_tensors_;
  Vector<UniquePtr<MNN::Tensor>> host_inputs_;
  Vector<UniquePtr<MNN::Tensor>> host_outputs_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_MNN_ENGINE_HPP
