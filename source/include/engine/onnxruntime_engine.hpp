// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/onnxruntime_engine.hpp
// Description: Declares the ONNX Runtime backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_ONNXRUNTIME_ENGINE_HPP
#define HRT_ENGINE_ONNXRUNTIME_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <onnxruntime_c_api.h>

namespace hrt::engine {

// C API (no exceptions); host buffers are bound once through an IoBinding.
// Config (all optional):
//   "intra_op_num_threads", "inter_op_num_threads": 0 keeps the default
//   "execution_provider": "cpu" (default) | "coreml", falling back to the CPU
//   "provider_options": CoreML only: "ModelFormat": "NeuralNetwork" |
//       "MLProgram", "MLComputeUnits": "ALL" | "CPUOnly" | "CPUAndGPU",
//       "RequireStaticInputShapes", "EnableOnSubgraphs": "0" | "1"
//   "custom_op_libraries": paths for RegisterCustomOpsLibrary_V2
//   "input_shapes": {"<input>": [dims]} fixing dynamic dims at load; without
//       it only a dynamic leading batch dim is fixed, to 1
class OrtEngine final : public Engine {
public:
  OrtEngine();

  ~OrtEngine() override;

  OrtEngine(const OrtEngine&) = delete;
  auto operator=(const OrtEngine&) -> OrtEngine& = delete;
  OrtEngine(OrtEngine&&) = delete;
  auto operator=(OrtEngine&&) -> OrtEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  auto read_io(bool input) -> bool;

  auto resolve_outputs() -> bool;

  auto bind(const core::Buffers& buffers, bool input) -> bool;

  auto release_values(Vector<OrtValue*>& values) -> void;

  // Message of a failed call, releasing its status; empty on success.
  KZ_NODISCARD auto error(OrtStatus* status) const -> String;

  const OrtApi* api_;
  OrtEnv* env_ = nullptr;
  OrtSession* session_ = nullptr;
  OrtMemoryInfo* memory_info_ = nullptr;
  OrtIoBinding* binding_ = nullptr;
  Vector<OrtValue*> input_values_;
  Vector<OrtValue*> output_values_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_ONNXRUNTIME_ENGINE_HPP
