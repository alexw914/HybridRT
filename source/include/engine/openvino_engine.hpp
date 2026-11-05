// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/openvino_engine.hpp
// Description: Declares the OpenVINO CPU inference backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_OPENVINO_ENGINE_HPP
#define HRT_ENGINE_OPENVINO_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <openvino/c/openvino.h>

namespace hrt::engine {

// EngineParam.proto/data hold the IR XML/BIN pair. Config: {"num_threads": 0}
// uses the OpenVINO default; positive values limit CPU inference threads.
class OpenVinoEngine final : public Engine {
public:
  OpenVinoEngine() = default;

  ~OpenVinoEngine() override;

  OpenVinoEngine(const OpenVinoEngine&) = delete;
  auto operator=(const OpenVinoEngine&) -> OpenVinoEngine& = delete;
  OpenVinoEngine(OpenVinoEngine&&) = delete;
  auto operator=(OpenVinoEngine&&) -> OpenVinoEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  auto read_ports(bool input) -> bool;

  ov_core_t* core_ = nullptr;
  ov_model_t* model_ = nullptr;
  ov_tensor_t* weights_ = nullptr;
  ov_compiled_model_t* compiled_model_ = nullptr;
  ov_infer_request_t* request_ = nullptr;
  Vector<ov_tensor_t*> input_tensors_;
  Vector<ov_tensor_t*> output_tensors_;
  Vector<std::size_t> input_indexes_;
  Vector<std::size_t> output_indexes_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_OPENVINO_ENGINE_HPP
