// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/rknn_engine.hpp
// Description: Declares the Rockchip RKNN backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_RKNN_ENGINE_HPP
#define HRT_ENGINE_RKNN_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <functional>
#include <rknn_api.h>

namespace hrt::engine {

// EngineParam.data holds the .rknn model. Config:
//   "image_input": true    uint8 inputs (false: int8/float inputs)
//   "core_mask": "RKNN_NPU_CORE_AUTO" | "RKNN_NPU_CORE_0" | ...
//   "custom_op_libraries": ["libops.so"]  each exporting
//       extern "C" int hrt_rknn_register_custom_ops(rknn_context)
// Inputs are NHWC in RkFd memory (float inputs in host memory); outputs are
// NCHW host buffers, int8 ones carrying their quantization.
class RknnEngine final : public Engine {
public:
  RknnEngine() = default;

  ~RknnEngine() override;

  RknnEngine(const RknnEngine&) = delete;
  auto operator=(const RknnEngine&) -> RknnEngine& = delete;
  RknnEngine(RknnEngine&&) = delete;
  auto operator=(RknnEngine&&) -> RknnEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  using TensorMem = UniquePtr<rknn_tensor_mem, std::function<void(rknn_tensor_mem*)>>;

  auto make_mem(rknn_tensor_mem* mem) -> TensorMem;

  rknn_context context_ = 0;
  // Attributes and NPU memory in name order, matching the IO protos.
  Vector<rknn_tensor_attr> input_attrs_;
  Vector<rknn_tensor_attr> output_attrs_;
  Vector<TensorMem> input_mems_;
  Vector<TensorMem> output_mems_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_RKNN_ENGINE_HPP
