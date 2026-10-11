// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/ncnn_engine.hpp
// Description: Declares the ncnn backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_NCNN_ENGINE_HPP
#define HRT_ENGINE_NCNN_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <net.h>

namespace hrt::engine {

// EngineParam.proto holds the .param text and .data the .bin weights. Config:
// {"num_threads": 4}. Shapes gain a leading batch dimension of 1.
class NcnnEngine final : public Engine {
public:
  NcnnEngine() = default;

  ~NcnnEngine() override;

  NcnnEngine(const NcnnEngine&) = delete;
  auto operator=(const NcnnEngine&) -> NcnnEngine& = delete;
  NcnnEngine(NcnnEngine&&) = delete;
  auto operator=(NcnnEngine&&) -> NcnnEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  UniquePtr<ncnn::Net> net_;
  // ncnn reads weights in place, so the model bytes outlive the net.
  String param_text_;
  SharedPtr<const hrtc::Bytes> weights_;
  int num_threads_ = 4;
  Vector<int> input_indexes_;
  Vector<int> output_indexes_;
  Vector<ncnn::Mat> input_mats_;
  // Reference counts for the wrapped inputs, see set_inputs().
  UniquePtr<int[]> input_refcounts_;
  Vector<ncnn::Mat> output_mats_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_NCNN_ENGINE_HPP
