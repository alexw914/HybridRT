// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/engine/cann_engine.hpp
// Description: Declares the Ascend CANN backend.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_CANN_ENGINE_HPP
#define HRT_ENGINE_CANN_ENGINE_HPP

#include <hrt/engine/engine.hpp>

#include <acl/acl.h>

namespace hrt::engine {

// EngineParam.data holds the .om model. Inputs prefer DVPP memory, outputs ACL
// memory; contiguous Ascend buffers bind directly, others are staged through
// device copies. Call with the AclContext of the target device current.
class CannEngine final : public Engine {
public:
  CannEngine() = default;

  ~CannEngine() override;

  CannEngine(const CannEngine&) = delete;
  auto operator=(const CannEngine&) -> CannEngine& = delete;
  CannEngine(CannEngine&&) = delete;
  auto operator=(CannEngine&&) -> CannEngine& = delete;

  auto set_inputs(const core::Buffers& inputs) -> bool override;

  auto set_outputs(const core::Buffers& outputs) -> bool override;

  auto inference(core::Stream* stream) -> bool override;

protected:
  auto do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool override;

  auto do_release() -> void override;

private:
  auto make_dataset(const core::Buffers& buffers, const Vector<core::BufferProto>& protos,
                    const Vector<std::size_t>& indexes, core::Buffers& staging, aclmdlDataset*& dataset) -> bool;

  std::uint32_t model_id_ = 0;
  bool loaded_ = false;
  aclmdlDesc* desc_ = nullptr;
  aclmdlDataset* input_dataset_ = nullptr;
  aclmdlDataset* output_dataset_ = nullptr;
  // Model IO indexes in name order.
  Vector<std::size_t> input_indexes_;
  Vector<std::size_t> output_indexes_;
  core::Buffers input_staging_;
  core::Buffers output_staging_;
};

} // namespace hrt::engine

#endif // HRT_ENGINE_CANN_ENGINE_HPP
