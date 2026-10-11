// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/fake_engine.hpp
// Description: Declares an engine stub with fixed IO protos for postprocessor
//              tests, plus helpers to fill output buffers.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_FAKE_ENGINE_HPP
#define HRT_POSTPROCESS_FAKE_ENGINE_HPP

#include <hrt/engine/engine.hpp>

namespace hrt::test {

class FakeEngine final : public engine::Engine {
public:
  FakeEngine(hrt::Vector<core::BufferProto> inputs, hrt::Vector<core::BufferProto> outputs,
             engine::DataLayout output_layout = engine::DataLayout::Nchw) {
    input_protos_ = std::move(inputs);
    output_protos_ = std::move(outputs);
    proto_.output_layout = output_layout;
  }

  auto inference(core::Stream* /*stream*/) -> bool override { return true; }

protected:
  auto do_initialize(const hrtc::EngineParam& /*param*/, const kitzoo::json::Json& /*config*/) -> bool override {
    return true;
  }

  auto do_release() -> void override {}
};

inline auto proto(core::DataType type, core::Shape shape) -> core::BufferProto {
  return {core::MemoryType::Cpu, type, std::move(shape), {}};
}

// Host buffer holding `values` in the proto's shape.
template <typename T>
auto buffer_of(const core::BufferProto& proto, const hrt::Vector<T>& values) -> hrt::SharedPtr<core::Buffer> {
  auto buffer = std::make_shared<core::Buffer>(proto);
  auto* data = static_cast<T*>(buffer->memory()->mutable_cpu());
  std::copy(values.begin(), values.end(), data);
  return buffer;
}

inline auto identity(int width, int height) -> preprocess::ImageTransform {
  return {{0, 0, width, height}, {0, 0, width, height}, {width, height}};
}

} // namespace hrt::test

#endif // HRT_POSTPROCESS_FAKE_ENGINE_HPP
