// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/engine.cpp
// Description: Implements the engine lifecycle, IO binding and backend
//              creation.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>

#include <kitzoo/log/logger.hpp>
#include <kitzoo/utilities/str.hpp>

#ifdef HRT_USE_ORT
#include "include/engine/onnxruntime_engine.hpp"
#endif
#ifdef HRT_USE_NCNN
#include "include/engine/ncnn_engine.hpp"
#endif
#ifdef HRT_USE_MNN
#include "include/engine/mnn_engine.hpp"
#endif
#ifdef HRT_USE_TENSORRT
#include "include/engine/tensorrt_engine.hpp"
#endif
#ifdef HRT_USE_RKNN
#include "include/engine/rknn_engine.hpp"
#endif
#ifdef HRT_USE_CANN
#include "include/engine/cann_engine.hpp"
#endif

namespace hrt::engine {

namespace {

auto bind_buffers(core::Buffers& bound, const core::Buffers& buffers, const Vector<core::BufferProto>& protos,
                  std::string_view kind) -> bool {
  if (buffers.size() != protos.size()) {
    KZ_LOG_ERROR("Engine: expected {} {} buffers, got {}", protos.size(), kind, buffers.size());
    return false;
  }
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    if (buffer == nullptr || buffer->data_type() != protos[i].data_type || buffer->shape() != protos[i].shape) {
      KZ_LOG_ERROR("Engine: {} buffer {} does not match its proto", kind, i);
      return false;
    }
  }
  bound = buffers;
  return true;
}

} // namespace

auto engine_type_from(std::string_view name) -> EngineType {
  static constexpr Array<Pair<std::string_view, EngineType>, 6> kNames{{
      {"ncnn", EngineType::Ncnn},
      {"mnn", EngineType::Mnn},
      {"ort", EngineType::Ort},
      {"rknn", EngineType::Rknn},
      {"cann", EngineType::Cann},
      {"trt", EngineType::TensorRt},
  }};
  for (const auto& [key, type] : kNames) {
    if (kitzoo::util::equals_ignore_case(key, name))
      return type;
  }
  return EngineType::Unknown;
}

auto device_type_of(EngineType type) -> core::DeviceType {
  switch (type) {
  case EngineType::TensorRt:
    return core::DeviceType::Cuda;
  case EngineType::Cann:
    return core::DeviceType::Ascend;
  default:
    return core::DeviceType::Cpu;
  }
}

auto Engine::create(EngineType type) -> UniquePtr<Engine> {
  switch (type) {
#ifdef HRT_USE_ORT
  case EngineType::Ort:
    return std::make_unique<OrtEngine>();
#endif
#ifdef HRT_USE_NCNN
  case EngineType::Ncnn:
    return std::make_unique<NcnnEngine>();
#endif
#ifdef HRT_USE_MNN
  case EngineType::Mnn:
    return std::make_unique<MnnEngine>();
#endif
#ifdef HRT_USE_TENSORRT
  case EngineType::TensorRt:
    return std::make_unique<TensorrtEngine>();
#endif
#ifdef HRT_USE_RKNN
  case EngineType::Rknn:
    return std::make_unique<RknnEngine>();
#endif
#ifdef HRT_USE_CANN
  case EngineType::Cann:
    return std::make_unique<CannEngine>();
#endif
  default:
    KZ_LOG_WARN("Engine: type {} is not available in this build", static_cast<int>(type));
    return nullptr;
  }
}

auto Engine::create(std::string_view name) -> UniquePtr<Engine> {
  return create(engine_type_from(name));
}

Engine::~Engine() = default;

auto Engine::initialize(const hrtc::EngineParam& param) -> bool {
  release();
  const kitzoo::json::Reader reader{param.config.empty() ? std::string_view{"{}"} : std::string_view{param.config}};
  KZ_CHECK_MSG(reader.is_parse_success() && reader.raw().is_object(), "Engine: config of {} is not a JSON object: {}",
               param.engine_type, reader.error_info());
  if (do_initialize(param, reader.raw()))
    return true;
  release();
  return false;
}

auto Engine::release() -> void {
  do_release();
  input_names_.clear();
  output_names_.clear();
  input_protos_.clear();
  output_protos_.clear();
  inputs_.clear();
  outputs_.clear();
  proto_ = {};
}

auto Engine::set_inputs(const core::Buffers& inputs) -> bool {
  return bind_buffers(inputs_, inputs, input_protos_, "input");
}

auto Engine::set_outputs(const core::Buffers& outputs) -> bool {
  return bind_buffers(outputs_, outputs, output_protos_, "output");
}

} // namespace hrt::engine
