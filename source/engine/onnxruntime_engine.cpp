// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/onnxruntime_engine.cpp
// Description: Implements the ONNX Runtime backend.
// -----------------------------------------------------------------------------

#include "include/engine/onnxruntime_engine.hpp"

#include <kitzoo/core/macro.hpp>
#include <kitzoo/core/scopeguard.hpp>
#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>

#include "include/core/config.hpp"

#if defined(__APPLE__) && __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#define HRT_ORT_COREML
#endif

namespace hrt::engine {

namespace {

#ifdef HRT_ORT_COREML
// Provider option names follow newer ONNX Runtime releases; the flags cannot
// express MLComputeUnits "CPUAndNeuralEngine".
auto coreml_flags(const Map<String, String>& options, core::Problems& problems) -> std::uint32_t {
  std::uint32_t flags = COREML_FLAG_USE_NONE;
  for (const auto& [key, value] : options) {
    const auto bad = [&] {
      problems.push_back("config.provider_options." + key + ": unknown value \"" + value + "\"");
    };
    if (key == "ModelFormat") {
      if (value == "MLProgram")
        flags |= COREML_FLAG_CREATE_MLPROGRAM;
      else if (value != "NeuralNetwork")
        bad();
    } else if (key == "MLComputeUnits") {
      if (value == "CPUOnly")
        flags |= COREML_FLAG_USE_CPU_ONLY;
      else if (value == "CPUAndGPU")
        flags |= COREML_FLAG_USE_CPU_AND_GPU;
      else if (value != "ALL")
        bad();
    } else if (key == "RequireStaticInputShapes" || key == "EnableOnSubgraphs") {
      if (value == "1")
        flags |=
            key == "EnableOnSubgraphs" ? COREML_FLAG_ENABLE_ON_SUBGRAPH : COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES;
      else if (value != "0")
        bad();
    } else {
      problems.push_back("config.provider_options." + key + ": unknown key");
    }
  }
  return flags;
}
#endif

auto to_data_type(ONNXTensorElementDataType type) -> std::optional<core::DataType> {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return core::DataType::Float32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return core::DataType::Float16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return core::DataType::Int64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return core::DataType::Int32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return core::DataType::Int8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return core::DataType::Uint8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return core::DataType::Bool;
  default:
    return std::nullopt;
  }
}

auto to_ort_type(core::DataType type) -> ONNXTensorElementDataType {
  switch (type) {
  case core::DataType::Float32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case core::DataType::Float16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case core::DataType::Int64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case core::DataType::Int32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case core::DataType::Int8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case core::DataType::Uint8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case core::DataType::Bool:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  case core::DataType::Unknown:
    break;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

} // namespace

OrtEngine::OrtEngine() : api_(OrtGetApiBase()->GetApi(ORT_API_VERSION)) {}

OrtEngine::~OrtEngine() {
  release();
}

auto OrtEngine::error(OrtStatus* status) const -> String {
  if (status == nullptr)
    return {};
  String message = api_->GetErrorMessage(status);
  api_->ReleaseStatus(status);
  return message;
}

auto OrtEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (api_ == nullptr) {
    KZ_LOG_ERROR("onnxruntime: runtime does not provide API version {}", ORT_API_VERSION);
    return false;
  }
  if (param.data == nullptr || param.data->empty()) {
    KZ_LOG_ERROR("onnxruntime: engine data (the .onnx model) is empty");
    return false;
  }
  int intra_threads = 0;
  int inter_threads = 0;
  String provider = "cpu";
  Map<String, String> provider_options;
  Vector<String> custom_op_libraries;
  core::Problems problems;
  core::check_keys(
      config,
      {"intra_op_num_threads", "inter_op_num_threads", "execution_provider", "provider_options", "custom_op_libraries"},
      "config", problems);
  core::read(config, "intra_op_num_threads", intra_threads, "config", problems);
  core::read(config, "inter_op_num_threads", inter_threads, "config", problems);
  core::read(config, "execution_provider", provider, "config", problems);
  core::read(config, "provider_options", provider_options, "config", problems);
  core::read(config, "custom_op_libraries", custom_op_libraries, "config", problems);
  KZ_MAYBE_UNUSED std::uint32_t coreml = 0;
  if (provider == "coreml") {
#ifdef HRT_ORT_COREML
    coreml = coreml_flags(provider_options, problems);
#else
    // A speed preference: packages that ask for CoreML still run elsewhere.
    KZ_LOG_WARN("onnxruntime: CoreML needs an Apple build of ONNX Runtime; using the CPU provider");
#endif
  } else if (provider != "cpu") {
    problems.push_back("config.execution_provider: unknown value \"" + provider + "\"");
  } else if (!provider_options.empty()) {
    problems.emplace_back("config.provider_options: the CPU provider takes none");
  }
  KZ_CHECK_MSG(problems.empty(), "onnxruntime: {}", core::join(problems));
  auto failure = error(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hybridrt", &env_));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  OrtSessionOptions* session_options = nullptr;
  failure = error(api_->CreateSessionOptions(&session_options));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  KZ_SCOPE_EXIT {
    api_->ReleaseSessionOptions(session_options);
  };
  for (auto* status : {api_->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL),
                       api_->SetIntraOpNumThreads(session_options, intra_threads),
                       api_->SetInterOpNumThreads(session_options, inter_threads)}) {
    failure = error(status);
    KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  }
  for (const auto& library : custom_op_libraries) {
    // Optional: a model that needs the operators fails to load without them.
    if (auto* status = api_->RegisterCustomOpsLibrary_V2(session_options, library.c_str()); status != nullptr) {
      KZ_LOG_WARN("onnxruntime: skipping custom op library {}: {}", library, api_->GetErrorMessage(status));
      api_->ReleaseStatus(status);
    }
  }
#ifdef HRT_ORT_COREML
  if (provider == "coreml") {
    failure = error(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: CoreML: {}", failure);
  }
#endif
  // A model that does not load fails this backend, so the next one can be tried.
  failure =
      error(api_->CreateSessionFromArray(env_, param.data->data(), param.data->size(), session_options, &session_));
  KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  failure = error(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  failure = error(api_->CreateIoBinding(session_, &binding_));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  return read_io(true) && read_io(false);
}

auto OrtEngine::read_io(bool input) -> bool {
  OrtAllocator* allocator = nullptr;
  std::size_t count = 0;
  // Queries on a loaded session only fail on internal errors.
  const auto ok = [this](OrtStatus* status) -> void {
    const auto failure = error(status);
    KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  };
  ok(api_->GetAllocatorWithDefaultOptions(&allocator));
  ok(input ? api_->SessionGetInputCount(session_, &count) : api_->SessionGetOutputCount(session_, &count));

  Vector<String> names(count);
  for (std::size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    ok(input ? api_->SessionGetInputName(session_, i, allocator, &name)
             : api_->SessionGetOutputName(session_, i, allocator, &name));
    names[i] = name;
    static_cast<void>(api_->AllocatorFree(allocator, name));
  }
  Vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&names](std::size_t a, std::size_t b) -> bool { return names[a] < names[b]; });

  auto& out_names = input ? input_names_ : output_names_;
  auto& out_protos = input ? input_protos_ : output_protos_;
  for (auto index : order) {
    OrtTypeInfo* type_info = nullptr;
    ok(input ? api_->SessionGetInputTypeInfo(session_, index, &type_info)
             : api_->SessionGetOutputTypeInfo(session_, index, &type_info));
    KZ_SCOPE_EXIT {
      api_->ReleaseTypeInfo(type_info);
    };

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::size_t rank = 0;
    ok(api_->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    KZ_CHECK_RETURN(tensor_info != nullptr, false, "onnxruntime: {} is not a tensor", names[index]);
    ok(api_->GetTensorElementType(tensor_info, &element_type));
    ok(api_->GetDimensionsCount(tensor_info, &rank));
    core::Shape shape(rank);
    ok(api_->GetDimensions(tensor_info, shape.data(), rank));

    auto data_type = to_data_type(element_type);
    // ponytail: static shapes only; dynamic dims need per-run reshaping of the bound buffers.
    KZ_CHECK_RETURN(data_type && !core::is_dynamic(shape), false,
                    "onnxruntime: tensor {} has an unsupported type or a dynamic shape", names[index]);
    out_names.push_back(names[index]);
    out_protos.push_back({core::MemoryType::Cpu, *data_type, shape, {}});
  }
  return true;
}

auto OrtEngine::bind(const core::Buffers& buffers, bool input) -> bool {
  auto& values = input ? input_values_ : output_values_;
  const auto& protos = input ? input_protos_ : output_protos_;
  const auto& names = input ? input_names_ : output_names_;
  input ? api_->ClearBoundInputs(binding_) : api_->ClearBoundOutputs(binding_);
  release_values(values);
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    auto* data = buffers[i]->memory()->mutable_cpu();
    KZ_CHECK_RETURN(data != nullptr && buffers[i]->is_contiguous(), false,
                    "onnxruntime: buffer {} needs contiguous host memory", i);
    OrtValue* value = nullptr;
    const auto& shape = protos[i].shape;
    auto failure = error(api_->CreateTensorWithDataAsOrtValue(memory_info_, data, buffers[i]->byte_size(), shape.data(),
                                                              shape.size(), to_ort_type(protos[i].data_type), &value));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
    values.push_back(value);
    failure = error(input ? api_->BindInput(binding_, names[i].c_str(), value)
                          : api_->BindOutput(binding_, names[i].c_str(), value));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  }
  return true;
}

auto OrtEngine::set_inputs(const core::Buffers& inputs) -> bool {
  return Engine::set_inputs(inputs) && bind(inputs_, true);
}

auto OrtEngine::set_outputs(const core::Buffers& outputs) -> bool {
  return Engine::set_outputs(outputs) && bind(outputs_, false);
}

auto OrtEngine::inference(core::Stream* /*stream*/) -> bool {
  const auto failure = error(api_->RunWithBinding(session_, nullptr, binding_));
  KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  return true;
}

auto OrtEngine::release_values(Vector<OrtValue*>& values) -> void {
  for (auto* value : values)
    api_->ReleaseValue(value);
  values.clear();
}

auto OrtEngine::do_release() -> void {
  release_values(input_values_);
  release_values(output_values_);
  if (binding_ != nullptr)
    api_->ReleaseIoBinding(binding_);
  binding_ = nullptr;
  if (memory_info_ != nullptr)
    api_->ReleaseMemoryInfo(memory_info_);
  if (session_ != nullptr)
    api_->ReleaseSession(session_);
  if (env_ != nullptr)
    api_->ReleaseEnv(env_);
  memory_info_ = nullptr;
  session_ = nullptr;
  env_ = nullptr;
}

} // namespace hrt::engine
