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
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>

#include "include/core/config.hpp"

#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define HRT_ORT_DML
#endif

namespace hrt::engine {

namespace {

// Config names and the GetAvailableProviders names, which the generic append also accepts.
constexpr Array<Pair<std::string_view, const char*>, 12> kProviders{{
    {"cpu", "CPUExecutionProvider"},
    {"coreml", "CoreMLExecutionProvider"},
    {"cuda", "CUDAExecutionProvider"},
    {"tensorrt", "TensorrtExecutionProvider"},
    {"openvino", "OpenVINOExecutionProvider"},
    {"dml", "DmlExecutionProvider"},
    {"qnn", "QNNExecutionProvider"},
    {"xnnpack", "XnnpackExecutionProvider"},
    {"cann", "CANNExecutionProvider"},
    {"dnnl", "DnnlExecutionProvider"},
    {"webgpu", "WebGpuExecutionProvider"},
    {"vitisai", "VitisAIExecutionProvider"},
}};

// Providers configured through an options object, which takes the string options before it is appended.
template <typename Options>
auto append_with_options(OrtSessionOptions* session_options, const Vector<const char*>& keys,
                         const Vector<const char*>& values, auto create, auto update, auto append, auto release)
    -> OrtStatus* {
  Options* options = nullptr;
  if (auto* status = create(&options); status != nullptr)
    return status;
  KZ_SCOPE_EXIT {
    release(options);
  };
  if (auto* status = update(options, keys.data(), values.data(), keys.size()); status != nullptr)
    return status;
  return append(session_options, options);
}

// DirectML takes only an adapter index, and needs memory patterns off and sequential execution.
auto append_dml(const OrtApi* api, KZ_MAYBE_UNUSED OrtSessionOptions* session_options,
                KZ_MAYBE_UNUSED const Map<String, String>& options) -> OrtStatus* {
#ifdef HRT_ORT_DML
  int device_id = 0;
  for (const auto& [key, value] : options) {
    const auto* end = value.data() + value.size();
    if (key != "device_id" || std::from_chars(value.data(), end, device_id).ptr != end)
      return api->CreateStatus(ORT_INVALID_ARGUMENT, ("bad provider option " + key + "=" + value).c_str());
  }
  const OrtDmlApi* dml = nullptr;
  for (auto* status :
       {api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)),
        api->DisableMemPattern(session_options), api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL)}) {
    if (status != nullptr)
      return status;
  }
  return dml->SessionOptionsAppendExecutionProvider_DML(session_options, device_id);
#else
  return api->CreateStatus(ORT_NOT_IMPLEMENTED, "built without dml_provider_factory.h");
#endif
}

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

auto OrtEngine::available(const char* provider) const -> bool {
  char** names = nullptr;
  int count = 0;
  if (auto failure = error(api_->GetAvailableProviders(&names, &count)); !failure.empty()) {
    KZ_LOG_WARN("onnxruntime: {}", failure);
    return false;
  }
  KZ_SCOPE_EXIT {
    static_cast<void>(error(api_->ReleaseAvailableProviders(names, count)));
  };
  return std::any_of(names, names + count, [&](const char* name) -> bool { return std::strcmp(name, provider) == 0; });
}

auto OrtEngine::append_provider(OrtSessionOptions* session_options, const Pair<std::string_view, const char*>& provider,
                                const Map<String, String>& options) const -> String {
  Vector<const char*> keys, values;
  for (const auto& [key, value] : options) {
    keys.push_back(key.c_str());
    values.push_back(value.c_str());
  }
  const auto& name = provider.first;
  if (name == "cuda")
    return error(append_with_options<OrtCUDAProviderOptionsV2>(
        session_options, keys, values, api_->CreateCUDAProviderOptions, api_->UpdateCUDAProviderOptions,
        api_->SessionOptionsAppendExecutionProvider_CUDA_V2, api_->ReleaseCUDAProviderOptions));
  if (name == "tensorrt")
    return error(append_with_options<OrtTensorRTProviderOptionsV2>(
        session_options, keys, values, api_->CreateTensorRTProviderOptions, api_->UpdateTensorRTProviderOptions,
        api_->SessionOptionsAppendExecutionProvider_TensorRT_V2, api_->ReleaseTensorRTProviderOptions));
  if (name == "cann")
    return error(append_with_options<OrtCANNProviderOptions>(
        session_options, keys, values, api_->CreateCANNProviderOptions, api_->UpdateCANNProviderOptions,
        api_->SessionOptionsAppendExecutionProvider_CANN, api_->ReleaseCANNProviderOptions));
  if (name == "dnnl")
    return error(append_with_options<OrtDnnlProviderOptions>(
        session_options, keys, values, api_->CreateDnnlProviderOptions, api_->UpdateDnnlProviderOptions,
        api_->SessionOptionsAppendExecutionProvider_Dnnl, api_->ReleaseDnnlProviderOptions));
  if (name == "dml")
    return error(append_dml(api_, session_options, options));
  // ORT asks for the dedicated functions of providers that have them.
  if (name == "openvino")
    return error(api_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(session_options, keys.data(), values.data(),
                                                                         keys.size()));
  if (name == "vitisai")
    return error(
        api_->SessionOptionsAppendExecutionProvider_VitisAI(session_options, keys.data(), values.data(), keys.size()));
  return error(api_->SessionOptionsAppendExecutionProvider(session_options, provider.second, keys.data(), values.data(),
                                                           keys.size()));
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
  Map<String, Vector<std::int64_t>> input_shapes;
  core::Problems problems;
  core::check_keys(config,
                   {"intra_op_num_threads", "inter_op_num_threads", "execution_provider", "provider_options",
                    "custom_op_libraries", "input_shapes"},
                   "config", problems);
  core::read(config, "intra_op_num_threads", intra_threads, "config", problems);
  core::read(config, "inter_op_num_threads", inter_threads, "config", problems);
  core::read(config, "execution_provider", provider, "config", problems);
  core::read(config, "provider_options", provider_options, "config", problems);
  core::read(config, "custom_op_libraries", custom_op_libraries, "config", problems);
  core::read(config, "input_shapes", input_shapes, "config", problems);
  const auto entry = std::find_if(kProviders.begin(), kProviders.end(),
                                  [&](const auto& item) -> bool { return item.first == provider; });
  if (entry == kProviders.end())
    problems.push_back("config.execution_provider: unknown value \"" + provider + "\"");
  else if (provider == "cpu" && !provider_options.empty())
    problems.emplace_back("config.provider_options: the CPU provider takes none");
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
    const std::filesystem::path path(library);
    if (auto* status = api_->RegisterCustomOpsLibrary_V2(session_options, path.c_str()); status != nullptr) {
      KZ_LOG_WARN("onnxruntime: skipping custom op library {}: {}", library, api_->GetErrorMessage(status));
      api_->ReleaseStatus(status);
    }
  }
  // A speed preference: a provider the runtime lacks or cannot start, such as CUDA without its
  // libraries, leaves the session on the CPU provider.
  if (provider != "cpu") {
    if (!available(entry->second))
      KZ_LOG_WARN("onnxruntime: runtime lacks {}; using the CPU provider", entry->second);
    else if (failure = append_provider(session_options, *entry, provider_options); !failure.empty())
      KZ_LOG_WARN("onnxruntime: {}: {}; using the CPU provider", entry->second, failure);
  }
  // A model that does not load fails this backend, so the next one can be tried.
  failure =
      error(api_->CreateSessionFromArray(env_, param.data->data(), param.data->size(), session_options, &session_));
  KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  failure = error(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);
  failure = error(api_->CreateIoBinding(session_, &binding_));
  KZ_CHECK_MSG(failure.empty(), "onnxruntime: {}", failure);

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  if (!read_io(true) || !read_io(false))
    return false;
  core::fix_shapes(input_shapes, input_names_, input_protos_, "config.input_shapes", problems);
  KZ_CHECK_MSG(problems.empty(), "onnxruntime: {}", core::join(problems));
  const auto dynamic = [](const core::BufferProto& proto) -> bool { return core::is_dynamic(proto.shape); };
  KZ_CHECK_RETURN(std::none_of(input_protos_.begin(), input_protos_.end(), dynamic), false,
                  "onnxruntime: dynamic input dims other than the batch need config.input_shapes");
  return std::none_of(output_protos_.begin(), output_protos_.end(), dynamic) || resolve_outputs();
}

// Output dims follow from the fixed inputs, which a run on zeroed inputs reveals.
auto OrtEngine::resolve_outputs() -> bool {
  OrtAllocator* allocator = nullptr;
  auto failure = error(api_->GetAllocatorWithDefaultOptions(&allocator));
  KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  Vector<OrtValue*> inputs(input_protos_.size(), nullptr);
  Vector<OrtValue*> outputs(output_protos_.size(), nullptr);
  KZ_SCOPE_EXIT {
    release_values(inputs);
    release_values(outputs);
  };
  Vector<const char*> input_names;
  Vector<const char*> output_names;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& proto = input_protos_[i];
    failure = error(api_->CreateTensorAsOrtValue(allocator, proto.shape.data(), proto.shape.size(),
                                                 to_ort_type(proto.data_type), &inputs[i]));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
    void* data = nullptr;
    failure = error(api_->GetTensorMutableData(inputs[i], &data));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
    std::memset(data, 0, static_cast<std::size_t>(core::volume(proto.shape)) * core::data_type_size(proto.data_type));
    input_names.push_back(input_names_[i].c_str());
  }
  for (const auto& name : output_names_)
    output_names.push_back(name.c_str());
  failure = error(api_->Run(session_, nullptr, input_names.data(), inputs.data(), inputs.size(), output_names.data(),
                            outputs.size(), outputs.data()));
  KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: resolving output shapes: {}", failure);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    failure = error(api_->GetTensorTypeAndShape(outputs[i], &info));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
    KZ_SCOPE_EXIT {
      api_->ReleaseTensorTypeAndShapeInfo(info);
    };
    auto& shape = output_protos_[i].shape;
    failure = error(api_->GetDimensions(info, shape.data(), shape.size()));
    KZ_CHECK_RETURN(failure.empty(), false, "onnxruntime: {}", failure);
  }
  return true;
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
    api_->ReleaseStatus(api_->AllocatorFree(allocator, name));
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
    // ponytail: dynamic dims are fixed once at load; varying them per run needs rebinding resized buffers.
    KZ_CHECK_RETURN(data_type.has_value(), false, "onnxruntime: tensor {} has an unsupported type", names[index]);
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
    auto failure = error(api_->CreateTensorWithDataAsOrtValue(memory_info_, data, buffers[i]->bytes(), shape.data(),
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
