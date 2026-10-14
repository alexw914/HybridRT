// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/tensorrt_engine.cpp
// Description: Implements the TensorRT backend.
// -----------------------------------------------------------------------------

#include "include/engine/tensorrt_engine.hpp"

#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>

#include <NvInferPlugin.h>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <type_traits>

#include "include/backend/cuda/cuda_common.hpp"
#include "include/backend/cuda/cuda_host_memory.hpp"
#include "include/core/config.hpp"

namespace hrt::engine {

namespace {

class Logger final : public nvinfer1::ILogger {
public:
  auto log(Severity severity, const char* message) noexcept -> void override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      KZ_LOG_ERROR("tensorrt: {}", message);
      break;
    case Severity::kWARNING:
      KZ_LOG_WARN("tensorrt: {}", message);
      break;
    case Severity::kINFO:
      KZ_LOG_DEBUG("tensorrt: {}", message);
      break;
    case Severity::kVERBOSE:
      KZ_LOG_TRACE("tensorrt: {}", message);
      break;
    }
  }
};

auto logger() -> Logger& {
  static Logger instance;
  return instance;
}

// Each library loads once per process and registers through
// REGISTER_TENSORRT_PLUGIN or (8.6+) exported getPluginCreators/getCreators.
// A library that fails to load is skipped; deserialization reports the gap.
auto load_plugins(const Vector<String>& libraries) -> void {
  static std::once_flag builtin;
  std::call_once(builtin, [] { initLibNvInferPlugins(&logger(), ""); });
  static std::mutex mutex;
  static Set<String> loaded;
  const std::lock_guard lock(mutex);
  for (const auto& library : libraries) {
    if (loaded.contains(library))
      continue;
    auto* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
      KZ_LOG_WARN("tensorrt: skipping plugin library {}: {}", library, dlerror());
      continue;
    }
#if HRT_TENSORRT_VERSION >= 80600
    if ((dlsym(handle, "getPluginCreators") != nullptr || dlsym(handle, "getCreators") != nullptr) &&
        getPluginRegistry()->loadLibrary(library.c_str()) == nullptr) {
      KZ_LOG_WARN("tensorrt: cannot register the plugins of {}", library);
      continue;
    }
#endif
    loaded.insert(library);
  }
}

auto data_type_of(nvinfer1::DataType type) -> std::optional<core::DataType> {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return core::DataType::Float32;
  case nvinfer1::DataType::kHALF:
    return core::DataType::Float16;
  case nvinfer1::DataType::kINT8:
    return core::DataType::Int8;
  case nvinfer1::DataType::kINT32:
    return core::DataType::Int32;
#if HRT_TENSORRT_VERSION >= 100000
  case nvinfer1::DataType::kINT64:
    return core::DataType::Int64;
#endif
  case nvinfer1::DataType::kBOOL:
    return core::DataType::Bool;
  default:
    return std::nullopt;
  }
}

auto shape_of(const nvinfer1::Dims& dims) -> core::Shape {
  return core::Shape(dims.d, dims.d + dims.nbDims);
}

} // namespace

TensorrtEngine::~TensorrtEngine() {
  release();
}

auto TensorrtEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.data == nullptr || param.data->empty()) {
    KZ_LOG_ERROR("tensorrt: engine data (the serialized engine) is empty");
    return false;
  }
  Vector<String> plugin_libraries;
  Map<String, Vector<std::int64_t>> input_shapes;
  core::Problems problems;
  core::check_keys(config, {"plugin_libraries", "input_shapes"}, "config", problems);
  core::read(config, "plugin_libraries", plugin_libraries, "config", problems);
  core::read(config, "input_shapes", input_shapes, "config", problems);
  KZ_CHECK_MSG(problems.empty(), "tensorrt: {}", core::join(problems));
  load_plugins(plugin_libraries);

  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (runtime_ == nullptr)
    return false;
  engine_.reset(runtime_->deserializeCudaEngine(param.data->data(), param.data->size()));
  if (engine_ == nullptr || (context_.reset(engine_->createExecutionContext()), context_ == nullptr)) {
    KZ_LOG_ERROR("tensorrt: failed to deserialize the engine");
    return false;
  }

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  Set<String> tensors;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    tensors.emplace(engine_->getIOTensorName(i));
  for (const auto& name : tensors) {
    const auto type = data_type_of(engine_->getTensorDataType(name.c_str()));
    if (!type) {
      KZ_LOG_ERROR("tensorrt: tensor {} has an unsupported type", name);
      return false;
    }
    const core::BufferProto proto{core::MemoryType::Cuda, *type, shape_of(engine_->getTensorShape(name.c_str())), {}};
    if (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.push_back(name);
      input_protos_.push_back(proto);
    } else {
      output_names_.push_back(name);
      output_protos_.push_back(proto);
    }
  }
  // Dynamic dims are fixed once, within optimization profile 0; outputs then follow from the inputs.
  core::fix_shapes(input_shapes, input_names_, input_protos_, "config.input_shapes", problems);
  KZ_CHECK_MSG(problems.empty(), "tensorrt: {}", core::join(problems));
  for (std::size_t i = 0; i < input_protos_.size(); ++i) {
    const auto& shape = input_protos_[i].shape;
    KZ_CHECK_RETURN(!core::is_dynamic(shape), false,
                    "tensorrt: input {} has dynamic dims other than the batch; set config.input_shapes",
                    input_names_[i]);
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t d = 0; d < shape.size(); ++d)
      dims.d[d] = static_cast<std::remove_reference_t<decltype(dims.d[0])>>(shape[d]);
    // Reports dims outside the profile through the logger.
    KZ_CHECK_RETURN(context_->setInputShape(input_names_[i].c_str(), dims), false,
                    "tensorrt: input {} shape is outside the engine's optimization profile", input_names_[i]);
  }
  for (std::size_t i = 0; i < output_protos_.size(); ++i) {
    output_protos_[i].shape = shape_of(context_->getTensorShape(output_names_[i].c_str()));
    KZ_CHECK_RETURN(!core::is_dynamic(output_protos_[i].shape), false, "tensorrt: output {} has a data-dependent shape",
                    output_names_[i]);
  }
  input_staging_.resize(input_protos_.size());
  output_staging_.resize(output_protos_.size());
  return true;
}

auto TensorrtEngine::bind(std::size_t index, bool input, const SharedPtr<core::Buffer>& buffer) -> bool {
  auto& staging = (input ? input_staging_ : output_staging_)[index];
  std::uintptr_t address = 0;
  if (buffer->memory_type() == core::MemoryType::Cuda && buffer->is_contiguous()) {
    staging.reset();
    address = buffer->memory()->device();
  } else {
    // Kept across calls: cudaFree synchronizes the whole device.
    if (staging == nullptr || !staging->valid())
      staging = std::make_shared<core::Buffer>((input ? input_protos_ : output_protos_)[index]);
    if (!staging->valid())
      return false;
    address = staging->memory()->device();
  }
  auto* pointer = reinterpret_cast<void*>(address);
  const auto& name = (input ? input_names_ : output_names_)[index];
  return context_->setTensorAddress(name.c_str(), pointer);
}

auto TensorrtEngine::create_host_memory() const -> SharedPtr<core::Memory> {
  return backend::create_cuda_host_memory();
}

auto TensorrtEngine::set_inputs(const core::Buffers& inputs) -> bool {
  if (!Engine::set_inputs(inputs))
    return false;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!bind(i, true, inputs[i]))
      return false;
  }
  return true;
}

auto TensorrtEngine::set_outputs(const core::Buffers& outputs) -> bool {
  if (!Engine::set_outputs(outputs))
    return false;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (!bind(i, false, outputs[i]))
      return false;
  }
  return true;
}

auto TensorrtEngine::inference(core::Stream* stream) -> bool {
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    if (input_staging_[i] != nullptr &&
        !core::Buffer::copy(*input_staging_[i], *inputs_[i], core::CopyPolicy::DirectCopy, stream))
      return false;
  }
  const auto ok = context_->enqueueV3(backend::cuda_stream(stream));
  if (!ok) {
    KZ_LOG_ERROR("tensorrt: enqueue failed");
    return false;
  }
  bool host_outputs = false;
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    if (output_staging_[i] == nullptr)
      continue;
    if (!core::Buffer::copy(*outputs_[i], *output_staging_[i], core::CopyPolicy::DirectCopy, stream))
      return false;
    host_outputs = true;
  }
  // Host outputs must be complete before the caller reads them.
  if (host_outputs)
    KZ_CHECK_RETURN(cudaStreamSynchronize(backend::cuda_stream(stream)) == cudaSuccess, false,
                    "tensorrt: cudaStreamSynchronize failed: {}", cudaGetErrorString(cudaGetLastError()));
  return true;
}

auto TensorrtEngine::do_release() -> void {
  input_staging_.clear();
  output_staging_.clear();
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

} // namespace hrt::engine
