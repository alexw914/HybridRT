// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/mnn_engine.cpp
// Description: Implements the MNN backend.
// -----------------------------------------------------------------------------

#include "include/engine/mnn_engine.hpp"

#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>

#include <optional>

#include "include/core/config.hpp"
#include "include/memory/cpu_memory.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(MNNForwardType, {
                                                 {MNN_FORWARD_CPU, "cpu"},
                                                 {MNN_FORWARD_OPENCL, "opencl"},
                                                 {MNN_FORWARD_METAL, "metal"},
                                                 {MNN_FORWARD_VULKAN, "vulkan"},
                                                 {MNN_FORWARD_CUDA, "cuda"},
                                                 {MNN_FORWARD_NN, "nn"},
                                                 {MNN_FORWARD_AUTO, "auto"},
                                             })

namespace hrt::engine {

namespace {

auto data_type_of(const MNN::Tensor& tensor) -> std::optional<core::DataType> {
  const auto type = tensor.getType();
  if (type.code == halide_type_float && type.bits == 32)
    return core::DataType::Float32;
  if (type.code == halide_type_int && type.bits == 32)
    return core::DataType::Int32;
  if (type.code == halide_type_uint && type.bits == 8)
    return core::DataType::Uint8;
  return std::nullopt;
}

auto shape_of(const MNN::Tensor& tensor) -> core::Shape {
  core::Shape shape(static_cast<std::size_t>(tensor.dimensions()));
  for (std::size_t i = 0; i < shape.size(); ++i)
    shape[i] = tensor.length(static_cast<int>(i));
  return shape;
}

} // namespace

MnnEngine::~MnnEngine() {
  release();
}

auto MnnEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.data == nullptr || param.data->empty()) {
    KZ_LOG_ERROR("mnn: engine data (the .mnn model) is empty");
    return false;
  }
  auto forward_type = MNN_FORWARD_CPU;
  int num_thread = 4;
  String cache_file;
  String cache_dir;
  core::Problems problems;
  core::check_keys(config, {"forward_type", "num_thread", "cache_file", "cache_dir"}, "config", problems);
  core::read_enum(config, "forward_type", forward_type, "config", problems);
  core::read(config, "num_thread", num_thread, "config", problems);
  core::read(config, "cache_file", cache_file, "config", problems);
  core::read(config, "cache_dir", cache_dir, "config", problems);
  KZ_CHECK_MSG(problems.empty(), "mnn: {}", core::join(problems));

  interpreter_.reset(MNN::Interpreter::createFromBuffer(param.data->data(), param.data->size()));
  if (interpreter_ == nullptr) {
    KZ_LOG_ERROR("mnn: failed to load the model");
    return false;
  }
  interpreter_->setSessionMode(MNN::Interpreter::SessionMode::Session_Release);

  MNN::ScheduleConfig schedule;
  schedule.type = forward_type;
  schedule.numThread = num_thread;
  MNN::BackendConfig backend;
  backend.precision = MNN::BackendConfig::Precision_Low;
  backend.power = MNN::BackendConfig::Power_Normal;
  backend.memory = MNN::BackendConfig::Memory_High;
  schedule.backendConfig = &backend;
  if (schedule.type == MNN_FORWARD_OPENCL) {
    int tuning = MNN_GPU_TUNING_FAST;
    if (!cache_dir.empty() && !cache_file.empty()) {
      std::error_code ec;
      kitzoo::os::make_directory(cache_dir, ec, true);
      if (ec) {
        KZ_LOG_ERROR("mnn: cannot create cache dir {}: {}", cache_dir, ec.message());
        return false;
      }
      interpreter_->setCacheFile((std::filesystem::path(cache_dir) / cache_file).string().c_str());
      tuning = MNN_GPU_TUNING_WIDE;
    }
    schedule.mode = tuning | MNN_GPU_MEMORY_BUFFER | MNN_GPU_MEMORY_IMAGE;
  }

  session_ = interpreter_->createSession(schedule);
  if (session_ == nullptr) {
    KZ_LOG_ERROR("mnn: failed to create a session");
    return false;
  }
  interpreter_->updateCacheFile(session_);
  interpreter_->releaseModel();

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  // getSession*All return name-ordered maps.
  for (const auto& [name, tensor] : interpreter_->getSessionInputAll(session_)) {
    const auto type = data_type_of(*tensor);
    if (!type) {
      KZ_LOG_ERROR("mnn: input {} has an unsupported type", name);
      return false;
    }
    input_names_.push_back(name);
    input_tensors_.push_back(tensor);
    // Packed: host tensors wrap the buffer, and dma-bufs would otherwise pad rows for media devices.
    const auto shape = shape_of(*tensor);
    input_protos_.push_back({memory::host_engine_input_memory(), *type, shape, core::contiguous_stride(shape, *type)});
  }
  for (const auto& [name, tensor] : interpreter_->getSessionOutputAll(session_)) {
    const auto type = data_type_of(*tensor);
    if (!type) {
      KZ_LOG_ERROR("mnn: output {} has an unsupported type", name);
      return false;
    }
    output_names_.push_back(name);
    output_tensors_.push_back(tensor);
    output_protos_.push_back({core::MemoryType::Cpu, *type, shape_of(*tensor), {}});
  }
  return true;
}

auto MnnEngine::set_inputs(const core::Buffers& inputs) -> bool {
  if (!Engine::set_inputs(inputs))
    return false;
  host_inputs_.clear();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto* data = inputs[i]->memory()->mutable_cpu();
    if (data == nullptr || !inputs[i]->is_contiguous()) {
      KZ_LOG_ERROR("mnn: input {} needs contiguous host memory", i);
      return false;
    }
    host_inputs_.emplace_back(
        MNN::Tensor::create(input_tensors_[i]->shape(), input_tensors_[i]->getType(), data, MNN::Tensor::CAFFE));
  }
  return true;
}

auto MnnEngine::set_outputs(const core::Buffers& outputs) -> bool {
  if (!Engine::set_outputs(outputs))
    return false;
  host_outputs_.clear();
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto* data = outputs[i]->memory()->mutable_cpu();
    if (data == nullptr || !outputs[i]->is_contiguous()) {
      KZ_LOG_ERROR("mnn: output {} needs contiguous host memory", i);
      return false;
    }
    host_outputs_.emplace_back(
        MNN::Tensor::create(output_tensors_[i]->shape(), output_tensors_[i]->getType(), data, MNN::Tensor::CAFFE));
  }
  return true;
}

auto MnnEngine::inference(core::Stream* stream) -> bool {
  for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
    // Preprocessing writes through the buffer, so refresh the host view.
    static_cast<void>(inputs_[i]->memory()->cpu(stream));
    input_tensors_[i]->copyFromHostTensor(host_inputs_[i].get());
  }
  if (interpreter_->runSession(session_) != MNN::NO_ERROR) {
    KZ_LOG_ERROR("mnn: runSession failed");
    return false;
  }
  for (std::size_t i = 0; i < output_tensors_.size(); ++i) {
    output_tensors_[i]->copyToHostTensor(host_outputs_[i].get());
    static_cast<void>(outputs_[i]->memory()->mutable_cpu());
  }
  return true;
}

auto MnnEngine::do_release() -> void {
  host_inputs_.clear();
  host_outputs_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
  if (interpreter_ != nullptr && session_ != nullptr)
    interpreter_->releaseSession(session_);
  session_ = nullptr;
  interpreter_.reset();
}

} // namespace hrt::engine
