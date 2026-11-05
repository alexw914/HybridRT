// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/openvino_engine.cpp
// Description: Implements the OpenVINO CPU inference backend.
// -----------------------------------------------------------------------------

#include "include/engine/openvino_engine.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <charconv>
#include <optional>
#include <string_view>

#include "include/core/config.hpp"

namespace hrt::engine {

namespace {

auto check_status(ov_status_e status, std::string_view operation) -> bool {
  if (status == OK)
    return true;
  const auto* message = ov_get_last_err_msg();
  KZ_LOG_ERROR("openvino: {} failed: {}", operation, message == nullptr ? ov_get_error_info(status) : message);
  return false;
}

auto data_type(ov_element_type_e type) -> std::optional<core::DataType> {
  switch (type) {
  case F32:
    return core::DataType::Float32;
  case F16:
    return core::DataType::Float16;
  case I64:
    return core::DataType::Int64;
  case I32:
    return core::DataType::Int32;
  case I8:
    return core::DataType::Int8;
  case U8:
    return core::DataType::Uint8;
  case BOOLEAN:
    return core::DataType::Bool;
  default:
    return std::nullopt;
  }
}

auto ov_type(core::DataType type) -> ov_element_type_e {
  switch (type) {
  case core::DataType::Float32:
    return F32;
  case core::DataType::Float16:
    return F16;
  case core::DataType::Int64:
    return I64;
  case core::DataType::Int32:
    return I32;
  case core::DataType::Int8:
    return I8;
  case core::DataType::Uint8:
    return U8;
  case core::DataType::Bool:
    return BOOLEAN;
  default:
    return DYNAMIC;
  }
}

auto tensor_shape(const core::BufferProto& proto) -> ov_shape_t {
  return {.rank = static_cast<std::int64_t>(proto.shape.size()), .dims = const_cast<std::int64_t*>(proto.shape.data())};
}

auto release_tensors(Vector<ov_tensor_t*>& tensors) -> void {
  for (auto* tensor : tensors)
    if (tensor != nullptr)
      ov_tensor_free(tensor);
  tensors.clear();
}

} // namespace

OpenVinoEngine::~OpenVinoEngine() {
  release();
}

auto OpenVinoEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.proto == nullptr || param.proto->empty()) {
    KZ_LOG_ERROR("openvino: IR XML in engine proto is empty");
    return false;
  }

  int num_threads = 0;
  core::Problems problems;
  core::check_keys(config, {"num_threads"}, "config", problems);
  core::read(config, "num_threads", num_threads, "config", problems);
  KZ_CHECK_MSG(problems.empty(), "openvino: invalid config: {}", core::join(problems));
  KZ_CHECK_MSG(num_threads >= 0, "openvino: num_threads must be non-negative");

  if (!check_status(ov_core_create(&core_), "create core"))
    return false;
  if (param.data != nullptr && !param.data->empty()) {
    std::int64_t weights_size = static_cast<std::int64_t>(param.data->size());
    const ov_shape_t weights_shape{.rank = 1, .dims = &weights_size};
    if (!check_status(
            ov_tensor_create_from_host_ptr(U8, weights_shape, const_cast<std::uint8_t*>(param.data->data()), &weights_),
            "create IR weights tensor"))
      return false;
  }
  if (!check_status(ov_core_read_model_from_memory_buffer(core_, reinterpret_cast<const char*>(param.proto->data()),
                                                          param.proto->size(), weights_, &model_),
                    "read IR model"))
    return false;

  if (!read_ports(true) || !read_ports(false))
    return false;

  ov_property_t properties[2]{{ov_property_key_hint_inference_precision, "f32"}, {}};
  std::size_t property_count = 1;
  Array<char, 16> thread_value{};
  if (num_threads > 0) {
    const auto [end, error] =
        std::to_chars(thread_value.data(), thread_value.data() + thread_value.size() - 1, num_threads);
    if (error != std::errc{}) {
      KZ_LOG_ERROR("openvino: cannot encode num_threads");
      return false;
    }
    *end = '\0';
    properties[1] = {ov_property_key_inference_num_threads, thread_value.data()};
    property_count = 2;
  }
  if (!check_status(ov_core_compile_model_props(core_, model_, "CPU", property_count, properties, &compiled_model_),
                    "compile model"))
    return false;
  if (!check_status(ov_compiled_model_create_infer_request(compiled_model_, &request_), "create infer request"))
    return false;

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  return true;
}

auto OpenVinoEngine::do_release() -> void {
  if (request_ != nullptr)
    ov_infer_request_free(request_);
  release_tensors(input_tensors_);
  release_tensors(output_tensors_);
  input_indexes_.clear();
  output_indexes_.clear();
  if (compiled_model_ != nullptr)
    ov_compiled_model_free(compiled_model_);
  if (model_ != nullptr)
    ov_model_free(model_);
  if (weights_ != nullptr)
    ov_tensor_free(weights_);
  if (core_ != nullptr)
    ov_core_free(core_);
  request_ = nullptr;
  compiled_model_ = nullptr;
  model_ = nullptr;
  core_ = nullptr;
}

auto OpenVinoEngine::set_inputs(const core::Buffers& inputs) -> bool {
  if (!Engine::set_inputs(inputs))
    return false;
  release_tensors(input_tensors_);
  input_tensors_.resize(inputs.size(), nullptr);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto* data = inputs[i]->memory()->mutable_cpu();
    if (data == nullptr || !inputs[i]->is_contiguous()) {
      KZ_LOG_ERROR("openvino: input {} needs contiguous host memory", i);
      release_tensors(input_tensors_);
      return false;
    }
    const auto& proto = input_protos_[i];
    if (!check_status(
            ov_tensor_create_from_host_ptr(ov_type(proto.data_type), tensor_shape(proto), data, &input_tensors_[i]),
            "create input tensor") ||
        !check_status(ov_infer_request_set_input_tensor_by_index(request_, input_indexes_[i], input_tensors_[i]),
                      "bind input tensor")) {
      release_tensors(input_tensors_);
      return false;
    }
  }
  return true;
}

auto OpenVinoEngine::set_outputs(const core::Buffers& outputs) -> bool {
  if (!Engine::set_outputs(outputs))
    return false;
  release_tensors(output_tensors_);
  output_tensors_.resize(outputs.size(), nullptr);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto* data = outputs[i]->memory()->mutable_cpu();
    if (data == nullptr || !outputs[i]->is_contiguous()) {
      KZ_LOG_ERROR("openvino: output {} needs contiguous host memory", i);
      release_tensors(output_tensors_);
      return false;
    }
    const auto& proto = output_protos_[i];
    if (!check_status(
            ov_tensor_create_from_host_ptr(ov_type(proto.data_type), tensor_shape(proto), data, &output_tensors_[i]),
            "create output tensor") ||
        !check_status(ov_infer_request_set_output_tensor_by_index(request_, output_indexes_[i], output_tensors_[i]),
                      "bind output tensor")) {
      release_tensors(output_tensors_);
      return false;
    }
  }
  return true;
}

auto OpenVinoEngine::inference(KZ_MAYBE_UNUSED core::Stream* stream) -> bool {
  return check_status(ov_infer_request_infer(request_), "run inference");
}

auto OpenVinoEngine::read_ports(bool input) -> bool {
  size_t count = 0;
  const auto status = input ? ov_model_inputs_size(model_, &count) : ov_model_outputs_size(model_, &count);
  if (!check_status(status, input ? "query input count" : "query output count"))
    return false;
  if (count == 0) {
    KZ_LOG_ERROR("openvino: model has no {} tensors", input ? "input" : "output");
    return false;
  }

  Vector<Pair<String, std::size_t>> names;
  auto& tensor_names = input ? input_names_ : output_names_;
  auto& protos = input ? input_protos_ : output_protos_;
  auto& indexes = input ? input_indexes_ : output_indexes_;
  for (std::size_t i = 0; i < count; ++i) {
    ov_output_const_port_t* port = nullptr;
    const auto port_status =
        input ? ov_model_const_input_by_index(model_, i, &port) : ov_model_const_output_by_index(model_, i, &port);
    if (!check_status(port_status, input ? "query input port" : "query output port"))
      return false;

    char* name = nullptr;
    ov_shape_t shape{};
    ov_element_type_e element_type = DYNAMIC;
    const auto ok = check_status(ov_port_get_any_name(port, &name), "query tensor name") &&
                    check_status(ov_const_port_get_shape(port, &shape), "query tensor shape") &&
                    check_status(ov_port_get_element_type(port, &element_type), "query tensor type");
    ov_output_const_port_free(port);
    if (!ok) {
      ov_free(name);
      ov_shape_free(&shape);
      return false;
    }
    const auto type = data_type(element_type);
    if (name == nullptr || shape.rank <= 0 || shape.dims == nullptr || !type) {
      KZ_LOG_ERROR("openvino: {} {} has a dynamic shape or unsupported type", input ? "input" : "output", i);
      ov_free(name);
      ov_shape_free(&shape);
      return false;
    }
    const String tensor_name{name};
    core::Shape dims(shape.dims, shape.dims + shape.rank);
    ov_free(name);
    ov_shape_free(&shape);
    if (std::any_of(dims.begin(), dims.end(), [](std::int64_t dim) { return dim <= 0; })) {
      KZ_LOG_ERROR("openvino: {} {} has a dynamic or empty dimension", input ? "input" : "output", i);
      return false;
    }
    names.emplace_back(tensor_name, i);
    protos.push_back({core::MemoryType::Cpu, *type, dims, core::contiguous_stride(dims, *type)});
  }

  Vector<std::size_t> order(count);
  for (std::size_t i = 0; i < count; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return names[a].first < names[b].first; });
  auto sorted_protos = protos;
  protos.clear();
  tensor_names.clear();
  indexes.clear();
  for (const auto i : order) {
    tensor_names.push_back(names[i].first);
    indexes.push_back(names[i].second);
    protos.push_back(sorted_protos[i]);
  }
  return true;
}

} // namespace hrt::engine
