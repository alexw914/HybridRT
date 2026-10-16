// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/rknn_engine.cpp
// Description: Implements the Rockchip RKNN backend.
// -----------------------------------------------------------------------------

#include "include/engine/rknn_engine.hpp"

#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>

#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <optional>

#include "include/core/config.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(rknn_core_mask, {
                                                 {RKNN_NPU_CORE_AUTO, "RKNN_NPU_CORE_AUTO"},
                                                 {RKNN_NPU_CORE_0, "RKNN_NPU_CORE_0"},
                                                 {RKNN_NPU_CORE_1, "RKNN_NPU_CORE_1"},
                                                 {RKNN_NPU_CORE_2, "RKNN_NPU_CORE_2"},
                                                 {RKNN_NPU_CORE_0_1, "RKNN_NPU_CORE_0_1"},
                                                 {RKNN_NPU_CORE_0_1_2, "RKNN_NPU_CORE_0_1_2"},
                                             })

namespace hrt::engine {

namespace {

auto data_type_of(rknn_tensor_type type) -> std::optional<core::DataType> {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return core::DataType::Float32;
  case RKNN_TENSOR_UINT8:
    return core::DataType::Uint8;
  case RKNN_TENSOR_INT8:
    return core::DataType::Int8;
  case RKNN_TENSOR_INT32:
    return core::DataType::Int32;
  case RKNN_TENSOR_INT64:
    return core::DataType::Int64;
  default:
    return std::nullopt;
  }
}

auto shape_of(const rknn_tensor_attr& attr) -> core::Shape {
  return core::Shape(attr.dims, attr.dims + attr.n_dims);
}

// Byte strides honoring the NPU's row (w_stride) and height (h_stride) padding.
auto stride_of(const rknn_tensor_attr& attr, core::DataType type) -> core::Stride {
  const auto dims = static_cast<std::size_t>(attr.n_dims);
  core::Stride stride(dims);
  stride[dims - 1] = attr.dims[dims - 1] * core::data_type_size(type);
  for (auto i = dims - 1; i > 0; --i) {
    std::size_t extent = attr.dims[i - 1];
    if (i - 1 == 2 && attr.w_stride != 0)
      extent = attr.w_stride;
    else if (i - 1 == 1 && attr.h_stride != 0)
      extent = attr.h_stride;
    stride[i - 1] = stride[i] * extent;
  }
  return stride;
}

// Optional: a library that does not load is skipped, and a model that needs its
// operators then fails to run.
auto register_custom_ops(rknn_context context, const Vector<String>& libraries) -> void {
  using Register = int (*)(rknn_context);
  for (const auto& library : libraries) {
    auto* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_GLOBAL);
    auto* symbol = handle == nullptr ? nullptr : dlsym(handle, "hrt_rknn_register_custom_ops");
    if (symbol == nullptr) {
      KZ_LOG_WARN("rknn: skipping custom op library {}: {}", library, dlerror());
      continue;
    }
    if (reinterpret_cast<Register>(symbol)(context) != RKNN_SUCC)
      KZ_LOG_WARN("rknn: custom op library {} failed to register its operators", library);
  }
}

} // namespace

RknnEngine::~RknnEngine() {
  release();
}

auto RknnEngine::make_mem(rknn_tensor_mem* mem) -> TensorMem {
  return {mem, [context = context_](rknn_tensor_mem* m) {
            if (m != nullptr)
              rknn_destroy_mem(context, m);
          }};
}

auto RknnEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.data == nullptr || param.data->empty()) {
    KZ_LOG_ERROR("rknn: engine data (the .rknn model) is empty");
    return false;
  }
  bool image_input = true;
  Vector<String> custom_op_libraries;
  auto core_mask = RKNN_NPU_CORE_AUTO;
  core::Problems problems;
  core::check_keys(config, {"image_input", "custom_op_libraries", "core_mask"}, "config", problems);
  core::read(config, "image_input", image_input, "config", problems);
  core::read(config, "custom_op_libraries", custom_op_libraries, "config", problems);
  core::read_enum(config, "core_mask", core_mask, "config", problems);
  KZ_CHECK_MSG(problems.empty(), "rknn: {}", core::join(problems));

  // A model that does not load fails this backend, so the next one can be tried.
  auto status = rknn_init(&context_, const_cast<std::uint8_t*>(param.data->data()),
                          static_cast<std::uint32_t>(param.data->size()), 0, nullptr);
  KZ_CHECK_RETURN(status == RKNN_SUCC, false, "rknn: rknn_init failed: {}", status);
  register_custom_ops(context_, custom_op_libraries);
  status = rknn_set_core_mask(context_, core_mask);
  KZ_CHECK_RETURN(status == RKNN_SUCC, false, "rknn: rknn_set_core_mask failed: {}", status);

  rknn_input_output_num io{};
  status = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  KZ_CHECK_MSG(status == RKNN_SUCC, "rknn: querying the IO count failed: {}", status);
  proto_ = {DataLayout::Nhwc, DataLayout::Nchw};

  Map<String, rknn_tensor_attr> inputs;
  for (std::uint32_t i = 0; i < io.n_input; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    status = rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &attr, sizeof(attr));
    if (status == RKNN_SUCC && attr.fmt == RKNN_TENSOR_NC1HWC2)
      status = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    KZ_CHECK_MSG(status == RKNN_SUCC, "rknn: querying input {} failed: {}", i, status);
    inputs.emplace(attr.name, attr);
  }
  for (auto& [name, attr] : inputs) {
    // Images feed uint8; other inputs use int8 or float32.
    switch (attr.type) {
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_INT8:
      attr.type = image_input ? RKNN_TENSOR_UINT8 : RKNN_TENSOR_INT8;
      break;
    case RKNN_TENSOR_FLOAT16:
      attr.type = image_input ? RKNN_TENSOR_UINT8 : RKNN_TENSOR_FLOAT32;
      attr.size = image_input ? attr.size / 2 : attr.size * 2;
      attr.size_with_stride = image_input ? attr.size_with_stride / 2 : attr.size_with_stride * 2;
      break;
    default:
      break;
    }
    const auto type = data_type_of(attr.type);
    if (!type) {
      KZ_LOG_ERROR("rknn: input {} has an unsupported type", name);
      return false;
    }
    const auto memory = *type == core::DataType::Float32 ? core::MemoryType::Cpu : core::MemoryType::DmaBuffer;
    input_names_.push_back(name);
    input_protos_.push_back({memory, *type, shape_of(attr), stride_of(attr, *type)});
    input_attrs_.push_back(attr);
  }

  Map<String, rknn_tensor_attr> outputs;
  for (std::uint32_t i = 0; i < io.n_output; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    status = rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    KZ_CHECK_MSG(status == RKNN_SUCC, "rknn: querying output {} failed: {}", i, status);
    if (attr.type == RKNN_TENSOR_FLOAT16) {
      attr.type = RKNN_TENSOR_FLOAT32;
      attr.size *= 2;
      attr.size_with_stride *= 2;
    }
    outputs.emplace(attr.name, attr);
  }
  for (auto& [name, attr] : outputs) {
    const auto type = data_type_of(attr.type);
    auto mem = make_mem(rknn_create_mem(context_, attr.size));
    KZ_CHECK_RETURN(type.has_value(), false, "rknn: output {} has an unsupported data type", name);
    KZ_CHECK_MSG(mem != nullptr && rknn_set_io_mem(context_, mem.get(), &attr) == RKNN_SUCC,
                 "rknn: cannot prepare output {}", name);
    output_names_.push_back(name);
    output_protos_.push_back({core::MemoryType::Cpu, *type, shape_of(attr), {}});
    output_attrs_.push_back(attr);
    output_mems_.push_back(std::move(mem));
  }
  return true;
}

auto RknnEngine::set_inputs(const core::Buffers& inputs) -> bool {
  if (!Engine::set_inputs(inputs))
    return false;
  input_mems_.clear();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& memory = inputs[i]->memory();
    // DmaBuffer inputs are shared with the NPU by fd; others are copied each run.
    auto mem =
        memory->memory_type() == core::MemoryType::DmaBuffer
            ? make_mem(rknn_create_mem_from_fd(context_, static_cast<std::int32_t>(memory->device()),
                                               memory->mutable_cpu(), static_cast<std::uint32_t>(memory->bytes()), 0))
            : make_mem(rknn_create_mem(context_, static_cast<std::uint32_t>(memory->bytes())));
    KZ_CHECK_RETURN(mem != nullptr && rknn_set_io_mem(context_, mem.get(), &input_attrs_[i]) == RKNN_SUCC, false,
                    "rknn: cannot bind input {}", i);
    input_mems_.push_back(std::move(mem));
  }
  return true;
}

auto RknnEngine::inference(core::Stream* stream) -> bool {
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    const auto& memory = inputs_[i]->memory();
    if (memory->memory_type() == core::MemoryType::DmaBuffer) {
      if (!memory->sync_to_device(stream))
        return false;
    } else {
      std::memcpy(input_mems_[i]->virt_addr, memory->cpu(stream), memory->bytes());
    }
  }
  const auto status = rknn_run(context_, nullptr);
  KZ_CHECK_RETURN(status == RKNN_SUCC, false, "rknn: rknn_run failed: {}", status);
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    auto& output = *outputs_[i];
    const auto& attr = output_attrs_[i];
    std::memcpy(output.memory()->mutable_cpu(stream), output_mems_[i]->virt_addr,
                std::min<std::size_t>(output.byte_size(), output_mems_[i]->size));
    if (attr.type == RKNN_TENSOR_INT8) {
      output.scale = attr.scale;
      output.zero_point = attr.zp;
    }
  }
  return true;
}

auto RknnEngine::do_release() -> void {
  input_mems_.clear();
  output_mems_.clear();
  input_attrs_.clear();
  output_attrs_.clear();
  if (context_ != 0)
    rknn_destroy(context_);
  context_ = 0;
}

} // namespace hrt::engine
