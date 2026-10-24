// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/cann_engine.cpp
// Description: Implements the Ascend CANN backend.
// -----------------------------------------------------------------------------

#include "include/engine/cann_engine.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <optional>

#include "include/backend/ascend/acl_common.hpp"

namespace hrt::engine {

namespace {

auto data_type(aclDataType type) -> std::optional<core::DataType> {
  switch (type) {
  case ACL_FLOAT:
    return core::DataType::Float32;
  case ACL_FLOAT16:
    return core::DataType::Float16;
  case ACL_INT8:
    return core::DataType::Int8;
  case ACL_INT32:
    return core::DataType::Int32;
  case ACL_INT64:
    return core::DataType::Int64;
  case ACL_UINT8:
    return core::DataType::Uint8;
  default:
    return std::nullopt;
  }
}

auto tensor_shape(const aclmdlIODims& dims) -> core::Shape {
  return core::Shape(dims.dims, dims.dims + dims.dimCount);
}

auto destroy_dataset(aclmdlDataset*& dataset) -> void {
  if (dataset == nullptr)
    return;
  for (std::size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
    if (aclDestroyDataBuffer(aclmdlGetDatasetBuffer(dataset, i)) != ACL_SUCCESS)
      KZ_LOG_ERROR("cann: aclDestroyDataBuffer failed: {}", aclGetRecentErrMsg());
  }
  if (aclmdlDestroyDataset(dataset) != ACL_SUCCESS)
    KZ_LOG_ERROR("cann: aclmdlDestroyDataset failed: {}", aclGetRecentErrMsg());
  dataset = nullptr;
}

auto is_ascend(const core::Buffer& buffer) -> bool {
  return buffer.memory_type() == core::MemoryType::Acl || buffer.memory_type() == core::MemoryType::Dvpp;
}

// Name-ordered positions of the model's IO, as (name, model index).
template <typename NameAt>
auto sorted_by_name(std::size_t count, NameAt name_at) -> Map<String, std::size_t> {
  Map<String, std::size_t> sorted;
  for (std::size_t i = 0; i < count; ++i)
    sorted.emplace(name_at(i), i);
  return sorted;
}

} // namespace

CannEngine::~CannEngine() {
  release();
}

auto CannEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.data == nullptr || param.data->empty()) {
    KZ_LOG_ERROR("cann: engine data (the .om model) is empty");
    return false;
  }
  KZ_CHECK_MSG(config.empty(), "cann: config takes no settings");
  static_cast<void>(backend::AclRuntime::instance());
  // A model that does not load fails this backend, so the next one can be tried.
  KZ_CHECK_RETURN(aclmdlLoadFromMem(param.data->data(), param.data->size(), &model_id_) == ACL_SUCCESS, false,
                  "cann: aclmdlLoadFromMem failed: {}", aclGetRecentErrMsg());
  loaded_ = true;
  desc_ = aclmdlCreateDesc();
  KZ_CHECK_MSG(desc_ != nullptr && aclmdlGetDesc(desc_, model_id_) == ACL_SUCCESS, "cann: aclmdlGetDesc failed: {}",
               aclGetRecentErrMsg());

  const auto input_count = aclmdlGetNumInputs(desc_);
  if (input_count == 0)
    return false;
  switch (aclmdlGetInputFormat(desc_, 0)) {
  case ACL_FORMAT_NCHW:
    proto_.input_layout = DataLayout::Nchw;
    break;
  case ACL_FORMAT_NHWC:
    proto_.input_layout = DataLayout::Nhwc;
    break;
  default:
    KZ_LOG_ERROR("cann: unsupported input format");
    return false;
  }
  proto_.output_layout = DataLayout::Nchw;

  for (const auto& [name, index] :
       sorted_by_name(input_count, [&](std::size_t i) { return aclmdlGetInputNameByIndex(desc_, i); })) {
    aclmdlIODims dims{};
    const auto type = data_type(aclmdlGetInputDataType(desc_, index));
    KZ_CHECK_MSG(aclmdlGetInputDimsV2(desc_, index, &dims) == ACL_SUCCESS, "cann: aclmdlGetInputDimsV2 failed: {}",
                 aclGetRecentErrMsg());
    KZ_CHECK_RETURN(type.has_value(), false, "cann: input {} has an unsupported data type", index);
    input_names_.push_back(name);
    // VPC writes the inputs, so they come from the DVPP pool with its row padding.
    input_protos_.push_back({core::MemoryType::Dvpp, *type, tensor_shape(dims), {}});
    input_indexes_.push_back(index);
  }
  for (const auto& [name, index] : sorted_by_name(
           aclmdlGetNumOutputs(desc_), [&](std::size_t i) { return aclmdlGetOutputNameByIndex(desc_, i); })) {
    aclmdlIODims dims{};
    const auto type = data_type(aclmdlGetOutputDataType(desc_, index));
    KZ_CHECK_MSG(aclmdlGetOutputDims(desc_, index, &dims) == ACL_SUCCESS, "cann: aclmdlGetOutputDims failed: {}",
                 aclGetRecentErrMsg());
    KZ_CHECK_RETURN(type.has_value(), false, "cann: output {} has an unsupported data type", index);
    output_names_.push_back(name);
    output_protos_.push_back({core::MemoryType::Acl, *type, tensor_shape(dims), {}});
    output_indexes_.push_back(index);
  }
  input_staging_.resize(input_protos_.size());
  output_staging_.resize(output_protos_.size());
  return true;
}

auto CannEngine::make_dataset(const core::Buffers& buffers, const Vector<core::BufferProto>& protos,
                              const Vector<std::size_t>& indexes, core::Buffers& staging, aclmdlDataset*& dataset)
    -> bool {
  destroy_dataset(dataset);
  dataset = aclmdlCreateDataset();
  if (dataset == nullptr)
    return false;
  // Dataset buffers follow the model's own index order.
  for (std::size_t model_index = 0; model_index < buffers.size(); ++model_index) {
    const auto position = static_cast<std::size_t>(
        std::distance(indexes.begin(), std::find(indexes.begin(), indexes.end(), model_index)));
    const auto& buffer = buffers[position];
    auto& stage = staging[position];
    if (is_ascend(*buffer) && buffer->is_contiguous()) {
      stage.reset();
    } else {
      stage = std::make_shared<core::Buffer>(core::MemoryType::Acl, protos[position].data_type, protos[position].shape);
      if (!stage->valid())
        return false;
    }
    const auto& bound = stage != nullptr ? *stage : *buffer;
    auto* data = aclCreateDataBuffer(reinterpret_cast<void*>(bound.memory()->device()), bound.memory()->bytes());
    KZ_CHECK_RETURN(data != nullptr && aclmdlAddDatasetBuffer(dataset, data) == ACL_SUCCESS, false,
                    "cann: cannot add a dataset buffer: {}", aclGetRecentErrMsg());
  }
  return true;
}

auto CannEngine::set_inputs(const core::Buffers& inputs) -> bool {
  return Engine::set_inputs(inputs) &&
         make_dataset(inputs_, input_protos_, input_indexes_, input_staging_, input_dataset_);
}

auto CannEngine::set_outputs(const core::Buffers& outputs) -> bool {
  return Engine::set_outputs(outputs) &&
         make_dataset(outputs_, output_protos_, output_indexes_, output_staging_, output_dataset_);
}

auto CannEngine::inference(core::Stream* stream) -> bool {
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    if (input_staging_[i] != nullptr &&
        !core::Buffer::copy(*input_staging_[i], *inputs_[i], core::CopyPolicy::DirectCopy, stream))
      return false;
  }
  const auto status = stream == nullptr
                          ? aclmdlExecute(model_id_, input_dataset_, output_dataset_)
                          : aclmdlExecuteAsync(model_id_, input_dataset_, output_dataset_, backend::acl_stream(stream));
  KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "cann: execution failed: {}", aclGetRecentErrMsg());
  bool host_outputs = false;
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    if (output_staging_[i] == nullptr)
      continue;
    if (!core::Buffer::copy(*outputs_[i], *output_staging_[i], core::CopyPolicy::DirectCopy, stream))
      return false;
    host_outputs = true;
  }
  if (host_outputs && stream != nullptr)
    stream->synchronize();
  return true;
}

auto CannEngine::do_release() -> void {
  destroy_dataset(input_dataset_);
  destroy_dataset(output_dataset_);
  input_staging_.clear();
  output_staging_.clear();
  input_indexes_.clear();
  output_indexes_.clear();
  if (loaded_ && aclmdlUnload(model_id_) != ACL_SUCCESS)
    KZ_LOG_ERROR("cann: aclmdlUnload failed: {}", aclGetRecentErrMsg());
  loaded_ = false;
  if (desc_ != nullptr && aclmdlDestroyDesc(desc_) != ACL_SUCCESS)
    KZ_LOG_ERROR("cann: aclmdlDestroyDesc failed: {}", aclGetRecentErrMsg());
  desc_ = nullptr;
}

} // namespace hrt::engine
