// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/engine/ncnn_engine.cpp
// Description: Implements the ncnn backend.
// -----------------------------------------------------------------------------

#include "include/engine/ncnn_engine.hpp"

#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <optional>
#include <utility>

#include "include/core/config.hpp"
#include "include/memory/cpu_memory.hpp"

namespace hrt::engine {

namespace {

auto data_type_of(const ncnn::Mat& mat) -> std::optional<core::DataType> {
  switch (mat.elemsize) {
  case 1:
    return core::DataType::Uint8;
  case 4:
    return core::DataType::Float32;
  default:
    return std::nullopt;
  }
}

// Element shape with a leading batch of 1: [1, (c, (d,)) (h,) w].
auto shape_of(const ncnn::Mat& mat) -> core::Shape {
  switch (mat.dims) {
  case 1:
    return {1, mat.w};
  case 2:
    return {1, mat.h, mat.w};
  case 3:
    return {1, mat.c, mat.h, mat.w};
  default:
    return {1, mat.c, mat.d, mat.h, mat.w};
  }
}

// Byte strides over the byte shape; channels are cstep apart.
auto stride_of(const ncnn::Mat& mat) -> core::Stride {
  const auto row = static_cast<std::size_t>(mat.w) * mat.elemsize;
  const auto channel = mat.cstep * mat.elemsize;
  const auto total = channel * static_cast<std::size_t>(mat.c);
  switch (mat.dims) {
  case 1:
    return {row, row};
  case 2:
    return {row * static_cast<std::size_t>(mat.h), row * static_cast<std::size_t>(mat.h), row};
  case 3:
    return {total, total, channel, row};
  default:
    return {total, total, channel, row * static_cast<std::size_t>(mat.h), row};
  }
}

// Name-sorted (name, blob index) pairs so every backend orders IO alike.
auto sorted_indexes(const Vector<const char*>& names, const Vector<int>& indexes) -> Vector<int> {
  Map<String, int> by_name;
  for (std::size_t i = 0; i < names.size(); ++i)
    by_name.emplace(names[i], indexes[i]);
  Vector<int> sorted;
  for (const auto& [name, index] : by_name)
    sorted.push_back(index);
  return sorted;
}

// External-data Mat with the same dims as `shape`.
auto wrap_mat(const ncnn::Mat& shape, void* data) -> ncnn::Mat {
  switch (shape.dims) {
  case 1:
    return {shape.w, data, shape.elemsize, shape.elempack};
  case 2:
    return {shape.w, shape.h, data, shape.elemsize, shape.elempack};
  case 3:
    return {shape.w, shape.h, shape.c, data, shape.elemsize, shape.elempack};
  default:
    return {shape.w, shape.h, shape.d, shape.c, data, shape.elemsize, shape.elempack};
  }
}

} // namespace

NcnnEngine::~NcnnEngine() {
  release();
}

auto NcnnEngine::do_initialize(const hrtc::EngineParam& param, const kitzoo::json::Json& config) -> bool {
  if (param.proto == nullptr || param.proto->empty()) {
    KZ_LOG_ERROR("ncnn: engine proto (the .param text) is empty");
    return false;
  }
  core::Problems problems;
  core::check_keys(config, {"num_threads"}, "config", problems);
  core::read(config, "num_threads", num_threads_, "config", problems);
  KZ_CHECK_MSG(problems.empty(), "ncnn: {}", core::join(problems));

  net_ = std::make_unique<ncnn::Net>();
  net_->opt.num_threads = num_threads_;
  net_->opt.lightmode = true;
  param_text_.assign(param.proto->begin(), param.proto->end());
  weights_ = param.data;
  // load_model needs a readable pointer even for weightless graphs.
  static const unsigned char kNoWeights[1] = {};
  const auto* weights = weights_ == nullptr || weights_->empty() ? kNoWeights : weights_->data();
  if (net_->load_param_mem(param_text_.c_str()) != 0 ||
      // load_model returns int or size_t depending on the ncnn version.
      std::cmp_not_equal(net_->load_model(weights), weights_ == nullptr ? 0 : weights_->size())) {
    KZ_LOG_ERROR("ncnn: failed to load the model");
    return false;
  }

  proto_ = {DataLayout::Nchw, DataLayout::Nchw};
  input_indexes_ = sorted_indexes(net_->input_names(), net_->input_indexes());
  output_indexes_ = sorted_indexes(net_->output_names(), net_->output_indexes());
  for (auto index : input_indexes_) {
    const auto& blob = net_->blobs()[static_cast<std::size_t>(index)];
    ncnn::Mat mat;
    mat.create_like(blob.shape);
    const auto type = data_type_of(mat);
    // Inputs are wrapped in place, which needs densely packed channels.
    if (mat.dims == 0 || !type || mat.cstep != static_cast<std::size_t>(mat.w * mat.h * mat.d)) {
      KZ_LOG_ERROR("ncnn: input {} needs a fixed shape, packed channels and uint8/float32 data", blob.name);
      return false;
    }
    input_names_.push_back(blob.name);
    input_protos_.push_back({memory::host_engine_input_memory(), *type, shape_of(mat), stride_of(mat)});
  }
  for (auto index : output_indexes_) {
    const auto& blob = net_->blobs()[static_cast<std::size_t>(index)];
    ncnn::Mat mat;
    mat.create_like(blob.shape);
    const auto type = data_type_of(mat);
    if (mat.dims == 0 || !type) {
      KZ_LOG_ERROR("ncnn: output {} needs a fixed shape and uint8/float32 data", blob.name);
      return false;
    }
    output_names_.push_back(blob.name);
    output_protos_.push_back({core::MemoryType::Cpu, *type, shape_of(mat), stride_of(mat)});
  }
  output_mats_.resize(output_indexes_.size());
  return true;
}

auto NcnnEngine::set_inputs(const core::Buffers& inputs) -> bool {
  if (!Engine::set_inputs(inputs))
    return false;
  input_mats_.clear();
  input_refcounts_ = std::make_unique<int[]>(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto* data = inputs[i]->memory()->mutable_cpu();
    if (data == nullptr || !inputs[i]->is_contiguous()) {
      KZ_LOG_ERROR("ncnn: input {} needs contiguous host memory", i);
      return false;
    }
    const auto& shape = net_->blobs()[static_cast<std::size_t>(input_indexes_[i])].shape;
    auto& mat = input_mats_.emplace_back(wrap_mat(shape, data));
    // Light mode dereferences refcount before in-place layers; a count above 1
    // marks the data as shared, so ncnn clones instead of writing or freeing it.
    input_refcounts_[i] = 2;
    mat.refcount = &input_refcounts_[i];
  }
  return true;
}

auto NcnnEngine::inference(core::Stream* stream) -> bool {
  auto extractor = net_->create_extractor();
  extractor.set_light_mode(true);
  for (std::size_t i = 0; i < input_indexes_.size(); ++i) {
    // Preprocessing may write through a device (dma-buf); refresh the host view the Mat wraps.
    static_cast<void>(inputs_[i]->memory()->cpu(stream));
    if (extractor.input(input_indexes_[i], input_mats_[i]) != 0)
      return false;
  }
  for (std::size_t i = 0; i < output_indexes_.size(); ++i) {
    auto& mat = output_mats_[i];
    if (extractor.extract(output_indexes_[i], mat) != 0) {
      KZ_LOG_ERROR("ncnn: failed to extract output {}", output_names_[i]);
      return false;
    }
    auto& output = *outputs_[i];
    const auto stride = stride_of(mat);
    auto view =
        core::Memory::wrap(core::MemoryType::Cpu, mat.data, 0, stride.front(), output.memory()->shape(), stride);
    if (view == nullptr || !core::Memory::copy(*output.memory(), *view, core::CopyPolicy::DirectCopy, stream))
      return false;
  }
  return true;
}

auto NcnnEngine::do_release() -> void {
  input_mats_.clear();
  output_mats_.clear();
  input_indexes_.clear();
  output_indexes_.clear();
  net_.reset();
  weights_ = nullptr;
  param_text_.clear();
}

} // namespace hrt::engine
