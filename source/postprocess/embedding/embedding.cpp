// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/embedding/embedding.cpp
// Description: Implements the Embedding postprocessor, which returns the
//              single engine output as a float vector.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

class Embedding final : public EmbeddingPostprocessor {
public:
  auto init(const kitzoo::json::Json& /*config*/, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "embedding"))
      return false;
    KZ_CHECK_RETURN(engine.output_protos().size() == 1, false, "postprocess: Embedding needs one engine output, got {}",
                    engine.output_protos().size());
    scratch_ = create_scratch(engine);
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_json(config, kitzoo::json::Json::object(), "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(EmbeddingResult& result, const core::Buffers& outputs, const preprocess::ImageTransform& /*transform*/)
      -> bool override {
    const core::Buffer* floats = outputs.front().get();
    if (floats->data_type() != core::DataType::Float32) {
      dequantize_buffer(*floats, scratch_[0]);
      floats = &scratch_[0];
    }
    const auto* data = static_cast<const float*>(floats->memory()->cpu());
    result.assign(data, data + floats->volume());
    return true;
  }

private:
  Vector<core::Buffer> scratch_;
};

} // namespace

auto create_embedding(std::string_view type) -> UniquePtr<EmbeddingPostprocessor> {
  return type == "Embedding" ? std::make_unique<Embedding>() : nullptr;
}

auto create_embedding_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                    const engine::Engine& engine) -> UniquePtr<EmbeddingPostprocessor> {
  return check_init(create_embedding(type), type, config, engine);
}

} // namespace hrt::postprocess
