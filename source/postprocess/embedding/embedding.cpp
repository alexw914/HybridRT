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
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_json(config, kitzoo::json::Json::object(), "postprocess", {"type"}, problems);
    return problems;
  }

  auto init(const kitzoo::json::Json& /*config*/, const engine::Engine& engine) -> bool override {
    KZ_CHECK_RETURN(engine.output_protos().size() == 1, false, "postprocess: Embedding needs one engine output, got {}",
                    engine.output_protos().size());
    return true;
  }

protected:
  auto decode(EmbeddingResult& result, const core::Buffers& outputs, const preprocess::ImageTransform& /*transform*/)
      -> bool override {
    const auto floats = float_view(outputs.front(), scratch_);
    if (floats == nullptr)
      return false;
    const auto* data = static_cast<const float*>(floats->memory()->cpu());
    result.assign(data, data + floats->volume());
    return true;
  }

private:
  SharedPtr<core::Buffer> scratch_;
};

} // namespace

auto make_embedding(std::string_view type) -> UniquePtr<EmbeddingPostprocessor> {
  return type == "Embedding" ? std::make_unique<Embedding>() : nullptr;
}

auto create_embedding_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                    const engine::Engine& engine) -> UniquePtr<EmbeddingPostprocessor> {
  return checked(make_embedding(type), type, config, engine);
}

} // namespace hrt::postprocess
