// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/embedding.cpp
// Description: Implements the embedding postprocessor factory.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/embedding.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto create_embedding(std::string_view type) -> UniquePtr<EmbeddingPostprocessor> {
  return type == "Embedding" ? create_embedding() : nullptr;
}

auto create_embedding_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                    const engine::Engine& engine) -> UniquePtr<EmbeddingPostprocessor> {
  return check_init(create_embedding(type), type, config, engine);
}

} // namespace hrt::postprocess
