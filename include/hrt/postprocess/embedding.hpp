// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/embedding.hpp
// Description: Declares embedding results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_EMBEDDING_HPP
#define HRT_POSTPROCESS_EMBEDDING_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <string_view>

namespace hrt::postprocess {

// The single engine output as float, otherwise unchanged.
using EmbeddingResult = Vector<float>;

using EmbeddingPostprocessor = Postprocessor<EmbeddingResult>;

// Aborts on an unknown type, a config check_config rejects, or an engine
// without exactly one output.
auto create_embedding_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                    const engine::Engine& engine) -> UniquePtr<EmbeddingPostprocessor>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_EMBEDDING_HPP
