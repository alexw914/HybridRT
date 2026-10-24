// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/postprocessor_factory.hpp
// Description: Declares creation and lookup of the built-in postprocessors,
//              registered by postprocess.type.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_POSTPROCESSOR_FACTORY_HPP
#define HRT_POSTPROCESS_POSTPROCESSOR_FACTORY_HPP

#include <hrt/postprocess/postprocessor.hpp>

#include <string_view>

namespace hrt::postprocess {

// The built-in postprocessor of `type` for Result, checked and initialized for
// `engine`. Aborts on an unknown type, a type of another Result, a config
// check_config() rejects, or engine outputs init() rejects.
template <typename Result>
auto create_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<Postprocessor<Result>>;

// Whether built-in `type` produces Result; constructs nothing.
template <typename Result>
auto supports_postprocessor(std::string_view type) -> bool;

// Problems in a "postprocess" section for built-in `type`, including an
// unknown type.
auto check_postprocess_config(std::string_view type, const kitzoo::json::Json& config) -> Vector<String>;

extern template auto create_postprocessor<DetectResultMap>(std::string_view, const kitzoo::json::Json&,
                                                           const engine::Engine&) -> UniquePtr<DetectPostprocessor>;
extern template auto create_postprocessor<RotatedResultMap>(std::string_view, const kitzoo::json::Json&,
                                                            const engine::Engine&) -> UniquePtr<RotatedPostprocessor>;
extern template auto create_postprocessor<ClassifyResult>(std::string_view, const kitzoo::json::Json&,
                                                          const engine::Engine&) -> UniquePtr<ClassifyPostprocessor>;
extern template auto create_postprocessor<EmbeddingResult>(std::string_view, const kitzoo::json::Json&,
                                                           const engine::Engine&) -> UniquePtr<EmbeddingPostprocessor>;
extern template auto create_postprocessor<KeypointResultMap>(std::string_view, const kitzoo::json::Json&,
                                                             const engine::Engine&) -> UniquePtr<KeypointPostprocessor>;
extern template auto create_postprocessor<SegmentResultMap>(std::string_view, const kitzoo::json::Json&,
                                                            const engine::Engine&) -> UniquePtr<SegmentPostprocessor>;
extern template auto create_postprocessor<OcrResult>(std::string_view, const kitzoo::json::Json&, const engine::Engine&)
    -> UniquePtr<OcrPostprocessor>;

extern template auto supports_postprocessor<DetectResultMap>(std::string_view) -> bool;
extern template auto supports_postprocessor<RotatedResultMap>(std::string_view) -> bool;
extern template auto supports_postprocessor<ClassifyResult>(std::string_view) -> bool;
extern template auto supports_postprocessor<EmbeddingResult>(std::string_view) -> bool;
extern template auto supports_postprocessor<KeypointResultMap>(std::string_view) -> bool;
extern template auto supports_postprocessor<SegmentResultMap>(std::string_view) -> bool;
extern template auto supports_postprocessor<OcrResult>(std::string_view) -> bool;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_POSTPROCESSOR_FACTORY_HPP
