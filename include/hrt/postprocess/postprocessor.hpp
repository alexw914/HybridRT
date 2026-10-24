// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/postprocessor.hpp
// Description: Declares the postprocessor interface shared by all tasks.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_POSTPROCESSOR_HPP
#define HRT_POSTPROCESS_POSTPROCESSOR_HPP

#include <hrt/core/buffer.hpp>
#include <hrt/engine/engine.hpp>
#include <hrt/postprocess/result_types.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>

#include <string_view>

namespace hrt::postprocess {

// Postprocessing runs on the host: outputs without host memory are copied to
// reusable host buffers first, others are synced through Memory.
class PostprocessorBase {
public:
  virtual ~PostprocessorBase();

  PostprocessorBase(const PostprocessorBase&) = delete;
  auto operator=(const PostprocessorBase&) -> PostprocessorBase& = delete;
  PostprocessorBase(PostprocessorBase&&) = delete;
  auto operator=(PostprocessorBase&&) -> PostprocessorBase& = delete;

  virtual auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool = 0;

  // Problems in `config` (the "postprocess" section) that init() would
  // misread; never throws.
  KZ_NODISCARD virtual auto check_config(const kitzoo::json::Json& config) const -> Vector<String> = 0;

protected:
  PostprocessorBase() = default;

  auto to_host(const core::Buffers& outputs, core::Stream* stream) -> const core::Buffers*;

  // Sample `index` of every output of `host` (from to_host()); outputs whose
  // batch is 1 are returned as they are.
  auto sample_outputs(const core::Buffers& host, std::size_t index) -> const core::Buffers*;

private:
  core::Buffers host_;
  core::Buffers view_;
  core::Buffers sample_;
};

template <typename Result>
class Postprocessor : public PostprocessorBase {
public:
  using ResultType = Result;

  auto process(Result& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform,
               core::Stream* stream = nullptr) -> bool {
    result.clear();
    const auto* host = to_host(outputs, stream);
    return host != nullptr && decode(result, *host, transform);
  }

  // One result per transform from batched outputs: sample i of every output
  // belongs to transforms[i], and samples past transforms.size() are ignored.
  auto process(Vector<Result>& results, const core::Buffers& outputs,
               const Vector<preprocess::ImageTransform>& transforms, core::Stream* stream = nullptr) -> bool {
    results.assign(transforms.size(), Result{});
    const auto* host = to_host(outputs, stream);
    if (host == nullptr)
      return false;
    for (std::size_t i = 0; i < transforms.size(); ++i) {
      const auto* sample = sample_outputs(*host, i);
      if (sample == nullptr || !decode(results[i], *sample, transforms[i]))
        return false;
    }
    return true;
  }

protected:
  virtual auto decode(Result& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool = 0;
};

using DetectPostprocessor = Postprocessor<DetectResultMap>;

// Source mapping requires uniform scaling, allowing integer letterbox rounding.
using RotatedPostprocessor = Postprocessor<RotatedResultMap>;

using ClassifyPostprocessor = Postprocessor<ClassifyResult>;

using EmbeddingPostprocessor = Postprocessor<EmbeddingResult>;

using KeypointPostprocessor = Postprocessor<KeypointResultMap>;

using SegmentPostprocessor = Postprocessor<SegmentResultMap>;

using OcrPostprocessor = Postprocessor<OcrResult>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_POSTPROCESSOR_HPP
