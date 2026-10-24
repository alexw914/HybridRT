// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/model/image_model.hpp
// Description: Declares the image model that chains preprocessing, inference
//              and postprocessing for one hrtc algorithm with pooled resources.
// -----------------------------------------------------------------------------

#ifndef HRT_MODEL_IMAGE_MODEL_HPP
#define HRT_MODEL_IMAGE_MODEL_HPP

#include <hrt/core/context.hpp>
#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>
#include <hrt/postprocess/postprocessor.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/thread/object_pool.hpp>

#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace hrt::model {

enum class Task {
  Detect,
  Rotated,
  Classify,
  Keypoint,
  Segment,
  Embedding,
};

auto task_of(std::string_view postprocess_type) -> std::optional<Task>;
auto task_of(const hrtc::AlgoParam& algo) -> std::optional<Task>;

// Problems in the config of `algo` (unknown keys, mistyped or unknown values,
// unknown postprocess type), one line each; never throws.
auto check_algo_config(const hrtc::AlgoParam& algo) -> Vector<String>;

// How many calls each stage serves at once, 1 to 64. Every resource is pooled
// up front: preprocessors, engines, postprocessors, input and output buffers
// and streams.
struct ImageModelOptions {
  std::size_t preprocess = 1;
  std::size_t engine = 1;
  std::size_t postprocess = 1;
};

// Runs one algorithm: preprocessing, inference and postprocessing, with all
// device work in `context`.
template <typename Result>
class ImageModel {
public:
  using PostprocessorFactory = std::function<UniquePtr<postprocess::Postprocessor<Result>>(
      const kitzoo::json::Json& config, const engine::Engine& engine)>;

  // Runs everything in `context`: CPU engines run under any context, device
  // engines and device preprocessors (cuda, dvpp) need one of their device.
  // Aborts when the config fails check_algo_config (only "postprocess.type" with a
  // custom factory), `options` is out of range, no engine runs on `context`,
  // or the engine does not fit the config.
  static auto create(const hrtc::AlgoParam& algo, SharedPtr<core::Context> context,
                     const ImageModelOptions& options = {}, const PostprocessorFactory& factory = {})
      -> UniquePtr<ImageModel>;
  // Uses the first engine of `algo`, in package order, whose device is usable
  // and whose model loads, in a new context that serves it and the
  // preprocessor: the engine's device, else the preprocessor's (cuda, dvpp),
  // else the CPU.
  static auto create(const hrtc::AlgoParam& algo, const ImageModelOptions& options = {},
                     const PostprocessorFactory& factory = {}) -> UniquePtr<ImageModel>;
  // From one entry of a manifest's "algos", {"name", "config"}; file paths are
  // absolute or relative to the working directory. Aborts when it cannot be
  // read.
  static auto create(const kitzoo::json::Json& algo, SharedPtr<core::Context> context,
                     const ImageModelOptions& options = {}, const PostprocessorFactory& factory = {})
      -> UniquePtr<ImageModel>;
  static auto create(const kitzoo::json::Json& algo, const ImageModelOptions& options = {},
                     const PostprocessorFactory& factory = {}) -> UniquePtr<ImageModel>;

  ~ImageModel();

  ImageModel(const ImageModel&) = delete;
  auto operator=(const ImageModel&) -> ImageModel& = delete;
  ImageModel(ImageModel&&) = delete;
  auto operator=(ImageModel&&) -> ImageModel& = delete;

  // Thread-safe; results are in source coordinates. `src_rect` must lie
  // inside `src` unless the preprocess resize_type is pad_outside; cascades
  // grow and clip detections with core::pad_scale, core::cover_pixels and
  // core::intersect.
  auto process_sync(const core::Image& src, const core::Rect& src_rect = {}) -> std::optional<Result>;
  // Runs any number of images in batches of up to batch(): each image is
  // preprocessed into its own sample of the engine input, the engine runs once
  // per batch, and one result per image comes back in order. `src_rects` is
  // empty (whole images) or one per image.
  auto process_sync(const Vector<core::Image>& srcs, const Vector<core::Rect>& src_rects = {})
      -> std::optional<Vector<Result>>;

  KZ_NODISCARD auto context() const -> const SharedPtr<core::Context>& { return context_; }

  KZ_NODISCARD auto preprocessor_param() const -> const preprocess::ImagePreprocessorParam& {
    return preprocessor_param_;
  }

  KZ_NODISCARD auto options() const -> const ImageModelOptions& { return options_; }

  // Static batch of the engine's image input, from the leading dim of its shape.
  KZ_NODISCARD auto batch() const -> std::size_t { return batch_; }

private:
  ImageModel(SharedPtr<core::Context> context, const ImageModelOptions& options);

  auto build(const hrtc::AlgoParam& algo, const kitzoo::json::Json& config, const PostprocessorFactory& factory,
             UniquePtr<engine::Engine> first, const hrtc::EngineParam& chosen) -> void;

  auto run(std::span<const core::Image> srcs, std::span<const core::Rect> src_rects) -> std::optional<Vector<Result>>;

  // Runs at most batch() images as one engine call and appends their results.
  auto run_batch(std::span<const core::Image> srcs, std::span<const core::Rect> src_rects, core::Stream* stream,
                 Vector<Result>& results) -> bool;

  KZ_NODISCARD auto create_input_image(const core::Buffer& input) const -> std::optional<core::Image>;

  // Image view of sample `index` of the engine input `input`.
  KZ_NODISCARD auto input_image(const core::Buffers& input, std::size_t index) const -> std::optional<core::Image>;

  SharedPtr<core::Context> context_;
  ImageModelOptions options_;
  engine::DataLayout input_layout_ = engine::DataLayout::Nchw;
  std::size_t batch_ = 1;
  preprocess::ImagePreprocessorParam preprocessor_param_;
  kitzoo::thread::ObjectPool<UniquePtr<preprocess::ImagePreprocessor>> preprocessor_pool_;
  kitzoo::thread::ObjectPool<UniquePtr<engine::Engine>> engine_pool_;
  kitzoo::thread::ObjectPool<core::Buffers> input_pool_;
  kitzoo::thread::ObjectPool<core::Buffers> output_pool_;
  kitzoo::thread::ObjectPool<UniquePtr<core::Stream>> stream_pool_;
  kitzoo::thread::ObjectPool<UniquePtr<postprocess::Postprocessor<Result>>> postprocessor_pool_;
};

extern template class ImageModel<postprocess::DetectResultMap>;
extern template class ImageModel<postprocess::RotatedResultMap>;
extern template class ImageModel<postprocess::ClassifyResult>;
extern template class ImageModel<postprocess::EmbeddingResult>;
extern template class ImageModel<postprocess::KeypointResultMap>;
extern template class ImageModel<postprocess::SegmentResultMap>;
extern template class ImageModel<postprocess::OcrResult>;

} // namespace hrt::model

#endif // HRT_MODEL_IMAGE_MODEL_HPP
