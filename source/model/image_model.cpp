// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/model/image_model.cpp
// Description: Implements the pooled image model pipeline.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "include/core/batch.hpp"
#include "include/core/config.hpp"
#include "include/postprocess/params.hpp"

#ifdef HRT_USE_DMA_BUFFER
#include "include/memory/dma_buffer_memory.hpp"
#endif

namespace hrt::model {

namespace {

constexpr std::size_t kMaxStageNum = 64;

auto check_options(const ImageModelOptions& options, std::string_view algo) -> void {
  for (const auto num : {options.preprocess, options.engine, options.postprocess})
    KZ_CHECK_MSG(num >= 1 && num <= kMaxStageNum, "ImageModel: stage counts of algo {} must be in [1, {}]", algo,
                 kMaxStageNum);
}

auto read_config(const hrtc::AlgoParam& algo) -> kitzoo::json::Reader {
  return kitzoo::json::Reader{algo.config};
}

// nullptr when the backend is not compiled in or the model does not load.
auto load_engine(const hrtc::EngineParam& param) -> UniquePtr<engine::Engine> {
  auto engine = engine::Engine::create(engine::engine_type_from(param.engine_type));
  if (engine == nullptr || !engine->initialize(param))
    return nullptr;
  return engine;
}

struct LoadedEngine {
  SharedPtr<core::Context> context;
  UniquePtr<engine::Engine> engine;
  const hrtc::EngineParam* param = nullptr;
};

// Device a preprocessor needs as the current context; CPU-side ones (RGA
// included) and auto, which follows the engine, need none.
auto device_of(preprocess::ImagePreprocessorType type) -> core::DeviceType {
  switch (type) {
  case preprocess::ImagePreprocessorType::Cuda:
    return core::DeviceType::Cuda;
  case preprocess::ImagePreprocessorType::Dvpp:
    return core::DeviceType::Ascend;
  default:
    return core::DeviceType::Cpu;
  }
}

// Device the configured preprocessor needs; `algo` passed check().
auto preprocess_device(const hrtc::AlgoParam& algo) -> core::DeviceType {
  const auto reader = read_config(algo);
  const auto section = reader.raw().value("preprocess", kitzoo::json::Json::object());
  return device_of(section.get<preprocess::ImagePreprocessorParam>().device);
}

// First engine of `algo`, in package order, that loads on `context`, or on a
// new context when `context` is null. The context serves both the engine and
// the preprocessor: a device engine fixes it, a CPU engine takes the
// preprocessor's device, and both on the CPU get a CPU context.
auto load_first(const hrtc::AlgoParam& algo, core::DeviceType preprocess_device,
                const SharedPtr<core::Context>& context) -> std::optional<LoadedEngine> {
  for (const auto& param : algo.engines) {
    const auto type = engine::engine_type_from(param.engine_type);
    if (type == engine::EngineType::Unknown)
      continue;
    const auto engine_device = engine::device_type_of(type);
    if (engine_device != core::DeviceType::Cpu && preprocess_device != core::DeviceType::Cpu &&
        engine_device != preprocess_device) {
      KZ_LOG_WARN("ImageModel: skipping {} of algo {}: it and the preprocessor need different devices",
                  param.engine_type, algo.name);
      continue;
    }
    const auto device = engine_device != core::DeviceType::Cpu ? engine_device : preprocess_device;
    SharedPtr<core::Context> target = context;
    if (target == nullptr) {
      target = core::Context::create(device);
      if (target == nullptr) {
        KZ_LOG_WARN("ImageModel: skipping {} of algo {}: its device is not available", param.engine_type, algo.name);
        continue;
      }
    } else if (device != core::DeviceType::Cpu && device != target->device_type()) {
      continue;
    }
    target->set_current();
    if (auto engine = load_engine(param))
      return LoadedEngine{std::move(target), std::move(engine), &param};
  }
  return std::nullopt;
}

// Problems in the config of `algo`; the "postprocess" keys besides type are
// left to a custom factory unless `builtin_postprocess`.
auto check(const hrtc::AlgoParam& algo, bool builtin_postprocess) -> core::Problems {
  const auto reader = read_config(algo);
  if (!reader.is_parse_success())
    return {"config: invalid JSON: " + reader.error_info()};
  const auto& config = reader.raw();
  core::Problems problems;
  const auto empty = kitzoo::json::Json::object();
  core::check_json(config, {{"preprocess", empty}, {"postprocess", empty}}, "config", {}, problems);
  if (!problems.empty())
    return problems;
  const auto preprocess = config.value("preprocess", empty);
  const auto postprocess = config.value("postprocess", empty);
  core::check_param<preprocess::ImagePreprocessorParam>(preprocess, "preprocess", {}, problems);
  const auto type = postprocess.find("type");
  if (type == postprocess.end() || !type->is_string()) {
    problems.emplace_back("postprocess.type: expected a string");
  } else if (builtin_postprocess) {
    const auto found = postprocess::check_postprocess_config(type->get_ref<const String&>(), postprocess);
    problems.insert(problems.end(), found.begin(), found.end());
  }
  return problems;
}

auto is_planar(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::PlanarRgb || format == core::ImageFormat::PlanarBgr ||
         format == core::ImageFormat::Gray;
}

auto is_yuv(core::ImageFormat format) -> bool {
  return format == core::ImageFormat::Nv12 || format == core::ImageFormat::Nv21 || format == core::ImageFormat::Yu12 ||
         format == core::ImageFormat::Yv12;
}

template <typename Result>
auto builtin_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<postprocess::Postprocessor<Result>> {
  if constexpr (std::is_same_v<Result, postprocess::DetectResultMap>)
    return postprocess::create_detect_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, postprocess::RotatedResultMap>)
    return postprocess::create_rotated_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, postprocess::ClassifyResult>)
    return postprocess::create_classify_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, postprocess::EmbeddingResult>)
    return postprocess::create_embedding_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, postprocess::SegmentResultMap>)
    return postprocess::create_segment_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, postprocess::OcrResult>)
    return postprocess::create_ocr_postprocessor(config, engine);
  else
    return postprocess::create_keypoint_postprocessor(type, config, engine);
}

// With `host_engine`, buffers are written on the host: they use its fast host memory when it offers one, and
// dma-bufs come from any heap, as no device with 32-bit DMA (RGA2) touches them.
auto make_buffers(const Vector<core::BufferProto>& protos, const engine::Engine* host_engine)
    -> std::optional<core::Buffers> {
  core::Buffers buffers;
  for (const auto& proto : protos) {
    SharedPtr<core::Buffer> buffer;
    if (auto memory = host_engine != nullptr ? host_engine->create_host_memory() : nullptr) {
      buffer = std::make_shared<core::Buffer>(std::move(memory), proto);
      if (!buffer->valid()) {
        KZ_LOG_WARN("ImageModel: engine host memory is unavailable; using the engine IO memory");
        buffer.reset();
      }
    }
#ifdef HRT_USE_DMA_BUFFER
    if (buffer == nullptr && host_engine != nullptr && proto.memory_type == core::MemoryType::DmaBuffer) {
      buffer = std::make_shared<core::Buffer>(
          std::make_shared<memory::DmaBufferMemory>(core::Shape{}, core::Stride{}, /*device32=*/false), proto);
      if (!buffer->valid())
        buffer.reset();
    }
#endif
    if (buffer == nullptr)
      buffer = std::make_shared<core::Buffer>(proto);
    if (!buffer->valid()) {
      KZ_LOG_ERROR("ImageModel: cannot allocate engine IO");
      return std::nullopt;
    }
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

auto read_algo(const kitzoo::json::Json& algo) -> hrtc::AlgoParam {
  auto param = hrtc::load_algo(algo);
  KZ_CHECK_MSG(param.has_value(), "ImageModel: cannot read the algo JSON");
  return std::move(*param);
}

} // namespace

auto task_of(std::string_view postprocess_type) -> std::optional<Task> {
  if (postprocess::create_detect(postprocess_type) != nullptr)
    return Task::Detect;
  if (postprocess::create_rotated(postprocess_type) != nullptr)
    return Task::Rotated;
  if (postprocess::create_embedding(postprocess_type) != nullptr)
    return Task::Embedding;
  if (postprocess::create_classify(postprocess_type) != nullptr)
    return Task::Classify;
  if (postprocess::create_keypoint(postprocess_type) != nullptr)
    return Task::Keypoint;
  if (postprocess::create_segment(postprocess_type) != nullptr)
    return Task::Segment;
  return std::nullopt;
}

auto task_of(const hrtc::AlgoParam& algo) -> std::optional<Task> {
  const auto reader = read_config(algo);
  if (!reader.is_parse_success() || !reader.raw().is_object())
    return std::nullopt;
  const auto& config = reader.raw();
  const auto postprocess = config.find("postprocess");
  if (postprocess == config.end() || !postprocess->is_object())
    return std::nullopt;
  const auto type = postprocess->find("type");
  return type != postprocess->end() && type->is_string() ? task_of(type->get_ref<const String&>()) : std::nullopt;
}

auto check_algo_config(const hrtc::AlgoParam& algo) -> Vector<String> {
  return check(algo, true);
}

template <typename Result>
ImageModel<Result>::ImageModel(SharedPtr<core::Context> context, const ImageModelOptions& options)
    : context_(std::move(context)), options_(options) {}

// Device buffers and engines are freed under the model's context.
template <typename Result>
ImageModel<Result>::~ImageModel() {
  context_->set_current();
}

template <typename Result>
auto ImageModel<Result>::create(const hrtc::AlgoParam& algo, SharedPtr<core::Context> context,
                                const ImageModelOptions& options, const PostprocessorFactory& factory)
    -> UniquePtr<ImageModel> {
  KZ_CHECK_MSG(context != nullptr, "ImageModel: algo {} needs a context", algo.name);
  check_options(options, algo.name);
  const auto problems = check(algo, !factory);
  KZ_CHECK_MSG(problems.empty(), "ImageModel: invalid config of algo {}: {}", algo.name, core::join(problems));
  auto loaded = load_first(algo, preprocess_device(algo), context);
  KZ_CHECK_MSG(loaded.has_value(), "ImageModel: no engine of algo {} runs on this context in this build", algo.name);
  UniquePtr<ImageModel> model(new ImageModel(std::move(loaded->context), options));
  model->build(algo, read_config(algo).raw(), factory, std::move(loaded->engine), *loaded->param);
  return model;
}

template <typename Result>
auto ImageModel<Result>::create(const hrtc::AlgoParam& algo, const ImageModelOptions& options,
                                const PostprocessorFactory& factory) -> UniquePtr<ImageModel> {
  check_options(options, algo.name);
  const auto problems = check(algo, !factory);
  KZ_CHECK_MSG(problems.empty(), "ImageModel: invalid config of algo {}: {}", algo.name, core::join(problems));
  auto loaded = load_first(algo, preprocess_device(algo), nullptr);
  KZ_CHECK_MSG(loaded.has_value(), "ImageModel: no engine of algo {} runs on this machine in this build", algo.name);
  UniquePtr<ImageModel> model(new ImageModel(std::move(loaded->context), options));
  model->build(algo, read_config(algo).raw(), factory, std::move(loaded->engine), *loaded->param);
  return model;
}

template <typename Result>
auto ImageModel<Result>::create(const kitzoo::json::Json& algo, SharedPtr<core::Context> context,
                                const ImageModelOptions& options, const PostprocessorFactory& factory)
    -> UniquePtr<ImageModel> {
  return create(read_algo(algo), std::move(context), options, factory);
}

template <typename Result>
auto ImageModel<Result>::create(const kitzoo::json::Json& algo, const ImageModelOptions& options,
                                const PostprocessorFactory& factory) -> UniquePtr<ImageModel> {
  return create(read_algo(algo), options, factory);
}

template <typename Result>
auto ImageModel<Result>::build(const hrtc::AlgoParam& algo, const kitzoo::json::Json& config,
                               const PostprocessorFactory& factory, UniquePtr<engine::Engine> first,
                               const hrtc::EngineParam& chosen) -> void {
  context_->set_current();
  preprocessor_param_ =
      config.value("preprocess", kitzoo::json::Json::object()).get<preprocess::ImagePreprocessorParam>();
  const auto preprocess_num = options_.preprocess;
  const auto engine_num = options_.engine;
  const auto postprocess_num = options_.postprocess;

  const auto& engine = *first;
  input_layout_ = engine.proto().input_layout;

  const auto postprocess_config = config.value("postprocess", kitzoo::json::Json::object());
  for (std::size_t i = 0; i < postprocess_num; ++i) {
    auto postprocessor = factory ? factory(postprocess_config, engine)
                                 : builtin_postprocessor<Result>(postprocess_config["type"].get_ref<const String&>(),
                                                                 postprocess_config, engine);
    KZ_CHECK_MSG(postprocessor != nullptr, "ImageModel: no postprocessor for algo {}", algo.name);
    postprocessor_pool_.add(std::make_unique<UniquePtr<postprocess::Postprocessor<Result>>>(std::move(postprocessor)));
  }

  // An input set is held from preprocessing through inference, an output set
  // from inference through postprocessing, and a stream for the whole call;
  // a call above the batch runs its batches one after another.
  // Outputs are read on the host; inputs are written there only by the CPU
  // preprocessor (Auto follows the engine's input memory).
  const auto* host_input_engine =
      preprocessor_param_.device == preprocess::ImagePreprocessorType::Cpu ? &engine : nullptr;
  // A batch above 1 has every input sample viewed in place and decoded from
  // the same sample of every output.
  const auto& inputs = engine.input_protos();
  KZ_CHECK_MSG(!inputs.empty() && !inputs.front().shape.empty() && inputs.front().shape[0] >= 1,
               "ImageModel: algo {} needs an image input with a static batch", algo.name);
  batch_ = static_cast<std::size_t>(inputs.front().shape[0]);
  if (batch_ > 1) {
    for (const auto& output : engine.output_protos()) {
      KZ_CHECK_MSG(!output.shape.empty() && output.shape[0] == inputs.front().shape[0],
                   "ImageModel: every output of algo {} needs the input batch {} as its leading dim", algo.name,
                   batch_);
    }
  }

  core::MemoryType input_memory = core::MemoryType::Cpu;
  for (std::size_t i = 0; i < preprocess_num + engine_num; ++i) {
    auto buffers = make_buffers(engine.input_protos(), host_input_engine);
    KZ_CHECK_MSG(buffers && !buffers->empty(), "ImageModel: cannot allocate the inputs of algo {}", algo.name);
    auto image = input_image(*buffers, 0);
    KZ_CHECK_MSG(image.has_value(), "ImageModel: the engine input of algo {} does not fit the preprocess config",
                 algo.name);
    input_memory = buffers->front()->memory_type();
    input_pool_.add(std::make_unique<core::Buffers>(std::move(*buffers)));
  }
  // VPC addresses whole DVPP buffers; samples at an offset are not verified.
  KZ_CHECK_MSG(batch_ == 1 || (input_memory != core::MemoryType::Dvpp &&
                               preprocessor_param_.device != preprocess::ImagePreprocessorType::Dvpp),
               "ImageModel: the dvpp preprocessor of algo {} supports only a batch of 1", algo.name);
  for (std::size_t i = 0; i < preprocess_num; ++i) {
    auto preprocessor = preprocess::ImagePreprocessor::create(preprocessor_param_.device, input_memory);
    KZ_CHECK_MSG(preprocessor != nullptr, "ImageModel: preprocess device of algo {} is not available in this build",
                 algo.name);
    preprocessor_pool_.add(std::make_unique<UniquePtr<preprocess::ImagePreprocessor>>(std::move(preprocessor)));
  }
  for (std::size_t i = 0; i < engine_num + postprocess_num; ++i) {
    auto buffers = make_buffers(engine.output_protos(), &engine);
    KZ_CHECK_MSG(buffers.has_value(), "ImageModel: cannot allocate the outputs of algo {}", algo.name);
    output_pool_.add(std::make_unique<core::Buffers>(std::move(*buffers)));
  }
  for (std::size_t i = 0; i < preprocess_num + engine_num + postprocess_num; ++i)
    stream_pool_.add(std::make_unique<UniquePtr<core::Stream>>(context_->create_stream()));

  engine_pool_.add(std::make_unique<UniquePtr<engine::Engine>>(std::move(first)));
  for (std::size_t i = 1; i < engine_num; ++i) {
    auto extra = load_engine(chosen);
    KZ_CHECK_MSG(extra != nullptr, "ImageModel: cannot load engine {} of algo {}", i, algo.name);
    engine_pool_.add(std::make_unique<UniquePtr<engine::Engine>>(std::move(extra)));
  }
}

template <typename Result>
auto ImageModel<Result>::create_input_image(const core::Buffer& input) const -> std::optional<core::Image> {
  const auto format = preprocessor_param_.dst_image_format;
  const auto& shape = input.shape();
  const auto nchw = input_layout_ == engine::DataLayout::Nchw;
  if (shape.size() != 4 || shape[0] != 1 || (!is_yuv(format) && is_planar(format) != nchw)) {
    KZ_LOG_ERROR("ImageModel: image input must be batch-1 4-D and match dst_image_format's layout");
    return std::nullopt;
  }
  core::Size size = nchw ? core::Size{static_cast<int>(shape[3]), static_cast<int>(shape[2])}
                         : core::Size{static_cast<int>(shape[2]), static_cast<int>(shape[1])};
  if (is_yuv(format))
    size.height = size.height * 2 / 3;
  const auto& stride = input.stride();
  core::Stride image_stride{stride[0]};
  image_stride.insert(image_stride.end(), stride.begin() + 2, stride.end());
  core::Image image(input.memory()->ref(), format, input.data_type(), size, image_stride);
  if (!image.valid()) {
    KZ_LOG_ERROR("ImageModel: the engine input does not fit a {}x{} image", size.width, size.height);
    return std::nullopt;
  }
  return image;
}

template <typename Result>
auto ImageModel<Result>::input_image(const core::Buffers& input, std::size_t index) const
    -> std::optional<core::Image> {
  // A batch of 1 keeps the engine's own input memory, as RGA and VPC write it directly.
  if (batch_ == 1)
    return create_input_image(*input.front());
  const auto sample = core::batch_sample(*input.front(), index, true);
  if (sample == nullptr)
    return std::nullopt;
  return create_input_image(*sample);
}

template <typename Result>
auto ImageModel<Result>::process_sync(const core::Image& src, const core::Rect& src_rect) -> std::optional<Result> {
  auto results = run({&src, 1}, {&src_rect, 1});
  if (!results)
    return std::nullopt;
  return std::move(results->front());
}

template <typename Result>
auto ImageModel<Result>::process_sync(const Vector<core::Image>& srcs, const Vector<core::Rect>& src_rects)
    -> std::optional<Vector<Result>> {
  return run(srcs, src_rects);
}

template <typename Result>
auto ImageModel<Result>::run(std::span<const core::Image> srcs, std::span<const core::Rect> src_rects)
    -> std::optional<Vector<Result>> {
  KZ_CHECK_RETURN(!srcs.empty(), std::nullopt, "ImageModel: got no images");
  KZ_CHECK_RETURN(src_rects.empty() || src_rects.size() == srcs.size(), std::nullopt,
                  "ImageModel: got {} source rects for {} images", src_rects.size(), srcs.size());
  for (const auto& src : srcs) {
    KZ_CHECK_RETURN(src.valid() && src.size().width > 0 && src.size().height > 0, std::nullopt,
                    "ImageModel: invalid source image");
  }

  context_->set_current();
  auto stream_slot = stream_pool_.acquire();
  Vector<Result> results;
  results.reserve(srcs.size());
  for (std::size_t first = 0; first < srcs.size(); first += batch_) {
    const auto count = std::min(batch_, srcs.size() - first);
    const auto rects = src_rects.empty() ? src_rects : src_rects.subspan(first, count);
    if (!run_batch(srcs.subspan(first, count), rects, stream_slot->get(), results))
      return std::nullopt;
  }
  return results;
}

template <typename Result>
auto ImageModel<Result>::run_batch(std::span<const core::Image> srcs, std::span<const core::Rect> src_rects,
                                   core::Stream* stream, Vector<Result>& results) -> bool {
  // Only the given images are preprocessed, each straight into its own sample;
  // the samples after them keep stale data and are not decoded.
  Vector<preprocess::ImageTransform> transforms;
  transforms.reserve(srcs.size());
  decltype(output_pool_.acquire()) output;
  {
    auto input = input_pool_.acquire();
    {
      auto preprocessor = preprocessor_pool_.acquire();
      for (std::size_t i = 0; i < srcs.size(); ++i) {
        // Rebuilt per call: preprocess settings may change between frames of a stream.
        auto image = input_image(*input, i);
        const auto src_rect = src_rects.empty() ? core::Rect{} : src_rects[i];
        auto transform =
            image ? (*preprocessor)->process(*image, srcs[i], src_rect, preprocessor_param_, stream) : std::nullopt;
        if (!transform)
          break;
        transforms.push_back(*transform);
      }
      // Scratch buffers may be reused on another stream only after this completes.
      stream->synchronize();
    }
    if (transforms.size() != srcs.size())
      return false;

    output = output_pool_.acquire();
    auto engine = engine_pool_.acquire();
    const auto ran = (*engine)->set_inputs(*input) && (*engine)->set_outputs(*output) && (*engine)->inference(stream);
    // The device must be done with the inputs before they go back to the pool, also after a failure.
    stream->synchronize();
    if (!ran)
      return false;
  }
  Vector<Result> batch;
  auto postprocessor = postprocessor_pool_.acquire();
  if (!(*postprocessor)->process(batch, *output, transforms, stream))
    return false;
  std::move(batch.begin(), batch.end(), std::back_inserter(results));
  return true;
}

template class ImageModel<postprocess::DetectResultMap>;
template class ImageModel<postprocess::RotatedResultMap>;
template class ImageModel<postprocess::ClassifyResult>;
template class ImageModel<postprocess::EmbeddingResult>;
template class ImageModel<postprocess::KeypointResultMap>;
template class ImageModel<postprocess::SegmentResultMap>;
template class ImageModel<postprocess::OcrResult>;

} // namespace hrt::model
