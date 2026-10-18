// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/model/ocr_model.cpp
// Description: Implements sequential OCR with one reusable engine per stage.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/model/ocr_model.hpp>
#include <hrt/preprocess/text_cropper.hpp>

#include <kitzoo/log/logger.hpp>

#include <cmath>

#include "include/core/config.hpp"

namespace hrt::model {

namespace {

auto check_stage(const hrtc::AlgoParam& algo, std::string_view stage) -> core::Problems {
  auto problems = check_config(algo);
  if (!problems.empty())
    return problems;
  const kitzoo::json::Reader reader{algo.config};
  const auto& post = reader.raw()["postprocess"];
  if (post.value("type", String{}) != "Ocr" || post.value("stage", String{}) != stage)
    problems.emplace_back("postprocess: expected Ocr " + String(stage));
  const auto param =
      reader.raw().value("preprocess", kitzoo::json::Json::object()).get<preprocess::ImagePreprocessorParam>();
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(param.mean[i]) || !std::isfinite(param.std[i]) || param.std[i] <= 0)
      problems.emplace_back("preprocess: mean must be finite and std positive and finite");
  }
  if (algo.engines.empty())
    problems.emplace_back("engine: no backends");
  return problems;
}

auto preprocess_device(preprocess::ImagePreprocessorType type) -> core::DeviceType {
  switch (type) {
  case preprocess::ImagePreprocessorType::Cuda:
    return core::DeviceType::Cuda;
  case preprocess::ImagePreprocessorType::Dvpp:
    return core::DeviceType::Ascend;
  default:
    return core::DeviceType::Cpu;
  }
}

struct Stage {
  SharedPtr<core::Context> context;
  UniquePtr<engine::Engine> engine;
  UniquePtr<preprocess::ImagePreprocessor> preprocessor;
  UniquePtr<postprocess::OcrPostprocessor> postprocessor;
  UniquePtr<core::Stream> stream;
  core::Buffers inputs;
  core::Buffers outputs;
  std::optional<core::Image> image;
  preprocess::ImagePreprocessorParam param;

  ~Stage() {
    if (context != nullptr)
      context->make_current();
  }

  auto load(const hrtc::AlgoParam& algo) -> void {
    const kitzoo::json::Reader reader{algo.config};
    param = reader.raw().value("preprocess", kitzoo::json::Json::object()).get<preprocess::ImagePreprocessorParam>();
    const auto pre_device = preprocess_device(param.device);
    for (const auto& backend : algo.engines) {
      const auto type = engine::engine_type_from(backend.engine_type);
      if (type == engine::EngineType::Unknown)
        continue;
      const auto device = engine::device_type_of(type);
      if (device != core::DeviceType::Cpu && pre_device != core::DeviceType::Cpu && device != pre_device)
        continue;
      context = core::Context::create(device == core::DeviceType::Cpu ? pre_device : device);
      if (context == nullptr)
        continue;
      context->make_current();
      engine = engine::Engine::create(type);
      if (engine != nullptr && engine->initialize(backend))
        break;
      engine.reset();
    }
    KZ_CHECK_MSG(engine != nullptr, "OcrModel: no usable engine for {}", algo.name);
    KZ_CHECK_MSG(engine->input_protos().size() == 1, "OcrModel: expected one image input");
    for (const auto& proto : engine->input_protos()) {
      inputs.push_back(std::make_shared<core::Buffer>(proto));
      KZ_CHECK_MSG(inputs.back()->valid(), "OcrModel: cannot allocate input");
    }
    for (const auto& proto : engine->output_protos()) {
      outputs.push_back(std::make_shared<core::Buffer>(proto));
      KZ_CHECK_MSG(outputs.back()->valid(), "OcrModel: cannot allocate output");
    }
    const auto& input = *inputs[0];
    const auto& shape = input.shape();
    const auto nchw = engine->proto().input_layout == engine::DataLayout::Nchw;
    const auto planar = param.dst_image_format == core::ImageFormat::PlanarBgr ||
                        param.dst_image_format == core::ImageFormat::PlanarRgb;
    const auto inter =
        param.dst_image_format == core::ImageFormat::InterBgr || param.dst_image_format == core::ImageFormat::InterRgb;
    KZ_CHECK_MSG(shape.size() == 4 && shape[0] == 1 && shape[nchw ? 1 : 3] == 3 &&
                     ((nchw && planar) || (!nchw && inter)) && shape[nchw ? 2 : 1] > 0 &&
                     shape[nchw ? 2 : 1] <= 32768 && shape[nchw ? 3 : 2] > 0 && shape[nchw ? 3 : 2] <= 32768,
                 "OcrModel: expected a batch-1 three-channel image matching the engine layout");
    const core::Size size{static_cast<int>(shape[nchw ? 3 : 2]), static_cast<int>(shape[nchw ? 2 : 1])};
    const auto& stride = input.stride();
    core::Stride image_stride{stride[0]};
    image_stride.insert(image_stride.end(), stride.begin() + 2, stride.end());
    image.emplace(input.memory()->ref(), param.dst_image_format, input.data_type(), size, image_stride);
    preprocessor = preprocess::ImagePreprocessor::create(param.device, input.memory_type());
    stream = context->create_stream();
    KZ_CHECK_MSG(image->valid() && preprocessor != nullptr && stream != nullptr, "OcrModel: unusable preprocessing");
    postprocessor = postprocess::create_ocr_postprocessor(reader.raw()["postprocess"], *engine);
    KZ_CHECK_MSG(engine->set_inputs(inputs) && engine->set_outputs(outputs), "OcrModel: cannot bind IO");
  }

  auto run(const core::Image& src) -> std::optional<postprocess::OcrResult> {
    context->make_current();
    const auto transform = preprocessor->process(*image, src, {}, param, stream.get());
    stream->synchronize();
    if (!transform)
      return std::nullopt;
    const auto success = engine->inference(stream.get());
    stream->synchronize();
    if (!success)
      return std::nullopt;
    postprocess::OcrResult result;
    if (!postprocessor->process(result, outputs, *transform, stream.get()))
      return std::nullopt;
    return result;
  }
};

} // namespace

struct OcrModel::Impl {
  Stage detection;
  Stage recognition;
  preprocess::TextCropper cropper;
  UniquePtr<preprocess::ImagePreprocessor> host_preprocessor =
      preprocess::ImagePreprocessor::create(preprocess::ImagePreprocessorType::Cpu);
  core::Image host{core::MemoryType::Cpu, core::ImageFormat::InterBgr};
};

auto check_ocr_config(const hrtc::AlgoParam& detection, const hrtc::AlgoParam& recognition) -> Vector<String> {
  auto problems = check_stage(detection, "detection");
  const auto rec = check_stage(recognition, "recognition");
  problems.insert(problems.end(), rec.begin(), rec.end());
  return problems;
}

OcrModel::OcrModel() : impl_(std::make_unique<Impl>()) {}

OcrModel::~OcrModel() = default;

auto OcrModel::create(const hrtc::AlgoParam& detection, const hrtc::AlgoParam& recognition) -> UniquePtr<OcrModel> {
  const auto problems = check_ocr_config(detection, recognition);
  KZ_CHECK_MSG(problems.empty(), "OcrModel: invalid config: {}", core::join(problems));
  UniquePtr<OcrModel> model(new OcrModel);
  model->impl_->detection.load(detection);
  model->impl_->recognition.load(recognition);
  return model;
}

auto OcrModel::create(const hrtc::Package& package, std::string_view detection, std::string_view recognition)
    -> UniquePtr<OcrModel> {
  const auto* det = package.find(detection);
  const auto* rec = package.find(recognition);
  KZ_CHECK_MSG(det != nullptr && rec != nullptr, "OcrModel: detection or recognition algo is missing");
  return create(*det, *rec);
}

auto OcrModel::process_sync(const core::Image& src) -> std::optional<postprocess::OcrResult> {
  KZ_CHECK_RETURN(src.valid() && src.size().width > 0 && src.size().height > 0 && src.size().width <= 32768 &&
                      src.size().height <= 32768,
                  std::nullopt, "OcrModel: invalid source image");
  auto result = impl_->detection.run(src);
  if (!result || result->empty())
    return result;
  const core::Image* source = &src;
  if (!src.memory()->have_cpu() || src.data_type() != core::DataType::Uint8 ||
      (src.format() != core::ImageFormat::InterBgr && src.format() != core::ImageFormat::InterRgb)) {
    impl_->detection.context->make_current();
    KZ_CHECK_RETURN(impl_->host.reshape(src.size()), std::nullopt, "OcrModel: cannot allocate host image");
    preprocess::ImagePreprocessorParam param;
    param.dst_image_format = core::ImageFormat::InterBgr;
    param.normalize_type = preprocess::NormalizeType::None;
    if (!impl_->host_preprocessor->process(impl_->host, src, {}, param))
      return std::nullopt;
    source = &impl_->host;
  }
  for (auto& text : *result) {
    auto crop = impl_->cropper.process(*source, text.rect);
    if (!crop)
      return std::nullopt;
    auto recognition = impl_->recognition.run(*crop);
    if (!recognition || recognition->size() != 1)
      return std::nullopt;
    text.text = std::move(recognition->front().text);
    text.recognition_score = recognition->front().recognition_score;
  }
  return result;
}

} // namespace hrt::model
