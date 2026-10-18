// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/model/ocr_model.cpp
// Description: Implements sequential OCR over one detection and one recognition ImageModel.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/model/ocr_model.hpp>
#include <hrt/preprocess/text_cropper.hpp>

#include <kitzoo/log/logger.hpp>

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
  return problems;
}

} // namespace

struct OcrModel::Impl {
  UniquePtr<ImageModel<postprocess::OcrResult>> detection;
  UniquePtr<ImageModel<postprocess::OcrResult>> recognition;
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
  model->impl_->detection = ImageModel<postprocess::OcrResult>::create(detection);
  model->impl_->recognition = ImageModel<postprocess::OcrResult>::create(recognition);
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
  auto result = impl_->detection->process_sync(src);
  if (!result || result->empty())
    return result;
  const core::Image* source = &src;
  if (!src.memory()->have_cpu() || src.data_type() != core::DataType::Uint8 ||
      (src.format() != core::ImageFormat::InterBgr && src.format() != core::ImageFormat::InterRgb)) {
    impl_->detection->context()->make_current();
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
    auto recognition = impl_->recognition->process_sync(*crop);
    if (!recognition || recognition->size() != 1)
      return std::nullopt;
    text.text = std::move(recognition->front().text);
    text.recognition_score = recognition->front().recognition_score;
  }
  return result;
}

} // namespace hrt::model
