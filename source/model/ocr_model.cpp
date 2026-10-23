// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/model/ocr_model.cpp
// Description: Implements sequential OCR over one detection and one recognition ImageModel.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/model/ocr_model.hpp>

#include <kitzoo/log/logger.hpp>

#include <cmath>
#include <opencv2/core.hpp>

#include "include/core/config.hpp"

namespace hrt::model {

namespace {

auto check_stage(const hrtc::AlgoParam& algo, std::string_view stage) -> core::Problems {
  auto problems = check_algo_config(algo);
  if (!problems.empty())
    return problems;
  const kitzoo::json::Reader reader{algo.config};
  const auto& post = reader.raw()["postprocess"];
  if (post.value("type", String{}) != "Ocr" || post.value("stage", String{}) != stage)
    problems.emplace_back("postprocess: expected Ocr " + String(stage));
  return problems;
}

// The box truncated to whole pixels like PP-OCR; it must lie inside the source and span 2 x 2 pixels.
auto text_region(const core::Rect2f& rect, core::Size size) -> std::optional<core::Rect> {
  KZ_CHECK_RETURN(std::isfinite(rect.x) && std::isfinite(rect.y) && std::isfinite(rect.width) &&
                      std::isfinite(rect.height) && rect.x >= 0 && rect.y >= 0 &&
                      rect.x + rect.width <= static_cast<float>(size.width) &&
                      rect.y + rect.height <= static_cast<float>(size.height),
                  std::nullopt, "OcrModel: text box leaves the source image");
  const core::Rect region{static_cast<int>(rect.x), static_cast<int>(rect.y), static_cast<int>(rect.width),
                          static_cast<int>(rect.height)};
  KZ_CHECK_RETURN(region.width >= 2 && region.height >= 2, std::nullopt, "OcrModel: degenerate text box");
  return region;
}

auto is_host_bgr(const core::Image& image) -> bool {
  return image.memory()->have_cpu() && image.data_type() == core::DataType::Uint8 &&
         (image.format() == core::ImageFormat::InterBgr || image.format() == core::ImageFormat::InterRgb);
}

// Memory the preprocessors of a `device` context read without staging it.
// ponytail: RGA also counts as Cpu here but stages host memory, so host NV12 into RKNN is copied per line;
// tell RGA apart if that input matters.
auto is_native(core::DeviceType device, core::MemoryType type) -> bool {
  switch (device) {
  case core::DeviceType::Cpu:
    return type == core::MemoryType::Cpu || type == core::MemoryType::DmaBuffer;
  case core::DeviceType::Cuda:
    return type == core::MemoryType::Cuda;
  case core::DeviceType::Ascend:
    return type == core::MemoryType::Dvpp;
  }
  return false;
}

} // namespace

struct OcrModel::Impl {
  UniquePtr<ImageModel<postprocess::OcrResult>> detection;
  UniquePtr<ImageModel<postprocess::OcrResult>> recognition;
  UniquePtr<preprocess::ImagePreprocessor> host_preprocessor =
      preprocess::ImagePreprocessor::create(preprocess::ImagePreprocessorType::Cpu);
  core::Image host{core::MemoryType::Cpu, core::ImageFormat::InterBgr};
  cv::Mat rotated;

  // src when it is host uint8 BGR/RGB, else src converted into `host`.
  auto host_image(const core::Image& src) -> const core::Image*;

  // PP-OCR turns crops at least 1.5 times taller than wide counterclockwise into lines; the
  // preprocessors cannot rotate, so this runs on the host and is valid until the next call.
  auto rotate(const core::Image& host_src, const core::Rect& region) -> std::optional<core::Image>;
};

auto OcrModel::Impl::host_image(const core::Image& src) -> const core::Image* {
  if (is_host_bgr(src))
    return &src;
  detection->context()->set_current();
  KZ_CHECK_RETURN(host.reshape(src.size()), nullptr, "OcrModel: cannot allocate host image");
  preprocess::ImagePreprocessorParam param;
  param.dst_image_format = core::ImageFormat::InterBgr;
  param.normalize_type = preprocess::NormalizeType::None;
  return host_preprocessor->process(host, src, {}, param) ? &host : nullptr;
}

auto OcrModel::Impl::rotate(const core::Image& host_src, const core::Rect& region) -> std::optional<core::Image> {
  auto* data = static_cast<std::uint8_t*>(const_cast<void*>(host_src.memory()->cpu()));
  KZ_CHECK_RETURN(data != nullptr, std::nullopt, "OcrModel: cannot map host image");
  const cv::Mat image(host_src.size().height, host_src.size().width, CV_8UC3, data, host_src.stride()[1]);
  cv::rotate(image(cv::Rect(region.x, region.y, region.width, region.height)), rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
  return core::Image::wrap(core::MemoryType::Cpu, rotated.data, host_src.format(), {rotated.cols, rotated.rows},
                           core::DataType::Uint8, rotated.step);
}

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
  // Recognition shares the detection context, so both stages read the same source without staging it per line.
  // CPU engines run under any context, so engines of other devices are dropped first.
  const auto& context = model->impl_->detection->context();
  auto same_device = recognition;
  std::erase_if(same_device.engines, [&](const auto& param) -> bool {
    return engine::device_type_of(engine::engine_type_from(param.engine_type)) != context->device_type();
  });
  KZ_CHECK_MSG(!same_device.engines.empty(), "OcrModel: algo {} has no engine on the device of algo {}",
               recognition.name, detection.name);
  model->impl_->recognition = ImageModel<postprocess::OcrResult>::create(same_device, context);
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
  // Lines read their region of src when the stages' device takes src as is or stages only that region (host
  // BGR/RGB); other sources are converted to host BGR once. Tall lines are rotated on the host.
  const core::Image* host = nullptr;
  const core::Image* source = &src;
  if (!is_host_bgr(src) && !is_native(impl_->detection->context()->device_type(), src.memory()->memory_type())) {
    host = impl_->host_image(src);
    if (host == nullptr)
      return std::nullopt;
    source = host;
  }
  for (auto& text : *result) {
    const auto region = text_region(text.rect, src.size());
    if (!region)
      return std::nullopt;
    std::optional<postprocess::OcrResult> recognition;
    if (region->height < region->width * 1.5) {
      recognition = impl_->recognition->process_sync(*source, *region);
    } else {
      host = host != nullptr ? host : impl_->host_image(src);
      const auto rotated = host != nullptr ? impl_->rotate(*host, *region) : std::nullopt;
      if (!rotated)
        return std::nullopt;
      recognition = impl_->recognition->process_sync(*rotated);
    }
    if (!recognition || recognition->size() != 1)
      return std::nullopt;
    text.text = std::move(recognition->front().text);
    text.recognition_score = recognition->front().recognition_score;
  }
  return result;
}

} // namespace hrt::model
