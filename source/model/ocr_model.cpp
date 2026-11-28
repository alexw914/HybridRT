// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/model/ocr_model.cpp
// Description: Implements sequential OCR over detection and recognition ImageModels.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/model/ocr_model.hpp>

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

// The box clipped to the source image and truncated to whole pixels like PP-OCR;
// degenerate boxes smaller than 2 x 2 pixels are skipped.
auto text_region(const core::Rect2f& rect, core::Size size) -> std::optional<core::Rect> {
  if (!std::isfinite(rect.x) || !std::isfinite(rect.y) || !std::isfinite(rect.width) || !std::isfinite(rect.height))
    return std::nullopt;
  const auto x0 = std::clamp(static_cast<int>(rect.x), 0, size.width);
  const auto y0 = std::clamp(static_cast<int>(rect.y), 0, size.height);
  const auto x1 = std::clamp(static_cast<int>(rect.x + rect.width), 0, size.width);
  const auto y1 = std::clamp(static_cast<int>(rect.y + rect.height), 0, size.height);
  const core::Rect region{x0, y0, x1 - x0, y1 - y0};
  if (region.width < 2 || region.height < 2)
    return std::nullopt;
  return region;
}

auto edge_length(const core::Point2f& a, const core::Point2f& b) -> float {
  return std::hypot(a.x - b.x, a.y - b.y);
}

// True when `points` holds a valid tilted quad rather than an axis-aligned box.
auto is_rotated_quad(const Array<core::Point2f, 4>& points) -> bool {
  for (const auto& point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
      return false;
  }
  const auto w = std::max(edge_length(points[0], points[1]), edge_length(points[2], points[3]));
  const auto h = std::max(edge_length(points[0], points[3]), edge_length(points[1], points[2]));
  if (w < 2.0F || h < 2.0F)
    return false;
  return std::abs(points[0].y - points[1].y) >= 1.0F || std::abs(points[0].x - points[3].x) >= 1.0F;
}

auto is_host_bgr(const core::Image& image) -> bool {
  return image.memory()->have_cpu() && image.data_type() == core::DataType::Uint8 &&
         (image.format() == core::ImageFormat::InterBgr || image.format() == core::ImageFormat::InterRgb);
}

// Memory the preprocessors of a `device` context read without staging it.
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

auto check_context_device(const hrtc::AlgoParam& algo, const core::Context& context, std::string_view det_name)
    -> bool {
  const auto device = engine::device_type_of(engine::engine_type_from(algo.engine.engine_type));
  KZ_CHECK_RETURN(device == core::DeviceType::Cpu || device == context.device_type(), false,
                  "OcrModel: algo {} engine does not run on the device of algo {}", algo.name, det_name);
  return true;
}

} // namespace

struct OcrModel::Impl {
  UniquePtr<ImageModel<postprocess::OcrResult>> detection;
  UniquePtr<ImageModel<postprocess::OcrResult>> recognition;
  UniquePtr<preprocess::ImagePreprocessor> host_preprocessor =
      preprocess::ImagePreprocessor::create(preprocess::ImagePreprocessorType::Cpu);
  Vector<core::Image> host_images; // Converted host BGR copies for non-host batch sources.
  Vector<cv::Mat> crops;           // Warped or rotated crops owned for the duration of the call.

  // src when it is host uint8 BGR/RGB, else src converted into `host_images[slot]`.
  auto host_image(const core::Image& src, std::size_t slot) -> const core::Image*;

  // PP-OCR perspective warp of a tilted 4-point quad, rotating tall crops 90 degrees CCW.
  auto warp_quad(const core::Image& host_src, const Array<core::Point2f, 4>& points, cv::Mat& out)
      -> std::optional<core::Image>;

  // PP-OCR turns crops at least 1.5 times taller than wide counterclockwise into lines.
  auto rotate_crop(const core::Image& host_src, const core::Rect& region, cv::Mat& out) -> std::optional<core::Image>;
};

auto OcrModel::Impl::host_image(const core::Image& src, std::size_t slot) -> const core::Image* {
  if (is_host_bgr(src))
    return &src;
  detection->context()->set_current();
  auto& host = host_images[slot];
  KZ_CHECK_RETURN(host.reshape(src.size()), nullptr, "OcrModel: cannot allocate host image");
  preprocess::ImagePreprocessorParam param;
  param.dst_image_format = core::ImageFormat::InterBgr;
  param.normalize_type = preprocess::NormalizeType::None;
  return host_preprocessor->process(host, src, {}, param) ? &host : nullptr;
}

auto OcrModel::Impl::warp_quad(const core::Image& host_src, const Array<core::Point2f, 4>& points, cv::Mat& out)
    -> std::optional<core::Image> {
  auto* data = static_cast<std::uint8_t*>(const_cast<void*>(host_src.memory()->cpu()));
  KZ_CHECK_RETURN(data != nullptr, std::nullopt, "OcrModel: cannot map host image");
  const auto crop_w =
      static_cast<int>(std::lround(std::max(edge_length(points[0], points[1]), edge_length(points[2], points[3]))));
  const auto crop_h =
      static_cast<int>(std::lround(std::max(edge_length(points[0], points[3]), edge_length(points[1], points[2]))));
  if (crop_w < 2 || crop_h < 2)
    return std::nullopt;
  const cv::Mat image(host_src.size().height, host_src.size().width, CV_8UC3, data, host_src.stride()[1]);
  const Array<cv::Point2f, 4> src_pts{
      {{points[0].x, points[0].y}, {points[1].x, points[1].y}, {points[2].x, points[2].y}, {points[3].x, points[3].y}}};
  const Array<cv::Point2f, 4> dst_pts{{{0.0F, 0.0F},
                                       {static_cast<float>(crop_w), 0.0F},
                                       {static_cast<float>(crop_w), static_cast<float>(crop_h)},
                                       {0.0F, static_cast<float>(crop_h)}}};
  const auto matrix = cv::getPerspectiveTransform(src_pts.data(), dst_pts.data());
  cv::warpPerspective(image, out, matrix, cv::Size(crop_w, crop_h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
  if (out.rows >= out.cols * 1.5)
    cv::rotate(out, out, cv::ROTATE_90_COUNTERCLOCKWISE);
  return core::Image::wrap(core::MemoryType::Cpu, out.data, host_src.format(), {out.cols, out.rows},
                           core::DataType::Uint8, out.step);
}

auto OcrModel::Impl::rotate_crop(const core::Image& host_src, const core::Rect& region, cv::Mat& out)
    -> std::optional<core::Image> {
  auto* data = static_cast<std::uint8_t*>(const_cast<void*>(host_src.memory()->cpu()));
  KZ_CHECK_RETURN(data != nullptr, std::nullopt, "OcrModel: cannot map host image");
  const cv::Mat image(host_src.size().height, host_src.size().width, CV_8UC3, data, host_src.stride()[1]);
  cv::rotate(image(cv::Rect(region.x, region.y, region.width, region.height)), out, cv::ROTATE_90_COUNTERCLOCKWISE);
  return core::Image::wrap(core::MemoryType::Cpu, out.data, host_src.format(), {out.cols, out.rows},
                           core::DataType::Uint8, out.step);
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
  KZ_CHECK_RETURN(problems.empty(), nullptr, "OcrModel: invalid config: {}", core::join(problems));
  UniquePtr<OcrModel> model(new OcrModel);
  model->impl_->detection = ImageModel<postprocess::OcrResult>::create(detection);
  if (model->impl_->detection == nullptr)
    return nullptr;
  const auto& context = model->impl_->detection->context();
  if (!check_context_device(recognition, *context, detection.name))
    return nullptr;
  model->impl_->recognition = ImageModel<postprocess::OcrResult>::create(recognition, context);
  if (model->impl_->recognition == nullptr)
    return nullptr;
  return model;
}
auto OcrModel::create(const hrtc::Package& package, std::string_view detection, std::string_view recognition)
    -> UniquePtr<OcrModel> {
  const auto* det = package.find(detection);
  const auto* rec = package.find(recognition);
  KZ_CHECK_RETURN(det != nullptr && rec != nullptr, nullptr, "OcrModel: detection or recognition algo is missing");
  return create(*det, *rec);
}

auto OcrModel::process_sync(const core::Image& src) -> std::optional<postprocess::OcrResult> {
  auto results = process_sync(Vector<core::Image>{src});
  if (!results || results->size() != 1)
    return std::nullopt;
  return std::move(results->front());
}
auto OcrModel::process_sync(const Vector<core::Image>& srcs) -> std::optional<Vector<postprocess::OcrResult>> {
  if (srcs.empty())
    return Vector<postprocess::OcrResult>{};
  auto detections = impl_->detection->process_sync(srcs);
  if (!detections || detections->size() != srcs.size())
    return std::nullopt;

  // Prepare every detected line across the batch: tilted quads are perspective-warped,
  // axis-aligned wide lines reference the source ROI, and tall lines are rotated 90 deg CCW.
  if (impl_->host_images.size() < srcs.size()) {
    impl_->host_images.reserve(srcs.size());
    while (impl_->host_images.size() < srcs.size())
      impl_->host_images.emplace_back(core::MemoryType::Cpu, core::ImageFormat::InterBgr);
  }
  Vector<postprocess::OcrResult> outputs(srcs.size());
  Vector<Pair<std::size_t, std::size_t>> owners;
  Vector<core::Image> lines;
  Vector<core::Rect> regions;
  Vector<const core::Image*> hosts(srcs.size(), nullptr);
  std::size_t crop_count = 0;

  for (std::size_t img = 0; img < srcs.size(); ++img) {
    const auto& src = srcs[img];
    auto& det = (*detections)[img];
    if (det.empty())
      continue;
    const core::Image* source = &src;
    if (!is_host_bgr(src) && !is_native(impl_->detection->context()->device_type(), src.memory()->memory_type())) {
      hosts[img] = impl_->host_image(src, img);
      if (hosts[img] == nullptr)
        return std::nullopt;
      source = hosts[img];
    }
    outputs[img].reserve(det.size());
    for (auto& text : det) {
      if (is_rotated_quad(text.points)) {
        if (hosts[img] == nullptr)
          hosts[img] = impl_->host_image(src, img);
        if (impl_->crops.size() <= crop_count)
          impl_->crops.resize(crop_count + 1);
        const auto warped =
            hosts[img] != nullptr ? impl_->warp_quad(*hosts[img], text.points, impl_->crops[crop_count]) : std::nullopt;
        if (!warped)
          continue;
        ++crop_count;
        owners.emplace_back(img, outputs[img].size());
        outputs[img].push_back(std::move(text));
        lines.push_back(*warped);
        regions.emplace_back();
        continue;
      }
      const auto region = text_region(text.rect, src.size());
      if (!region)
        continue;
      if (region->height < region->width * 1.5) {
        owners.emplace_back(img, outputs[img].size());
        outputs[img].push_back(std::move(text));
        lines.push_back(*source);
        regions.push_back(*region);
        continue;
      }
      if (hosts[img] == nullptr)
        hosts[img] = impl_->host_image(src, img);
      if (impl_->crops.size() <= crop_count)
        impl_->crops.resize(crop_count + 1);
      const auto rotated =
          hosts[img] != nullptr ? impl_->rotate_crop(*hosts[img], *region, impl_->crops[crop_count++]) : std::nullopt;
      if (!rotated)
        return std::nullopt;
      owners.emplace_back(img, outputs[img].size());
      outputs[img].push_back(std::move(text));
      lines.push_back(*rotated);
      regions.emplace_back();
    }
  }
  if (lines.empty())
    return outputs;

  auto recognitions = impl_->recognition->process_sync(lines, regions);
  if (!recognitions || recognitions->size() != lines.size())
    return std::nullopt;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    auto& recognition = (*recognitions)[i];
    if (recognition.size() != 1)
      return std::nullopt;
    auto& target = outputs[owners[i].first][owners[i].second];
    target.text = std::move(recognition.front().text);
    target.recognition_score = recognition.front().recognition_score;
  }
  return outputs;
}

auto OcrModel::detection_engine_type() const -> engine::EngineType {
  return impl_->detection->engine_type();
}

auto OcrModel::recognition_engine_type() const -> engine::EngineType {
  return impl_->recognition->engine_type();
}

auto OcrModel::detection_input_size() const -> core::Size {
  return impl_->detection->input_size();
}

auto OcrModel::recognition_input_size() const -> core::Size {
  return impl_->recognition->input_size();
}

auto OcrModel::recognition_batch() const -> std::size_t {
  return impl_->recognition->batch();
}

} // namespace hrt::model
