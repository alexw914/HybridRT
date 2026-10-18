// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/hrt.cpp
// Description: Implements the C API over packages, image models and flattened
//              results.
// -----------------------------------------------------------------------------

#include <hrt.h>
#include <hrt/core/context.hpp>
#include <hrt/core/version.hpp>
#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>
#include <hrt/model/image_model.hpp>
#include <hrt/model/ocr_model.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/log/sinks.hpp>

#include <filesystem>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <variant>

#include "include/core/config.hpp"

namespace core = hrt::core;
namespace model = hrt::model;
namespace postprocess = hrt::postprocess;

static_assert(static_cast<int>(model::Task::Detect) == HRT_TASK_DETECT);
static_assert(static_cast<int>(model::Task::Rotated) == HRT_TASK_ROTATED);
static_assert(static_cast<int>(model::Task::Classify) == HRT_TASK_CLASSIFY);
static_assert(static_cast<int>(model::Task::Keypoint) == HRT_TASK_KEYPOINT);
static_assert(static_cast<int>(model::Task::Segment) == HRT_TASK_SEGMENT);
static_assert(static_cast<int>(core::ImageFormat::InterBgr) == HRT_IMAGE_FORMAT_INTER_BGR);
static_assert(static_cast<int>(core::ImageFormat::Gray) == HRT_IMAGE_FORMAT_GRAY);
static_assert(static_cast<int>(kitzoo::log::Level::Off) == HRT_LOG_LEVEL_OFF);
static_assert(static_cast<int>(spdlog::level::off) == HRT_LOG_LEVEL_OFF);

struct HrtPackage {
  hrt::hrtc::Package package;
};

struct HrtOcrModel {
  hrt::UniquePtr<model::OcrModel> model;
};

struct HrtOcrResult {
  postprocess::OcrResult value;
  hrt::Vector<HrtOcrText> texts;
};

struct HrtModel {
  hrt::String algo;
  HrtTask task = HRT_TASK_DETECT;
  std::variant<hrt::UniquePtr<model::ImageModel<postprocess::DetectResultMap>>,
               hrt::UniquePtr<model::ImageModel<postprocess::RotatedResultMap>>,
               hrt::UniquePtr<model::ImageModel<postprocess::ClassifyResult>>,
               hrt::UniquePtr<model::ImageModel<postprocess::KeypointResultMap>>,
               hrt::UniquePtr<model::ImageModel<postprocess::SegmentResultMap>>>
      model;
};

// `value` owns the mask pixels the views point at; the flattened arrays keep
// their capacity across calls.
struct HrtResult {
  std::variant<std::monostate, postprocess::DetectResultMap, postprocess::RotatedResultMap, postprocess::ClassifyResult,
               postprocess::KeypointResultMap, postprocess::SegmentResultMap>
      value;
  hrt::Vector<HrtBox> boxes;
  hrt::Vector<HrtRotatedBox> rotated_boxes;
  hrt::Vector<std::size_t> keypoint_offsets;
  hrt::Vector<HrtKeypoint> keypoints;
  hrt::Vector<HrtMask> masks;
  hrt::Vector<std::size_t> score_offsets;
  hrt::Vector<float> scores;
  HrtResultData data{};
};

namespace {

constexpr std::size_t kMaxStageNum = 64;

auto last_error() -> hrt::String& {
  thread_local hrt::String error;
  return error;
}

// The latest error the library logged on this thread during the current call.
auto logged_error() -> hrt::String& {
  thread_local hrt::String error;
  return error;
}

// Where log records go besides the console: error capture for hrt_last_error
// and the callback. Changing the callback only swaps it under the mutex, as
// the default logger's sinks must not change while a thread logs.
struct LogRoute {
  std::mutex mutex;
  HrtLogCallback callback = nullptr;
  void* user_data = nullptr;
  kitzoo::log::SinkPtr console = std::make_shared<kitzoo::log::ConsoleSink>();
};

// Constructed before the default logger, so it outlives the sink that uses it.
auto log_route() -> LogRoute& {
  static LogRoute route;
  return route;
}

auto route_log() -> bool {
  auto& route = log_route();
  kitzoo::log::init({.console = false});
  auto& logger = kitzoo::log::default_logger();
  logger.add_sink(route.console);
  logger.add_sink(std::make_shared<kitzoo::log::CallbackSink>([&route](const spdlog::details::log_msg& record) {
    if (record.level >= spdlog::level::err)
      logged_error().assign(record.payload.data(), record.payload.size());
    const std::lock_guard lock(route.mutex);
    if (route.callback != nullptr) {
      const hrt::String message(record.payload.data(), record.payload.size());
      route.callback(static_cast<HrtLogLevel>(record.level), message.c_str(), message.size(), route.user_data);
    }
  }));
  return true;
}

// Set up when the library loads, before any thread can call into it.
KZ_MAYBE_UNUSED const bool kLogRouted = route_log();

// Starts a call that may fail.
auto begin() -> void {
  logged_error().clear();
}

auto fail(HrtStatus status, hrt::String message) -> HrtStatus {
  last_error() = std::move(message);
  return status;
}

// A failure the library has logged the cause of, which explains it best.
auto fail_logged(HrtStatus status, std::string_view fallback) -> HrtStatus {
  return fail(status, logged_error().empty() ? hrt::String{fallback} : logged_error());
}

auto device_of(hrt::preprocess::ImagePreprocessorType type) -> core::DeviceType {
  switch (type) {
  case hrt::preprocess::ImagePreprocessorType::Cuda:
    return core::DeviceType::Cuda;
  case hrt::preprocess::ImagePreprocessorType::Dvpp:
    return core::DeviceType::Ascend;
  default:
    return core::DeviceType::Cpu;
  }
}

// Whether ImageModel::create finds an engine to load instead of aborting:
// one compiled in whose device, shared with the preprocessor, is usable.
// `algo` passed check_config.
auto check_engines(const hrt::hrtc::AlgoParam& algo) -> HrtStatus {
  const kitzoo::json::Reader reader{algo.config};
  const auto preprocess = reader.raw()
                              .value("preprocess", kitzoo::json::Json::object())
                              .get<hrt::preprocess::ImagePreprocessorParam>()
                              .device;
  if (hrt::preprocess::ImagePreprocessor::create(preprocess) == nullptr)
    return fail(HRT_STATUS_UNAVAILABLE, "preprocess device of algo " + algo.name + " is not available in this build");
  const auto preprocess_device = device_of(preprocess);
  bool runnable = false;
  for (const auto& param : algo.engines) {
    // Engine::initialize aborts on such a config.
    if (!param.config.empty()) {
      const kitzoo::json::Reader config{param.config};
      if (!config.is_parse_success() || !config.raw().is_object())
        return fail(HRT_STATUS_INVALID_CONFIG,
                    "config of engine " + param.engine_type + " of algo " + algo.name + " is not a JSON object");
    }
    const auto type = hrt::engine::engine_type_from(param.engine_type);
    if (runnable || hrt::engine::Engine::create(type) == nullptr)
      continue;
    const auto engine_device = hrt::engine::device_type_of(type);
    if (engine_device != core::DeviceType::Cpu && preprocess_device != core::DeviceType::Cpu &&
        engine_device != preprocess_device)
      continue;
    const auto device = engine_device != core::DeviceType::Cpu ? engine_device : preprocess_device;
    runnable = device == core::DeviceType::Cpu || core::Context::create(device) != nullptr;
  }
  if (!runnable)
    return fail(HRT_STATUS_UNAVAILABLE, "no engine of algo " + algo.name + " runs on this machine in this build");
  return HRT_STATUS_OK;
}

auto to_box(const postprocess::ObjectBox& box) -> HrtBox {
  return {box.rect.x, box.rect.y, box.rect.width, box.rect.height, box.score, box.type, box.track_id};
}

auto fill(HrtResult& result, const postprocess::DetectResultMap& value) -> void {
  for (const auto& [type, boxes] : value) {
    for (const auto& box : boxes)
      result.boxes.push_back(to_box(box));
  }
  result.data.count = result.boxes.size();
  result.data.boxes = result.boxes.data();
}

auto fill(HrtResult& result, const postprocess::RotatedResultMap& value) -> void {
  for (const auto& [type, boxes] : value) {
    for (const auto& box : boxes) {
      const auto& rect = box.rect;
      result.rotated_boxes.push_back({rect.center.x, rect.center.y, rect.size.width, rect.size.height, rect.angle,
                                      box.score, box.type, box.track_id});
    }
  }
  result.data.count = result.rotated_boxes.size();
  result.data.rotated_boxes = result.rotated_boxes.data();
}

auto fill(HrtResult& result, const postprocess::ClassifyResult& value) -> void {
  result.score_offsets.push_back(0);
  for (const auto& scores : value) {
    result.scores.insert(result.scores.end(), scores.begin(), scores.end());
    result.score_offsets.push_back(result.scores.size());
  }
  result.data.count = value.size();
  result.data.score_offsets = result.score_offsets.data();
  result.data.scores = result.scores.data();
}

auto fill(HrtResult& result, const postprocess::KeypointResultMap& value) -> void {
  result.keypoint_offsets.push_back(0);
  for (const auto& [type, objects] : value) {
    for (const auto& object : objects) {
      result.boxes.push_back(to_box(object.box));
      for (const auto& keypoint : object.keypoints)
        result.keypoints.push_back(
            {keypoint.point.x, keypoint.point.y, keypoint.score, static_cast<std::int32_t>(keypoint.visible)});
      result.keypoint_offsets.push_back(result.keypoints.size());
    }
  }
  result.data.count = result.boxes.size();
  result.data.boxes = result.boxes.data();
  result.data.keypoint_offsets = result.keypoint_offsets.data();
  result.data.keypoints = result.keypoints.data();
}

auto fill(HrtResult& result, const postprocess::SegmentResultMap& value) -> void {
  for (const auto& [type, objects] : value) {
    for (const auto& object : objects) {
      result.boxes.push_back(to_box(object.box));
      const auto& rect = object.mask.rect;
      result.masks.push_back({{rect.x, rect.y, rect.width, rect.height}, object.mask.data.data()});
    }
  }
  result.data.count = result.boxes.size();
  result.data.boxes = result.boxes.data();
  result.data.masks = result.masks.data();
}

template <typename Result>
auto assign(HrtResult& result, HrtTask task, Result&& value) -> void {
  result.boxes.clear();
  result.rotated_boxes.clear();
  result.keypoint_offsets.clear();
  result.keypoints.clear();
  result.masks.clear();
  result.score_offsets.clear();
  result.scores.clear();
  result.data = {};
  result.data.task = task;
  fill(result, result.value.template emplace<std::decay_t<Result>>(std::forward<Result>(value)));
}

} // namespace

extern "C" {

auto hrt_ocr_model_create(const HrtPackage* package, const char* detection, const char* recognition,
                          HrtOcrModel** output) -> HrtStatus {
  begin();
  if (output == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_model_create: null output");
  *output = nullptr;
  if (package == nullptr || detection == nullptr || recognition == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_model_create: null argument");
  const auto* det = package->package.find(detection);
  const auto* rec = package->package.find(recognition);
  if (det == nullptr || rec == nullptr)
    return fail(HRT_STATUS_NOT_FOUND, "OCR detection or recognition algo is missing");
  const auto problems = model::check_ocr_config(*det, *rec);
  if (!problems.empty())
    return fail(HRT_STATUS_INVALID_CONFIG, core::join(problems));
  for (const auto* algo : {det, rec}) {
    if (const auto status = check_engines(*algo); status != HRT_STATUS_OK)
      return status;
  }
  *output = new HrtOcrModel{model::OcrModel::create(*det, *rec)};
  return HRT_STATUS_OK;
}

auto hrt_ocr_model_destroy(HrtOcrModel* value) -> void {
  delete value;
}

auto hrt_ocr_result_create(HrtOcrResult** result) -> HrtStatus {
  if (result == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_result_create: null output");
  *result = new HrtOcrResult;
  return HRT_STATUS_OK;
}

auto hrt_ocr_result_destroy(HrtOcrResult* result) -> void {
  delete result;
}

auto hrt_ocr_result_count(const HrtOcrResult* result) -> std::size_t {
  return result == nullptr ? 0 : result->texts.size();
}

auto hrt_ocr_result_text(const HrtOcrResult* result, std::size_t index) -> const HrtOcrText* {
  return result != nullptr && index < result->texts.size() ? &result->texts[index] : nullptr;
}

auto hrt_ocr_model_process(HrtOcrModel* value, const HrtImage* image, HrtOcrResult* result) -> HrtStatus {
  begin();
  if (result != nullptr) {
    result->texts.clear();
    result->value.clear();
  }
  if (value == nullptr || image == nullptr || result == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_model_process: null argument");
  if (image->data == nullptr || image->width <= 0 || image->height <= 0 || image->width > 32768 ||
      image->height > 32768 || image->format <= HRT_IMAGE_FORMAT_UNKNOWN || image->format > HRT_IMAGE_FORMAT_GRAY)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_model_process: invalid image");
  auto src = core::Image::wrap(core::MemoryType::Cpu, const_cast<void*>(image->data),
                               static_cast<core::ImageFormat>(image->format), {image->width, image->height},
                               core::DataType::Uint8, image->row_pitch);
  if (!src)
    return fail_logged(HRT_STATUS_INVALID_ARGUMENT, "hrt_ocr_model_process: cannot wrap image");
  auto output = value->model->process_sync(*src);
  if (!output)
    return fail_logged(HRT_STATUS_PROCESS_FAILED, "OCR processing failed");
  result->value = std::move(*output);
  for (const auto& text : result->value) {
    HrtOcrText view{};
    view.x = text.rect.x;
    view.y = text.rect.y;
    view.width = text.rect.width;
    view.height = text.rect.height;
    view.text = text.text.c_str();
    view.text_size = text.text.size();
    view.detection_score = text.detection_score;
    view.recognition_score = text.recognition_score;
    result->texts.push_back(view);
  }
  return HRT_STATUS_OK;
}

auto hrt_version() -> const char* {
  return hrt::core::kVersion.data();
}

auto hrt_last_error() -> const char* {
  return last_error().c_str();
}

auto hrt_set_log_level(HrtLogLevel level) -> void {
  kitzoo::log::default_logger().set_level(static_cast<kitzoo::log::Level>(level));
}

auto hrt_set_log_callback(HrtLogCallback callback, void* user_data) -> void {
  auto& route = log_route();
  const std::lock_guard lock(route.mutex);
  route.callback = callback;
  route.user_data = user_data;
  route.console->set_level(callback != nullptr ? spdlog::level::off : spdlog::level::trace);
}

auto hrt_engine_available(const char* engine_type) -> int {
  return engine_type != nullptr && hrt::engine::Engine::create(hrt::engine::engine_type_from(engine_type)) != nullptr
             ? 1
             : 0;
}

auto hrt_package_load(const char* path, HrtPackage** package) -> HrtStatus {
  begin();
  if (path == nullptr || package == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_package_load: null argument");
  *package = nullptr;
  // Paths are UTF-8 on every platform.
  const std::string_view text(path);
  const std::filesystem::path file(std::u8string(text.begin(), text.end()));
  auto loaded = file.extension() == ".json" ? hrt::hrtc::load_json(file) : hrt::hrtc::load(file);
  if (!loaded)
    return fail_logged(HRT_STATUS_NOT_FOUND, hrt::String{"cannot load package "} + path);
  *package = new HrtPackage{std::move(*loaded)};
  return HRT_STATUS_OK;
}

auto hrt_package_load_memory(const void* data, std::size_t size, HrtPackage** package) -> HrtStatus {
  begin();
  if (data == nullptr || package == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_package_load_memory: null argument");
  *package = nullptr;
  auto loaded = hrt::hrtc::load(std::span(static_cast<const std::uint8_t*>(data), size));
  if (!loaded)
    return fail_logged(HRT_STATUS_NOT_FOUND, "cannot load the package bytes");
  *package = new HrtPackage{std::move(*loaded)};
  return HRT_STATUS_OK;
}

auto hrt_package_destroy(HrtPackage* package) -> void {
  delete package;
}

auto hrt_package_algo_count(const HrtPackage* package) -> std::size_t {
  return package != nullptr ? package->package.algos.size() : 0;
}

auto hrt_package_algo_name(const HrtPackage* package, std::size_t index) -> const char* {
  if (package == nullptr || index >= package->package.algos.size())
    return nullptr;
  return package->package.algos[index].name.c_str();
}

auto hrt_model_options_default() -> HrtModelOptions {
  return {1, 1, 1};
}

auto hrt_model_create(const HrtPackage* package, const char* algo, const HrtModelOptions* options, HrtModel** model)
    -> HrtStatus {
  begin();
  if (package == nullptr || model == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_model_create: null argument");
  *model = nullptr;
  const auto& algos = package->package.algos;
  const hrt::hrtc::AlgoParam* param = nullptr;
  if (algo != nullptr)
    param = package->package.find(algo);
  else if (!algos.empty())
    param = &algos.front();
  if (param == nullptr)
    return fail(HRT_STATUS_NOT_FOUND, algo != nullptr ? hrt::String{"no algo "} + algo + " in the package"
                                                      : hrt::String{"the package has no algo"});
  const auto counts = options != nullptr ? *options : hrt_model_options_default();
  for (const auto num : {counts.preprocess, counts.engine, counts.postprocess}) {
    if (num < 1 || num > kMaxStageNum)
      return fail(HRT_STATUS_INVALID_ARGUMENT, "stage counts must be in [1, " + std::to_string(kMaxStageNum) + "]");
  }
  const auto problems = model::check_config(*param);
  if (!problems.empty())
    return fail(HRT_STATUS_INVALID_CONFIG, "invalid config of algo " + param->name + ": " + core::join(problems));
  const auto task = model::task_of(*param);
  if (!task)
    return fail(HRT_STATUS_INVALID_CONFIG, "unknown postprocess type of algo " + param->name);
  if (const auto status = check_engines(*param); status != HRT_STATUS_OK)
    return status;

  const model::ImageModelOptions image_options{
      .preprocess = counts.preprocess, .engine = counts.engine, .postprocess = counts.postprocess};
  auto created = std::make_unique<HrtModel>();
  created->algo = param->name;
  created->task = static_cast<HrtTask>(*task);
  const auto make = [&]<typename Result>() {
    created->model = model::ImageModel<Result>::create(*param, image_options);
  };
  switch (*task) {
  case model::Task::Detect:
    make.operator()<postprocess::DetectResultMap>();
    break;
  case model::Task::Rotated:
    make.operator()<postprocess::RotatedResultMap>();
    break;
  case model::Task::Classify:
    make.operator()<postprocess::ClassifyResult>();
    break;
  case model::Task::Keypoint:
    make.operator()<postprocess::KeypointResultMap>();
    break;
  case model::Task::Segment:
    make.operator()<postprocess::SegmentResultMap>();
    break;
  }
  *model = created.release();
  return HRT_STATUS_OK;
}

auto hrt_model_destroy(HrtModel* model) -> void {
  delete model;
}

auto hrt_model_task(const HrtModel* model) -> HrtTask {
  return model != nullptr ? model->task : HRT_TASK_DETECT;
}

auto hrt_model_algo_name(const HrtModel* model) -> const char* {
  return model != nullptr ? model->algo.c_str() : nullptr;
}

auto hrt_result_create(HrtResult** result) -> HrtStatus {
  if (result == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_result_create: null argument");
  *result = new HrtResult;
  return HRT_STATUS_OK;
}

auto hrt_result_destroy(HrtResult* result) -> void {
  delete result;
}

auto hrt_result_data(const HrtResult* result) -> const HrtResultData* {
  return result != nullptr ? &result->data : nullptr;
}

auto hrt_model_process(HrtModel* model, const HrtImage* image, const HrtRect* roi, HrtResult* result) -> HrtStatus {
  begin();
  if (model == nullptr || image == nullptr || result == nullptr)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_model_process: null argument");
  if (image->data == nullptr || image->width <= 0 || image->height <= 0 || image->format <= HRT_IMAGE_FORMAT_UNKNOWN ||
      image->format > HRT_IMAGE_FORMAT_GRAY)
    return fail(HRT_STATUS_INVALID_ARGUMENT, "hrt_model_process: invalid image");
  const auto src = core::Image::wrap(core::MemoryType::Cpu, const_cast<void*>(image->data),
                                     static_cast<core::ImageFormat>(image->format), {image->width, image->height},
                                     core::DataType::Uint8, image->row_pitch);
  if (!src)
    return fail_logged(HRT_STATUS_INVALID_ARGUMENT, "hrt_model_process: cannot wrap the image");
  const auto rect = roi != nullptr ? core::Rect{roi->x, roi->y, roi->width, roi->height} : core::Rect{};
  return std::visit(
      [&](auto& image_model) -> HrtStatus {
        auto output = image_model->process_sync(*src, rect);
        if (!output)
          return fail_logged(HRT_STATUS_PROCESS_FAILED, "processing of algo " + model->algo + " failed");
        assign(*result, model->task, std::move(*output));
        return HRT_STATUS_OK;
      },
      model->model);
}

} // extern "C"
