// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/pipeline_bench.cpp
// Description: Benchmarks ImageModel::process_sync end to end for each YOLO
//              task, single images and batches, and OcrModel::process_sync for
//              OCR.
// -----------------------------------------------------------------------------

#include <hrt/model/ocr_model.hpp>

#include <benchmark/benchmark.h>
#include <filesystem>
#include <mutex>

#include "yolo_fixture.hpp"

namespace {

// OpenCV changes process-wide TBB state; callers must not reconfigure it concurrently.
auto configure_opencv() -> void {
  static std::once_flag once;
  std::call_once(once, []() -> void { cv::setNumThreads(1); });
}

#ifdef HRT_BENCH_ORT
constexpr hrt::Array<const char*, 2> kProviders{"cpu", "coreml"};

// ORT CPU, plus CoreML on Apple platforms, the only ones that have it.
#ifdef __APPLE__
const hrt::Vector<std::int64_t> kProviderArgs{0, 1};
#else
const hrt::Vector<std::int64_t> kProviderArgs{0};
#endif

// Latency of one call; args: task, provider (0 ORT CPU, 1 ORT CoreML).
auto BM_Task(benchmark::State& state) -> void {
  configure_opencv();
  const auto task = static_cast<std::size_t>(state.range(0));
  const auto provider = static_cast<std::size_t>(state.range(1));
  state.SetLabel(hrt::String(hrt::bench::kTasks.at(task)) + "/" + kProviders.at(provider));
  const auto entry = hrt::bench::algo_entry(task, static_cast<hrt::bench::Provider>(provider));
  const auto bgr = hrt::bench::task_image(task);
  const auto image = hrt::bench::wrap_bgr(bgr);
  if (!entry.is_object() || !image) {
    state.SkipWithError("Template model or assets/test/images is missing");
    return;
  }
  hrt::bench::with_result(task, [&]<typename Result>() -> void {
    const auto model = hrt::model::ImageModel<Result>::create(entry);
    for (auto _ : state) {
      auto result = model->process_sync(*image);
      if (!result) {
        state.SkipWithError("process_sync failed");
        break;
      }
      benchmark::DoNotOptimize(result);
    }
  });
}

// Throughput of the detect model shared by state.threads() callers
// (items_per_second = calls per second); args: preprocess, engine and
// postprocess stage counts.
auto BM_Pooled(benchmark::State& state) -> void {
  configure_opencv();
  using Model = hrt::model::ImageModel<hrt::postprocess::DetectResultMap>;
  static std::mutex mutex;
  static hrt::Map<hrt::Array<std::int64_t, 3>, hrt::UniquePtr<Model>> models;
  static const auto kBgr = hrt::bench::task_image(0);
  static const auto kImage = hrt::bench::wrap_bgr(kBgr);
  Model* model = nullptr;
  {
    const std::lock_guard lock(mutex);
    auto& slot = models[{state.range(0), state.range(1), state.range(2)}];
    const auto entry = hrt::bench::algo_entry(0);
    if (slot == nullptr && entry.is_object() && kImage) {
      slot = Model::create(entry, {.preprocess = static_cast<std::size_t>(state.range(0)),
                                   .engine = static_cast<std::size_t>(state.range(1)),
                                   .postprocess = static_cast<std::size_t>(state.range(2))});
    }
    model = slot.get();
  }
  if (model == nullptr) {
    state.SkipWithError("Template model or assets/test/images is missing");
    return;
  }
  for (auto _ : state)
    benchmark::DoNotOptimize(model->process_sync(*kImage));
  state.SetItemsProcessed(state.iterations());
}

// OcrModel on the document image; args: recognizer (0 tiny, 1 tiny with a static batch of 8), provider.
// Counter
// "lines" is the number of recognized text lines per call.
auto BM_Ocr(benchmark::State& state) -> void {
  configure_opencv();
  const auto recognizer = static_cast<std::size_t>(state.range(0));
  const auto provider = static_cast<hrt::bench::Provider>(state.range(1));
  constexpr hrt::Array<const char*, 2> kRecognizers{"tiny", "tiny_batch8"};
  state.SetLabel(hrt::String(kRecognizers.at(recognizer)) + "/" +
                 kProviders.at(static_cast<std::size_t>(state.range(1))));
  auto det = hrt::bench::template_entry(hrt::bench::kOcrAlgos[0], provider);
  auto rec = hrt::bench::template_entry(hrt::bench::kOcrAlgos.at(recognizer + 1), provider);
  // As in hrtc_device_config.json; the default NeuralNetwork format runs these models several times slower.
  for (auto* entry : {&det, &rec}) {
    if (provider == hrt::bench::Provider::CoreMl && entry->is_object())
      (*entry)["config"]["engine"]["backends"][0]["config"]["provider_options"] = {{"ModelFormat", "MLProgram"},
                                                                                   {"RequireStaticInputShapes", "1"}};
  }
  const auto bgr = hrt::bench::read_image(hrt::bench::kOcrImage);
  const auto image = hrt::bench::wrap_bgr(bgr);
  const auto det_algo = det.is_object() ? hrt::hrtc::load_algo(det) : std::nullopt;
  const auto rec_algo = rec.is_object() ? hrt::hrtc::load_algo(rec) : std::nullopt;
  if (!det_algo || !rec_algo || !image) {
    state.SkipWithError("OCR models, dictionaries or assets/test/images is missing");
    return;
  }
  const auto model = hrt::model::OcrModel::create(*det_algo, *rec_algo);
  std::size_t lines = 0;
  for (auto _ : state) {
    auto result = model->process_sync(*image);
    if (!result) {
      state.SkipWithError("process_sync failed");
      break;
    }
    lines = result->size();
    benchmark::DoNotOptimize(result);
  }
  state.counters["lines"] = static_cast<double>(lines);
}

#endif

// Batch pipelines: 0 CPU preprocessing with ORT CoreML, 1 Metal with ORT CoreML, 2 CUDA with TensorRT, 3 RGA with
// RKNN; each build registers the ones of its platform.
#ifdef HRT_BENCH_RKNN
const hrt::Vector<std::int64_t> kBatchPipelines{3};
#elif defined(__APPLE__)
const hrt::Vector<std::int64_t> kBatchPipelines{0, 1};
#else
const hrt::Vector<std::int64_t> kBatchPipelines{2};
#endif

template <typename Result>
auto batch_models() -> hrt::Map<hrt::Array<std::int64_t, 3>, hrt::UniquePtr<hrt::model::ImageModel<Result>>>& {
  static hrt::Map<hrt::Array<std::int64_t, 3>, hrt::UniquePtr<hrt::model::ImageModel<Result>>> models;
  return models;
}

// The template algo of `task` with a static `batch`, rewritten for `pipeline`, or null when its model,
// TensorRT plan or RKNN file is missing or does not load here.
auto batch_entry(std::size_t task, std::int64_t pipeline, std::int64_t batch) -> kitzoo::json::Json {
  const auto name = hrt::String(hrt::bench::kAlgos.at(task)) + (batch == 1 ? "" : "_batch" + std::to_string(batch));
  auto entry = hrt::bench::template_entry(name, pipeline < 2 ? hrt::bench::Provider::CoreMl : hrt::bench::Provider::Cpu,
                                          pipeline < 3);
  if (!entry.is_object())
    return {};
  auto& preprocess = entry["config"]["preprocess"];
  auto path = std::filesystem::path(entry["config"]["engine"]["backends"][0]["data"].get<hrt::String>());
  if (pipeline == 1) {
    preprocess["device"] = "metal";
    if (!hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Metal))
      return {};
  } else if (pipeline == 2) {
    preprocess["device"] = "cuda";
    path = std::filesystem::path("assets/test/trt") / path.filename().replace_extension("engine");
    entry["config"]["engine"]["backends"] = {{{"engine_type", "trt"}, {"data", path.string()}}};
  } else if (pipeline == 3) {
    preprocess["device"] = "rga";
    preprocess["dst_image_format"] = "inter_rgb";
    preprocess["std"] = {1, 1, 1};
    path = std::filesystem::path("assets/test/rknn") / path.filename().replace_extension("rknn");
    entry["config"]["engine"]["backends"] = {
        {{"engine_type", "rknn"}, {"data", path.string()}, {"config", {{"image_input", true}}}}};
  }
  if (pipeline >= 2) {
    // TensorRT plans load only on the TensorRT version and platform that built them.
    const auto context = pipeline == 2 ? hrt::core::Context::create(hrt::core::DeviceType::Cuda) : nullptr;
    if (pipeline == 2 && context == nullptr)
      return {};
    if (context)
      context->set_current();
    const auto param = hrt::hrtc::load_algo(entry);
    auto probe =
        hrt::engine::Engine::create(pipeline == 2 ? hrt::engine::EngineType::TensorRt : hrt::engine::EngineType::Rknn);
    if (!param || !probe || !std::filesystem::is_regular_file(path) || !probe->initialize(param->engines.front()))
      return {};
  }
  return entry;
}

// The first 32 sorted COCO128 images, or none when the dataset is missing.
auto coco_frames() -> const hrt::Vector<cv::Mat>& {
  static const auto kFrames = [] {
    hrt::Vector<cv::Mat> frames;
    const auto dataset = std::filesystem::path(HRT_SOURCE_DIR) / "assets/test/calib/coco128/images/train2017";
    if (!std::filesystem::is_directory(dataset))
      return frames;
    hrt::Vector<std::filesystem::path> paths;
    for (const auto& file : std::filesystem::directory_iterator(dataset)) {
      if (file.path().extension() == ".jpg")
        paths.push_back(file.path());
    }
    std::sort(paths.begin(), paths.end());
    for (std::size_t i = 0; i < std::min(paths.size(), std::size_t{32}); ++i)
      frames.push_back(cv::imread(paths[i].string()));
    return frames;
  }();
  return kFrames;
}

// Throughput of a model with a static batch shared by 8 callers with 3/2/3 stage pools; args: task, pipeline,
// batch. Each call passes `batch` images, and every caller cycles through the same 32 COCO images from its own
// offset. Items are images.
auto BM_Batch(benchmark::State& state) -> void {
  configure_opencv();
  const auto task = static_cast<std::size_t>(state.range(0));
  const auto pipeline = state.range(1);
  const auto batch = state.range(2);
  constexpr hrt::Array<const char*, 4> kPipelines{"cpu_coreml", "metal_coreml", "cuda_trt", "rga_rknn"};
  state.SetLabel(hrt::String(hrt::bench::kTasks.at(task)) + "/" + kPipelines.at(static_cast<std::size_t>(pipeline)));
  hrt::bench::with_result(task, [&]<typename Result>() -> void {
    using Model = hrt::model::ImageModel<Result>;
    static std::mutex mutex;
    Model* model = nullptr;
    {
      const std::lock_guard lock(mutex);
      auto& slot = batch_models<Result>()[{static_cast<std::int64_t>(task), pipeline, batch}];
      if (slot == nullptr) {
        const auto entry = batch_entry(task, pipeline, batch);
        if (entry.is_object()) {
          const hrt::model::ImageModelOptions options{3, 2, 3};
          slot = pipeline == 2 ? Model::create(entry, hrt::core::Context::create(hrt::core::DeviceType::Cuda), options)
                               : Model::create(entry, options);
        }
      }
      model = slot.get();
    }
    if (model == nullptr) {
      state.SkipWithMessage("Batch model, its device or SDK is unavailable");
      return;
    }
    const auto& frames = coco_frames();
    if (frames.size() < 32) {
      state.SkipWithMessage("COCO dataset is missing or has too few images");
      return;
    }
    hrt::Vector<hrt::Vector<hrt::core::Image>> calls(frames.size() / static_cast<std::size_t>(batch));
    for (std::size_t i = 0; i < frames.size(); ++i) {
      const auto wrapped = hrt::bench::wrap_bgr(frames[i]);
      if (!wrapped) {
        state.SkipWithError("Cannot read COCO image");
        return;
      }
      calls[i / static_cast<std::size_t>(batch)].push_back(*wrapped);
    }
    auto next = static_cast<std::size_t>(state.thread_index()) % calls.size();
    for (int i = 0; i < 5; ++i)
      benchmark::DoNotOptimize(model->process_sync(calls[(next + static_cast<std::size_t>(i)) % calls.size()]));
    for (auto _ : state) {
      auto results = model->process_sync(calls[next]);
      if (!results) {
        state.SkipWithError("process_sync failed");
        break;
      }
      benchmark::DoNotOptimize(results);
      next = (next + 1) % calls.size();
    }
  });
  state.SetItemsProcessed(state.iterations() * batch);
}

} // namespace

#ifdef HRT_BENCH_ORT
BENCHMARK(BM_Task)
    ->ArgNames({"task", "provider"})
    ->ArgsProduct({{0, 1, 2, 3, 4}, kProviderArgs})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BM_Ocr)
    ->ArgNames({"rec", "provider"})
    ->ArgsProduct({{0, 1}, kProviderArgs})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BM_Pooled)
    ->ArgNames({"pre", "engine", "post"})
    ->Args({1, 1, 1})
    ->Args({3, 2, 3})
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
#endif
BENCHMARK(BM_Batch)
    ->Teardown([](const benchmark::State& state) {
      // Release device resources while CUDA and logging are still alive.
      hrt::bench::with_result(static_cast<std::size_t>(state.range(0)), [&]<typename Result>() {
        batch_models<Result>().erase({state.range(0), state.range(1), state.range(2)});
      });
    })
    ->ArgNames({"task", "pipeline", "batch"})
    ->ArgsProduct({{0, 1, 2, 4}, kBatchPipelines, {1, 4, 8}})
    ->Threads(8)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
