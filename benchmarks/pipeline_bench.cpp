// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/pipeline_bench.cpp
// Description: Benchmarks ImageModel::process_sync end to end for each YOLO
//              task, and OcrModel::process_sync for OCR.
// -----------------------------------------------------------------------------

#include <hrt/model/ocr_model.hpp>

#include <benchmark/benchmark.h>
#include <mutex>

#include "yolo_fixture.hpp"

namespace {

constexpr hrt::Array<const char*, 2> kProviders{"cpu", "coreml"};

// Latency of one call; args: task, provider (0 ORT CPU, 1 ORT CoreML).
auto BM_Task(benchmark::State& state) -> void {
  cv::setNumThreads(1);
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
  using Model = hrt::model::ImageModel<hrt::postprocess::DetectResultMap>;
  static std::mutex mutex;
  static hrt::Map<hrt::Array<std::int64_t, 3>, hrt::UniquePtr<Model>> models;
  static const auto bgr = hrt::bench::task_image(0);
  static const auto image = hrt::bench::wrap_bgr(bgr);
  Model* model = nullptr;
  {
    const std::lock_guard lock(mutex);
    auto& slot = models[{state.range(0), state.range(1), state.range(2)}];
    const auto entry = hrt::bench::algo_entry(0);
    if (slot == nullptr && entry.is_object() && image) {
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
    benchmark::DoNotOptimize(model->process_sync(*image));
  state.SetItemsProcessed(state.iterations());
}

// OcrModel on the document image; args: recognizer (0 tiny, 1 small), provider. Counter
// "lines" is the number of recognized text lines per call.
auto BM_Ocr(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const auto recognizer = static_cast<std::size_t>(state.range(0));
  const auto provider = static_cast<hrt::bench::Provider>(state.range(1));
  state.SetLabel(hrt::String(recognizer == 0 ? "tiny" : "small") + "/" +
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

} // namespace

BENCHMARK(BM_Task)
    ->ArgNames({"task", "provider"})
    ->ArgsProduct({{0, 1, 2, 3, 4}, {0, 1}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK(BM_Ocr)
    ->ArgNames({"rec", "provider"})
    ->ArgsProduct({{0, 1}, {0, 1}})
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
