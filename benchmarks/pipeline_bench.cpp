// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/pipeline_bench.cpp
// Description: Benchmarks ImageModel::process_sync end to end: preprocessing,
//              inference and postprocessing of each YOLO task.
// -----------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <mutex>

#include "yolo_fixture.hpp"

namespace {

constexpr hrt::Array<const char*, 2> kManifests{"yolo.json", "yolo_coreml.json"};

// Latency of one call; args: task, manifest (0 ORT CPU, 1 ORT CoreML).
auto BM_Task(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const auto task = static_cast<std::size_t>(state.range(0));
  const auto* manifest = kManifests.at(static_cast<std::size_t>(state.range(1)));
  state.SetLabel(hrt::String(hrt::bench::kTasks.at(task)) + "/" + manifest);
  const auto entry = hrt::bench::algo_entry(manifest, task);
  const auto bgr = hrt::bench::task_image(task);
  const auto image = hrt::bench::wrap_bgr(bgr);
  if (!entry.is_object() || !image) {
    state.SkipWithError("assets/yolo manifest or image is missing");
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
    const auto entry = hrt::bench::algo_entry("yolo.json", 0);
    if (slot == nullptr && entry.is_object() && image) {
      slot = Model::create(entry, {.preprocess = static_cast<std::size_t>(state.range(0)),
                                   .engine = static_cast<std::size_t>(state.range(1)),
                                   .postprocess = static_cast<std::size_t>(state.range(2))});
    }
    model = slot.get();
  }
  if (model == nullptr) {
    state.SkipWithError("assets/yolo manifest or image is missing");
    return;
  }
  for (auto _ : state)
    benchmark::DoNotOptimize(model->process_sync(*image));
  state.SetItemsProcessed(state.iterations());
}

} // namespace

BENCHMARK(BM_Task)
    ->ArgNames({"task", "manifest"})
    ->ArgsProduct({{0, 1, 2, 3, 4}, {0, 1}})
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
