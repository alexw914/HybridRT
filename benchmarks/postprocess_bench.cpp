// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/postprocess_bench.cpp
// Description: Benchmarks each YOLO task's postprocessing on the real outputs
//              of its model.
// -----------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <type_traits>

#include "yolo_fixture.hpp"

namespace {

template <typename Result>
auto create_postprocessor(const kitzoo::json::Json& config, const hrt::engine::Engine& engine)
    -> hrt::UniquePtr<hrt::postprocess::Postprocessor<Result>> {
  const auto& type = config["type"].get_ref<const hrt::String&>();
  if constexpr (std::is_same_v<Result, hrt::postprocess::DetectResultMap>)
    return hrt::postprocess::create_detect_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, hrt::postprocess::SegmentResultMap>)
    return hrt::postprocess::create_segment_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, hrt::postprocess::KeypointResultMap>)
    return hrt::postprocess::create_keypoint_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, hrt::postprocess::RotatedResultMap>)
    return hrt::postprocess::create_rotated_postprocessor(type, config, engine);
  else
    return hrt::postprocess::create_classify_postprocessor(type, config, engine);
}

// Decoding, NMS and mapping back to the source image; the model runs once
// beforehand and is not timed.
auto BM_Task(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const auto task = static_cast<std::size_t>(state.range(0));
  state.SetLabel(hrt::bench::kTasks.at(task));
  const auto entry = hrt::bench::algo_entry(task);
  const auto bgr = hrt::bench::task_image(task);
  const auto source = hrt::bench::wrap_bgr(bgr);
  const auto algo = entry.is_object() ? hrt::hrtc::load_algo(entry) : std::nullopt;
  if (!algo || !source) {
    state.SkipWithError("Template model or assets/test/images is missing");
    return;
  }
  auto engine = hrt::engine::Engine::create(algo->engines.front().engine_type);
  if (engine == nullptr || !engine->initialize(algo->engines.front())) {
    state.SkipWithError("The engine is not available or the model did not load");
    return;
  }
  hrt::core::Buffers inputs;
  hrt::core::Buffers outputs;
  for (const auto& proto : engine->input_protos())
    inputs.push_back(std::make_shared<hrt::core::Buffer>(proto));
  for (const auto& proto : engine->output_protos())
    outputs.push_back(std::make_shared<hrt::core::Buffer>(proto));
  const auto config = kitzoo::json::Json::parse(algo->config);
  const auto param = config["preprocess"].get<hrt::preprocess::ImagePreprocessorParam>();
  const auto& shape = inputs.front()->shape();
  hrt::core::Image input(inputs.front()->memory()->ref(), param.dst_image_format, inputs.front()->data_type(),
                         {static_cast<int>(shape[3]), static_cast<int>(shape[2])});
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  const auto transform = preprocessor->process(input, *source, {}, param);
  if (!transform || !engine->set_inputs(inputs) || !engine->set_outputs(outputs) || !engine->inference()) {
    state.SkipWithError("Preprocessing or inference failed");
    return;
  }

  hrt::bench::with_result(task, [&]<typename Result>() -> void {
    auto postprocessor = create_postprocessor<Result>(config["postprocess"], *engine);
    Result result;
    for (auto _ : state) {
      if (!postprocessor->process(result, outputs, *transform)) {
        state.SkipWithError("Postprocessing failed");
        break;
      }
      benchmark::DoNotOptimize(result);
    }
  });
}

} // namespace

BENCHMARK(BM_Task)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
