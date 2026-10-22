// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/postprocess_bench.cpp
// Description: Benchmarks every template postprocessor on real model outputs.
// -----------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <type_traits>

#include "yolo_fixture.hpp"

namespace {

template <typename Result>
auto check_init_postprocess(const kitzoo::json::Json& config, const hrt::engine::Engine& engine)
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
  else if constexpr (std::is_same_v<Result, hrt::postprocess::EmbeddingResult>)
    return hrt::postprocess::create_embedding_postprocessor(type, config, engine);
  else if constexpr (std::is_same_v<Result, hrt::postprocess::OcrResult>)
    return hrt::postprocess::create_ocr_postprocessor(config, engine);
  else
    return hrt::postprocess::create_classify_postprocessor(type, config, engine);
}

struct Prepared {
  hrt::UniquePtr<hrt::engine::Engine> engine;
  hrt::core::Buffers outputs;
  hrt::preprocess::ImageTransform transform;
  kitzoo::json::Json config;
};

// Runs the model of `entry` once on `source`, for timing its postprocessing alone.
auto prepare(benchmark::State& state, const kitzoo::json::Json& entry, const std::optional<hrt::core::Image>& source)
    -> std::optional<Prepared> {
  const auto algo = entry.is_object() ? hrt::hrtc::load_algo(entry) : std::nullopt;
  if (!algo || !source) {
    state.SkipWithError("Template model or assets/test/images is missing");
    return std::nullopt;
  }
  Prepared prepared{
      hrt::engine::Engine::create(algo->engines.front().engine_type), {}, {}, kitzoo::json::Json::parse(algo->config)};
  auto& engine = prepared.engine;
  if (engine == nullptr || !engine->initialize(algo->engines.front())) {
    state.SkipWithError("The engine is not available or the model did not load");
    return std::nullopt;
  }
  hrt::core::Buffers inputs;
  for (const auto& proto : engine->input_protos())
    inputs.push_back(std::make_shared<hrt::core::Buffer>(proto));
  for (const auto& proto : engine->output_protos())
    prepared.outputs.push_back(std::make_shared<hrt::core::Buffer>(proto));
  const auto param = prepared.config["preprocess"].get<hrt::preprocess::ImagePreprocessorParam>();
  const auto& shape = inputs.front()->shape();
  hrt::core::Image input(inputs.front()->memory()->ref(), param.dst_image_format, inputs.front()->data_type(),
                         {static_cast<int>(shape[3]), static_cast<int>(shape[2])});
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  const auto transform = preprocessor->process(input, *source, {}, param);
  if (!transform || !engine->set_inputs(inputs) || !engine->set_outputs(prepared.outputs) || !engine->inference()) {
    state.SkipWithError("Preprocessing or inference failed");
    return std::nullopt;
  }
  prepared.transform = *transform;
  return prepared;
}

template <typename Result>
auto measure_postprocessor(benchmark::State& state, Prepared& prepared) -> void {
  auto postprocessor = check_init_postprocess<Result>(prepared.config["postprocess"], *prepared.engine);
  if (!postprocessor) {
    state.SkipWithError("Postprocessor initialization failed");
    return;
  }
  Result result;
  for (auto _ : state) {
    if (!postprocessor->process(result, prepared.outputs, prepared.transform)) {
      state.SkipWithError("Postprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(result);
  }
}

auto BM_Type(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_SOURCE_DIR) / "assets/template/hrtc_template.json");
  const auto& algos = reader.raw()["algos"];
  const auto index = static_cast<std::size_t>(state.range(0));
  if (index >= algos.size()) {
    state.SkipWithError("Template algo index is out of range");
    return;
  }
  const auto name = algos[index]["name"].get<hrt::String>();
  const auto entry = hrt::bench::template_entry(name);
  if (!entry.is_object()) {
    state.SkipWithError("Template model is missing");
    return;
  }
  const auto& post = entry["config"]["postprocess"];
  const auto type = post["type"].get<hrt::String>();
  const auto is_ocr = type == "Ocr";
  const auto is_rotated = type.ends_with("Rotated");
  auto bgr = hrt::bench::read_image(is_ocr ? "ocr_document.png" : is_rotated ? "airport.jpg" : "bus.jpg");
  if (is_ocr && post["stage"] == "recognition" && !bgr.empty())
    bgr = bgr(cv::Rect(419, 16, 241, 50)).clone();
  auto prepared = prepare(state, entry, hrt::bench::wrap_bgr(bgr));
  if (!prepared)
    return;
  state.SetLabel(type + ":" + name);
  state.counters["input_width"] = prepared->transform.input_size.width;
  state.counters["input_height"] = prepared->transform.input_size.height;
  if (type.ends_with("Detect"))
    measure_postprocessor<hrt::postprocess::DetectResultMap>(state, *prepared);
  else if (type.ends_with("Segment"))
    measure_postprocessor<hrt::postprocess::SegmentResultMap>(state, *prepared);
  else if (type.ends_with("Keypoint") || type == "GeneralYoloPose")
    measure_postprocessor<hrt::postprocess::KeypointResultMap>(state, *prepared);
  else if (is_rotated)
    measure_postprocessor<hrt::postprocess::RotatedResultMap>(state, *prepared);
  else if (type == "Embedding")
    measure_postprocessor<hrt::postprocess::EmbeddingResult>(state, *prepared);
  else if (is_ocr)
    measure_postprocessor<hrt::postprocess::OcrResult>(state, *prepared);
  else
    measure_postprocessor<hrt::postprocess::ClassifyResult>(state, *prepared);
}

auto type_cases(benchmark::Benchmark* bench) -> void {
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_SOURCE_DIR) / "assets/template/hrtc_template.json");
  for (std::size_t i = 0; i < reader.raw()["algos"].size(); ++i)
    bench->Arg(static_cast<std::int64_t>(i));
}

} // namespace

BENCHMARK(BM_Type)->Apply(type_cases)->Unit(benchmark::kMillisecond)->UseRealTime();
