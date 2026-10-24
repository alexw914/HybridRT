// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/postprocess_bench.cpp
// Description: Benchmarks every template postprocessor on real model outputs.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

#include <benchmark/benchmark.h>

#include "yolo_fixture.hpp"

namespace {

struct Prepared {
  hrt::UniquePtr<hrt::engine::Engine> engine;
  hrt::core::Buffers outputs;
  hrt::Vector<hrt::preprocess::ImageTransform> transforms; // One per sample of the static batch.
  kitzoo::json::Json config;
};

// Runs the model of `entry` once with `source` in every sample, for timing its postprocessing alone.
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
  const auto& input = *inputs.front();
  const auto& shape = input.shape();
  auto* base = static_cast<std::byte*>(input.memory()->mutable_cpu());
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  for (std::int64_t i = 0; i < shape[0]; ++i) {
    auto sample = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu,
                                         base + static_cast<std::size_t>(i) * input.stride()[1], param.dst_image_format,
                                         {static_cast<int>(shape[3]), static_cast<int>(shape[2])}, input.data_type());
    const auto transform = sample ? preprocessor->process(*sample, *source, {}, param) : std::nullopt;
    if (!transform) {
      state.SkipWithError("Preprocessing failed");
      return std::nullopt;
    }
    prepared.transforms.push_back(*transform);
  }
  if (!engine->set_inputs(inputs) || !engine->set_outputs(prepared.outputs) || !engine->inference()) {
    state.SkipWithError("Inference failed");
    return std::nullopt;
  }
  return prepared;
}

template <typename Result>
auto measure_postprocessor(benchmark::State& state, Prepared& prepared) -> void {
  const auto& config = prepared.config["postprocess"];
  auto postprocessor = hrt::postprocess::create_postprocessor<Result>(config["type"].get_ref<const hrt::String&>(),
                                                                      config, *prepared.engine);
  if (!postprocessor) {
    state.SkipWithError("Postprocessor initialization failed");
    return;
  }
  // A static batch decodes every sample, as ImageModel does; items are samples.
  const auto batched = prepared.transforms.size() > 1;
  Result result;
  hrt::Vector<Result> results;
  for (auto _ : state) {
    const auto ok = batched ? postprocessor->process(results, prepared.outputs, prepared.transforms)
                            : postprocessor->process(result, prepared.outputs, prepared.transforms.front());
    if (!ok) {
      state.SkipWithError("Postprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(prepared.transforms.size()));
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
  state.counters["input_width"] = prepared->transforms.front().input_size.width;
  state.counters["input_height"] = prepared->transforms.front().input_size.height;
  // Instance masks decode on OpenCV's threads; the second argument sets their number.
  cv::setNumThreads(static_cast<int>(state.range(1)));
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
  cv::setNumThreads(1);
}

auto type_cases(benchmark::Benchmark* bench) -> void {
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_SOURCE_DIR) / "assets/template/hrtc_template.json");
  const auto& algos = reader.raw()["algos"];
  for (std::size_t i = 0; i < algos.size(); ++i) {
    bench->Args({static_cast<std::int64_t>(i), 1});
    const auto& type = algos[i]["config"]["postprocess"]["type"].get_ref<const hrt::String&>();
    if (type.ends_with("Segment") && type != "SemanticSegment")
      bench->Args({static_cast<std::int64_t>(i), 4});
  }
}

} // namespace

BENCHMARK(BM_Type)->Apply(type_cases)->Unit(benchmark::kMillisecond)->UseRealTime();
