// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/postprocess_bench.cpp
// Description: Benchmarks template postprocessors on real engine outputs.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

#include <kitzoo/os/filesys.hpp>

#include <benchmark/benchmark.h>
#include <cstring>
#include <type_traits>

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
  const auto size = engine->input_size();
  if (!size) {
    state.SkipWithError("The model has no single image input");
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
#ifdef HRT_BENCH_RKNN
  const auto path = std::filesystem::path(HRT_SOURCE_DIR) / "assets/test/calib/rknn_int8" /
                    entry["name"].get<hrt::String>() / "input_nhwc.bin";
  std::error_code error;
  const auto bytes = kitzoo::os::read_file(path, error);
  if (error || bytes.size() != input.bytes()) {
    state.SkipWithError("Prepared RKNN input is missing or has the wrong size");
    return std::nullopt;
  }
  std::memcpy(base, bytes.data(), bytes.size());
  hrt::preprocess::ImageTransform transform;
  transform.input_size = *size;
  if (!hrt::preprocess::calc_resize_position(transform.dst_roi, transform.src_roi, {0, 0, size->width, size->height},
                                             source->rect(), source->size(), param)) {
    state.SkipWithError("Invalid source mapping");
    return std::nullopt;
  }
  prepared.transforms.assign(static_cast<std::size_t>(shape[0]), transform);
#else
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  for (std::int64_t i = 0; i < shape[0]; ++i) {
    auto sample =
        hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, base + static_cast<std::size_t>(i) * input.stride()[1],
                               param.dst_image_format, *size, input.data_type());
    const auto transform = sample ? preprocessor->process(*sample, *source, {}, param) : std::nullopt;
    if (!transform) {
      state.SkipWithError("Preprocessing failed");
      return std::nullopt;
    }
    prepared.transforms.push_back(*transform);
  }
#endif
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
  if constexpr (requires { result.begin()->second.size(); }) {
    const auto count = [](const Result& sample) {
      std::size_t total = 0;
      for (const auto& [type, objects] : sample)
        total += objects.size();
      return total;
    };
    std::size_t objects = batched ? 0 : count(result);
    for (const auto& sample : results)
      objects += count(sample);
    state.counters["decoded_objects"] = static_cast<double>(objects);
  } else if constexpr (std::is_same_v<Result, hrt::postprocess::OcrResult>) {
    std::size_t objects = batched ? 0 : result.size();
    for (const auto& sample : results)
      objects += sample.size();
    state.counters["decoded_objects"] = static_cast<double>(objects);
  }
  state.counters["seconds_per_image"] =
      benchmark::Counter(static_cast<double>(prepared.transforms.size()),
                         benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
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
  auto entry = hrt::bench::template_entry(name, hrt::bench::Provider::Cpu, false);
  if (!entry.is_object()) {
    state.SkipWithError("Template model is missing");
    return;
  }
#ifdef HRT_BENCH_RKNN
  const auto quantized = state.range(1) != 0;
  const auto batch = state.range(2);
  auto path = std::filesystem::path(entry["config"]["engine"]["backends"][0]["data"].get<hrt::String>());
  const auto suffix = batch > 1 ? "Batch" + std::to_string(batch) : hrt::String{};
  path = std::filesystem::path(HRT_SOURCE_DIR) / "assets/test/rknn" /
         (path.stem().string() + suffix + (quantized ? "Int8.rknn" : ".rknn"));
  if (batch > 1)
    entry["name"] = name + "_batch" + std::to_string(batch);
  if (!std::filesystem::is_regular_file(path)) {
    state.SkipWithMessage("RKNN model variant is unavailable");
    return;
  }
  entry["config"]["engine"]["backends"] = {
      {{"engine_type", "rknn"}, {"data", path.string()}, {"config", {{"image_input", true}}}}};
#else
  const auto batch = state.range(1);
  auto& backend = entry["config"]["engine"]["backends"][0];
  auto path = std::filesystem::path(backend["data"].get<hrt::String>());
  if (batch > 1) {
    path = path.parent_path() / (path.stem().string() + "Batch" + std::to_string(batch) + ".onnx");
    entry["name"] = name + "_batch" + std::to_string(batch);
  }
  if (!std::filesystem::is_regular_file(path)) {
    state.SkipWithMessage("ONNX model variant is unavailable");
    return;
  }
  backend["engine_type"] = "ort";
  backend["data"] = path.string();
  backend["config"]["execution_provider"] = "cpu";
  backend["config"]["intra_op_num_threads"] = 1;
  backend["config"]["inter_op_num_threads"] = 1;
#endif
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
  auto label = type + ":" + entry["name"].get<hrt::String>();
#ifdef HRT_BENCH_RKNN
  label += quantized ? "/int8_model" : "/fp16_model";
#else
  label += "/onnx_fp32";
#endif
  state.SetLabel(label);
  state.counters["input_width"] = prepared->transforms.front().input_size.width;
  state.counters["input_height"] = prepared->transforms.front().input_size.height;
  std::size_t int8_outputs = 0, fp32_outputs = 0;
  for (const auto& output : prepared->outputs) {
    int8_outputs += output->data_type() == hrt::core::DataType::Int8;
    fp32_outputs += output->data_type() == hrt::core::DataType::Float32;
  }
  state.counters["int8_outputs"] = static_cast<double>(int8_outputs);
  state.counters["fp32_outputs"] = static_cast<double>(fp32_outputs);
#ifndef HRT_BENCH_RKNN
  if (fp32_outputs != prepared->outputs.size() || prepared->transforms.size() != static_cast<std::size_t>(batch)) {
    state.SkipWithError("ONNX outputs must be FP32 with the requested static batch");
    return;
  }
#endif
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
    const auto& name = algos[i]["name"].get_ref<const hrt::String&>();
    if (name.ends_with("_batch4") || name.ends_with("_batch8"))
      continue;
#ifdef HRT_BENCH_RKNN
    bench->Args({static_cast<std::int64_t>(i), 0, 1});
    for (const auto batch : {1, 4, 8})
      bench->Args({static_cast<std::int64_t>(i), 1, batch});
#else
    for (const auto batch : {1, 4, 8})
      bench->Args({static_cast<std::int64_t>(i), batch});
#endif
  }
}

} // namespace

BENCHMARK(BM_Type)->Apply(type_cases)->Unit(benchmark::kMillisecond)->UseRealTime();
