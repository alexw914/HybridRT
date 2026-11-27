// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tools/hrt_model_bench.cpp
// Description: Measures engine inference speed of every backend of one algo in
//              an .hrtc package or a JSON manifest on generated inputs.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/time/time.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <random>
#include <string_view>

namespace {

struct BenchOptions {
  int warmup = 10;
  int iterations = 100;
  double duration = 0.0; // Minimum benchmark duration in seconds; 0 uses iterations only.
  bool no_data_transfers = false;
};

struct Sample {
  double h2d = 0.0;
  double enqueue = 0.0;
  double compute = 0.0;
  double d2h = 0.0;
  double latency = 0.0;
};

struct PerfStats {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double total = 0.0;
};

auto elapsed_ms(const kitzoo::time::Stopwatch& watch) -> double {
  return watch.elapsed_as<std::chrono::duration<double, std::milli>>().count();
}

auto summarize(hrt::Vector<double>& values) -> PerfStats {
  std::sort(values.begin(), values.end());
  const auto total = std::accumulate(values.begin(), values.end(), 0.0);
  const auto percentile = [&](double p) -> double {
    return values[static_cast<std::size_t>(p * static_cast<double>(values.size() - 1))];
  };
  return {
      .min = values.front(),
      .max = values.back(),
      .mean = total / static_cast<double>(values.size()),
      .median = percentile(0.50),
      .p90 = percentile(0.90),
      .p95 = percentile(0.95),
      .p99 = percentile(0.99),
      .total = total,
  };
}

// Host buffer in the engine's fast host memory (e.g. CUDA-pinned) when available.
auto create_host_buffer(const hrt::engine::Engine& engine, const hrt::core::BufferProto& proto)
    -> hrt::SharedPtr<hrt::core::Buffer> {
  if (auto memory = engine.create_host_memory()) {
    auto buffer = std::make_shared<hrt::core::Buffer>(std::move(memory), proto.data_type, proto.shape);
    if (buffer->valid())
      return buffer;
  }
  auto buffer = std::make_shared<hrt::core::Buffer>(hrt::core::MemoryType::Cpu, proto.data_type, proto.shape);
  return buffer->valid() ? buffer : nullptr;
}

// Float32 inputs get uniform [0, 1) values, other types zeros.
auto fill_input(hrt::core::Buffer& buffer, std::mt19937& random) -> bool {
  auto* data = buffer.memory()->mutable_cpu();
  if (data == nullptr)
    return false;
  if (buffer.data_type() == hrt::core::DataType::Float32) {
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    auto* values = static_cast<float*>(data);
    for (std::int64_t i = 0; i < buffer.volume(); ++i)
      values[i] = uniform(random);
  } else {
    std::memset(data, 0, buffer.bytes());
  }
  return true;
}

auto log_stats(std::string_view label, const PerfStats& stats) -> void {
  KZ_LOG_INFO("  {:<10} min = {:.3f} ms, max = {:.3f} ms, mean = {:.3f} ms, median = {:.3f} ms, "
              "p90 = {:.3f} ms, p95 = {:.3f} ms, p99 = {:.3f} ms",
              label, stats.min, stats.max, stats.mean, stats.median, stats.p90, stats.p95, stats.p99);
}

// Times `options.iterations` (and at least `options.duration` s) runs after
// `options.warmup` untimed ones, reporting H2D, enqueue, compute, D2H and
// end-to-end latency statistics.
auto bench(const hrt::hrtc::AlgoParam& algo, const hrt::hrtc::EngineParam& param, const BenchOptions& options) -> bool {
  const auto name = algo.name + "/" + param.engine_type;
  const auto type = hrt::engine::engine_type_from(param.engine_type);
  const auto context = hrt::core::Context::create(hrt::engine::device_type_of(type));
  auto engine = hrt::engine::Engine::create(type);
  if (context == nullptr || engine == nullptr) {
    KZ_LOG_WARN("{}: skipped, not available in this build or on this machine", name);
    return true;
  }
  context->set_current();
  if (!engine->initialize(param)) {
    KZ_LOG_ERROR("{}: the model did not load", name);
    return false;
  }
  std::mt19937 random(0);
  hrt::core::Buffers inputs, outputs, host_inputs, host_outputs;
  bool has_device_input = false, has_device_output = false;
  for (const auto& proto : engine->input_protos()) {
    auto buffer = std::make_shared<hrt::core::Buffer>(proto);
    if (!buffer->valid()) {
      KZ_LOG_ERROR("{}: cannot allocate engine input", name);
      return false;
    }
    if (buffer->memory()->have_cpu()) {
      if (!fill_input(*buffer, random)) {
        KZ_LOG_ERROR("{}: cannot fill host input", name);
        return false;
      }
      host_inputs.push_back(nullptr);
    } else {
      has_device_input = true;
      auto host = create_host_buffer(*engine, proto);
      if (host == nullptr || !fill_input(*host, random) ||
          !hrt::core::Buffer::copy(*buffer, *host, hrt::core::CopyPolicy::DirectCopy)) {
        KZ_LOG_ERROR("{}: cannot prepare device input", name);
        return false;
      }
      host_inputs.push_back(std::move(host));
    }
    inputs.push_back(std::move(buffer));
  }
  for (const auto& proto : engine->output_protos()) {
    auto buffer = std::make_shared<hrt::core::Buffer>(proto);
    if (!buffer->valid()) {
      KZ_LOG_ERROR("{}: cannot allocate engine output", name);
      return false;
    }
    if (buffer->memory()->have_cpu()) {
      host_outputs.push_back(nullptr);
    } else {
      has_device_output = true;
      auto host = create_host_buffer(*engine, proto);
      if (host == nullptr) {
        KZ_LOG_ERROR("{}: cannot allocate host output", name);
        return false;
      }
      host_outputs.push_back(std::move(host));
    }
    outputs.push_back(std::move(buffer));
  }
  if (!engine->set_inputs(inputs) || !engine->set_outputs(outputs)) {
    KZ_LOG_ERROR("{}: cannot bind the engine IO", name);
    return false;
  }
  const auto stream = context->create_stream();
  const bool transfer_h2d = !options.no_data_transfers && has_device_input;
  const bool transfer_d2h = !options.no_data_transfers && has_device_output;
  const auto run_iteration = [&](Sample& sample) -> bool {
    const kitzoo::time::Stopwatch total_watch;
    if (transfer_h2d) {
      const kitzoo::time::Stopwatch h2d_watch;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (host_inputs[i] != nullptr &&
            !hrt::core::Buffer::copy(*inputs[i], *host_inputs[i], hrt::core::CopyPolicy::DirectCopy, stream.get()))
          return false;
      }
      if (stream != nullptr)
        stream->synchronize();
      sample.h2d = elapsed_ms(h2d_watch);
    }
    {
      const kitzoo::time::Stopwatch compute_watch;
      if (!engine->inference(stream.get()))
        return false;
      sample.enqueue = elapsed_ms(compute_watch);
      if (stream != nullptr)
        stream->synchronize();
      sample.compute = elapsed_ms(compute_watch);
    }
    if (transfer_d2h) {
      const kitzoo::time::Stopwatch d2h_watch;
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (host_outputs[i] != nullptr &&
            !hrt::core::Buffer::copy(*host_outputs[i], *outputs[i], hrt::core::CopyPolicy::DirectCopy, stream.get()))
          return false;
      }
      if (stream != nullptr)
        stream->synchronize();
      sample.d2h = elapsed_ms(d2h_watch);
    }
    sample.latency = elapsed_ms(total_watch);
    return true;
  };

  for (int i = 0; i < options.warmup; ++i) {
    Sample discarded;
    if (!run_iteration(discarded)) {
      KZ_LOG_ERROR("{}: inference failed", name);
      return false;
    }
  }
  hrt::Vector<double> latencies, enqueues, h2ds, computes, d2hs;
  const auto reserve = static_cast<std::size_t>(options.iterations);
  latencies.reserve(reserve);
  enqueues.reserve(reserve);
  computes.reserve(reserve);
  if (transfer_h2d)
    h2ds.reserve(reserve);
  if (transfer_d2h)
    d2hs.reserve(reserve);
  const double min_duration_ms = options.duration * 1000.0;
  const kitzoo::time::Stopwatch wall_watch;
  while (static_cast<int>(latencies.size()) < options.iterations || elapsed_ms(wall_watch) < min_duration_ms) {
    Sample sample;
    if (!run_iteration(sample)) {
      KZ_LOG_ERROR("{}: inference failed", name);
      return false;
    }
    latencies.push_back(sample.latency);
    enqueues.push_back(sample.enqueue);
    computes.push_back(sample.compute);
    if (transfer_h2d)
      h2ds.push_back(sample.h2d);
    if (transfer_d2h)
      d2hs.push_back(sample.d2h);
  }
  const double wall_ms = elapsed_ms(wall_watch);
  const auto runs = latencies.size();
  const auto batch = (!engine->input_protos().empty() && !engine->input_protos().front().shape.empty() &&
                      engine->input_protos().front().shape[0] > 0)
                         ? static_cast<std::size_t>(engine->input_protos().front().shape[0])
                         : std::size_t{1};
  const double qps = wall_ms > 0.0 ? static_cast<double>(runs) * 1000.0 / wall_ms : 0.0;
  const double fps = qps * static_cast<double>(batch);
  const auto latency_stats = summarize(latencies);
  const auto enqueue_stats = summarize(enqueues);
  const auto compute_stats = summarize(computes);
  const auto transfer_note = [&](bool has_device) -> std::string_view {
    return options.no_data_transfers ? "skipped (--no-data-transfers)" : (has_device ? "skipped" : "N/A (host memory)");
  };

  KZ_LOG_INFO("=== {} Performance summary (batch={}, runs={}, warmup={}) ===", name, batch, runs, options.warmup);
  KZ_LOG_INFO("  Throughput: {:.1f} qps ({:.1f} FPS)", qps, fps);
  log_stats("Latency:", latency_stats);
  log_stats("Enqueue:", enqueue_stats);
  if (transfer_h2d)
    log_stats("H2D:", summarize(h2ds));
  else
    KZ_LOG_INFO("  {:<10} {}", "H2D:", transfer_note(has_device_input));
  log_stats("Compute:", compute_stats);
  if (transfer_d2h)
    log_stats("D2H:", summarize(d2hs));
  else
    KZ_LOG_INFO("  {:<10} {}", "D2H:", transfer_note(has_device_output));
  KZ_LOG_INFO("  Total Host Walltime: {:.3f} s, Total Compute Time: {:.3f} s", wall_ms / 1000.0,
              compute_stats.total / 1000.0);
  return true;
}

} // namespace

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hrt-model-bench",
                                "Measure engine inference speed of one algo of an .hrtc package or manifest");
  auto add = options.add_options();
  add("m,model", ".hrtc package or .json manifest", kitzoo::core::value<hrt::String>());
  add("n,name", "Algo name", kitzoo::core::value<hrt::String>());
  add("e,engine", "Engine type, e.g. ort (default: all)", kitzoo::core::value<hrt::String>());
  add("i,iterations", "Minimum timed runs per engine", kitzoo::core::value<int>()->default_value("100"));
  add("w,warmup", "Untimed runs before timing", kitzoo::core::value<int>()->default_value("10"));
  add("d,duration", "Minimum benchmark duration in seconds", kitzoo::core::value<double>()->default_value("0"));
  add("no-data-transfers", "Disable H2D and D2H transfers during timing");
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  const BenchOptions bench_options{
      .warmup = args["warmup"].as<int>(),
      .iterations = args["iterations"].as<int>(),
      .duration = args["duration"].as<double>(),
      .no_data_transfers = args.count("no-data-transfers") != 0U,
  };
  if (args.count("help") != 0U || args.count("model") == 0U || args.count("name") == 0U ||
      bench_options.iterations < 1 || bench_options.warmup < 0 || bench_options.duration < 0.0) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0U ? 0 : 1;
  }

  const std::filesystem::path model = args["model"].as<hrt::String>();
  const auto package = model.extension() == ".json" ? hrt::hrtc::load_json(model) : hrt::hrtc::load(model);
  if (!package)
    return 1;
  const auto name = args["name"].as<hrt::String>();
  const auto* algo = package->find(name);
  if (algo == nullptr) {
    KZ_LOG_ERROR("no algo {} in {}", name, model.string());
    return 1;
  }
  const auto engine_type = args.count("engine") != 0U ? args["engine"].as<hrt::String>() : hrt::String{};
  if (!engine_type.empty() &&
      hrt::engine::engine_type_from(algo->engine.engine_type) != hrt::engine::engine_type_from(engine_type)) {
    KZ_LOG_ERROR("algo {} has no engine matching the engine filter", name);
    return 1;
  }
  return bench(*algo, algo->engine, bench_options) ? 0 : 1;
}
