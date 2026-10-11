// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tools/hrt_model_bench.cpp
// Description: Measures engine inference speed of every algo and backend in an
//              .hrtc package or a JSON manifest on generated inputs.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <random>

namespace {

// Float32 inputs get uniform [0, 1) values, other types zeros.
auto make_input(const hrt::core::BufferProto& proto, std::mt19937& random) -> hrt::SharedPtr<hrt::core::Buffer> {
  hrt::core::Buffer host(hrt::core::MemoryType::Cpu, proto.data_type, proto.shape);
  auto* data = host.valid() ? host.memory()->mutable_cpu() : nullptr;
  if (data == nullptr)
    return nullptr;
  if (proto.data_type == hrt::core::DataType::Float32) {
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    auto* values = static_cast<float*>(data);
    for (std::int64_t i = 0; i < host.volume(); ++i)
      values[i] = uniform(random);
  } else {
    std::memset(data, 0, host.byte_size());
  }
  auto buffer = std::make_shared<hrt::core::Buffer>(proto);
  if (!buffer->valid() || !hrt::core::Buffer::copy(*buffer, host, hrt::core::CopyPolicy::DirectCopy))
    return nullptr;
  return buffer;
}

// Times `iterations` runs of one engine after `warmup` untimed ones; each run
// includes stream completion.
auto bench(const hrt::hrtc::AlgoParam& algo, const hrt::hrtc::EngineParam& param, int warmup, int iterations) -> bool {
  const auto name = algo.name + "/" + param.engine_type;
  const auto type = hrt::engine::engine_type_from(param.engine_type);
  const auto context = hrt::core::Context::create(hrt::engine::device_type_of(type));
  auto engine = hrt::engine::Engine::create(type);
  if (context == nullptr || engine == nullptr) {
    KZ_LOG_WARN("{}: skipped, not available in this build or on this machine", name);
    return true;
  }
  context->make_current();
  if (!engine->initialize(param)) {
    KZ_LOG_ERROR("{}: the model did not load", name);
    return false;
  }
  std::mt19937 random(0);
  hrt::core::Buffers inputs;
  hrt::core::Buffers outputs;
  for (const auto& proto : engine->input_protos())
    inputs.push_back(make_input(proto, random));
  for (const auto& proto : engine->output_protos())
    outputs.push_back(std::make_shared<hrt::core::Buffer>(proto));
  const auto allocated = [](const hrt::core::Buffers& buffers) -> bool {
    return std::all_of(buffers.begin(), buffers.end(), [](const auto& b) { return b != nullptr && b->valid(); });
  };
  if (!allocated(inputs) || !allocated(outputs) || !engine->set_inputs(inputs) || !engine->set_outputs(outputs)) {
    KZ_LOG_ERROR("{}: cannot allocate or bind the engine IO", name);
    return false;
  }
  const auto stream = context->create_stream();
  const auto run = [&]() -> bool {
    if (!engine->inference(stream.get()))
      return false;
    if (stream != nullptr)
      stream->synchronize();
    return true;
  };

  for (int i = 0; i < warmup; ++i) {
    if (!run()) {
      KZ_LOG_ERROR("{}: inference failed", name);
      return false;
    }
  }
  hrt::Vector<double> times(static_cast<std::size_t>(iterations));
  for (auto& time : times) {
    const auto start = std::chrono::steady_clock::now();
    if (!run()) {
      KZ_LOG_ERROR("{}: inference failed", name);
      return false;
    }
    time = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::sort(times.begin(), times.end());
  const auto mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
  const auto percentile = [&](double p) -> double {
    return times[static_cast<std::size_t>(p * static_cast<double>(times.size() - 1))];
  };
  KZ_LOG_INFO("{}: mean {:.3f} ms, min {:.3f}, p50 {:.3f}, p99 {:.3f}, max {:.3f}, {:.1f} FPS ({} runs)", name, mean,
              times.front(), percentile(0.5), percentile(0.99), times.back(), 1000.0 / mean, times.size());
  return true;
}

} // namespace

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hrt-model-bench", "Measure engine inference speed of an .hrtc package or manifest");
  auto add = options.add_options();
  add("m,model", ".hrtc package or .json manifest", kitzoo::core::value<hrt::String>());
  add("a,algo", "Algo name (default: all)", kitzoo::core::value<hrt::String>());
  add("e,engine", "Engine type, e.g. ort (default: all)", kitzoo::core::value<hrt::String>());
  add("n,iterations", "Timed runs per engine", kitzoo::core::value<int>()->default_value("100"));
  add("w,warmup", "Untimed runs before timing", kitzoo::core::value<int>()->default_value("10"));
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  const auto iterations = args["iterations"].as<int>();
  const auto warmup = args["warmup"].as<int>();
  if (args.count("help") != 0U || args.count("model") == 0U || iterations < 1 || warmup < 0) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0U ? 0 : 1;
  }

  const std::filesystem::path model = args["model"].as<hrt::String>();
  const auto package = model.extension() == ".json" ? hrt::hrtc::load_json(model) : hrt::hrtc::load(model);
  if (!package)
    return 1;
  const auto algo_name = args.count("algo") != 0U ? args["algo"].as<hrt::String>() : hrt::String{};
  const auto engine_type = args.count("engine") != 0U ? args["engine"].as<hrt::String>() : hrt::String{};
  bool ok = true;
  bool matched = false;
  for (const auto& algo : package->algos) {
    if (!algo_name.empty() && algo.name != algo_name)
      continue;
    for (const auto& param : algo.engines) {
      if (!engine_type.empty() &&
          hrt::engine::engine_type_from(param.engine_type) != hrt::engine::engine_type_from(engine_type))
        continue;
      matched = true;
      ok = bench(algo, param, warmup, iterations) && ok;
    }
  }
  if (!matched)
    KZ_LOG_ERROR("no engine of the package matches the algo and engine filters");
  return ok && matched ? 0 : 1;
}
