// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/preprocess_bench.cpp
// Description: Benchmarks image preprocessing: format conversions checked
//              against OpenCV, YOLO task configs, and video resolutions.
// -----------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cmath>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_fixture.hpp"

namespace {

// Conversion, layout and resize+normalize cases on the bus image.
auto BM_Convert(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  cv::setNumThreads(1);
  const auto bgr = cv::imread((std::filesystem::path(HRT_SOURCE_DIR) / "assets/yolo/images/bus.jpg").string());
  if (bgr.empty()) {
    state.SkipWithError("bus.jpg is missing");
    return;
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  const auto index = state.range(0);
  const bool source_bgr = index == 2 || index == 7;
  const bool target_bgr = index == 1 || index == 4 || index == 6 || index == 8;
  const bool planar = index == 3 || index == 4 || index == 5 || index == 6 || index == 7;
  const bool normalized = index >= 5;
  const auto& pixels = source_bgr ? bgr : rgb;
  const auto source =
      Image::wrap(MemoryType::Cpu, pixels.data, source_bgr ? ImageFormat::InterBgr : ImageFormat::InterRgb,
                  {pixels.cols, pixels.rows}, DataType::Uint8, pixels.step);
  const auto format = planar ? (target_bgr ? ImageFormat::PlanarBgr : ImageFormat::PlanarRgb)
                             : (target_bgr ? ImageFormat::InterBgr : ImageFormat::InterRgb);
  const Size size = normalized ? Size{640, 480} : Size{pixels.cols, pixels.rows};
  Image target(MemoryType::Cpu, format, normalized ? DataType::Float32 : DataType::Uint8, size);
  ImagePreprocessorParam param;
  param.dst_image_format = format;
  param.normalize_type = normalized ? NormalizeType::Full : NormalizeType::None;
  param.std = normalized ? hrt::Array<float, 3>{255, 255, 255} : hrt::Array<float, 3>{1, 1, 1};
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  if (!source || !processor || !processor->process(target, *source, {}, param)) {
    state.SkipWithError("Preprocessing failed");
    return;
  }
  cv::Mat expected;
  cv::resize(target_bgr ? bgr : rgb, expected, {size.width, size.height});
  const auto* bytes = static_cast<const std::uint8_t*>(target.memory()->cpu());
  const auto* floats = static_cast<const float*>(target.memory()->cpu());
  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        const auto offset = planar ? (c * size.height + y) * size.width + x : (y * size.width + x) * 3 + c;
        const float actual = normalized ? floats[offset] * 255.0F : bytes[offset];
        if (std::abs(actual - expected.at<cv::Vec3b>(y, x)[c]) > 1.01F) {
          state.SkipWithError("Output differs from OpenCV reference");
          return;
        }
      }
    }
  }
  static constexpr hrt::Array<const char*, 9> kLabels{"RGB_to_RGB_u8",
                                                      "RGB_to_BGR_u8",
                                                      "BGR_to_RGB_u8",
                                                      "RGB_to_planar_RGB_u8",
                                                      "RGB_to_planar_BGR_u8",
                                                      "RGB_to_planar_RGB_f32_resize_normalize",
                                                      "RGB_to_planar_BGR_f32_resize_normalize",
                                                      "BGR_to_planar_RGB_f32_resize_normalize",
                                                      "RGB_to_BGR_f32_resize_normalize"};
  state.SetLabel(kLabels.at(static_cast<std::size_t>(index)));
  for (int i = 0; i < 20; ++i) {
    if (!processor->process(target, *source, {}, param)) {
      state.SkipWithError("Warmup failed");
      return;
    }
  }
  for (auto _ : state) {
    if (!processor->process(target, *source, {}, param)) {
      state.SkipWithError("Preprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(target.memory()->cpu());
  }
}

// The preprocess config of each YOLO task, from its test image into the
// 640x480 planar float input of the models.
auto BM_Task(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const auto task = static_cast<std::size_t>(state.range(0));
  state.SetLabel(hrt::bench::kTasks.at(task));
  const auto entry = hrt::bench::algo_entry("yolo.json", task);
  const auto bgr = hrt::bench::task_image(task);
  const auto source = hrt::bench::wrap_bgr(bgr);
  if (!entry.is_object() || !source) {
    state.SkipWithError("assets/yolo manifest or image is missing");
    return;
  }
  const auto param = entry["config"]["preprocess"].get<hrt::preprocess::ImagePreprocessorParam>();
  hrt::core::Image target(hrt::core::MemoryType::Cpu, param.dst_image_format, hrt::core::DataType::Float32, {640, 480});
  auto processor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  for (auto _ : state) {
    if (!processor->process(target, *source, {}, param)) {
      state.SkipWithError("Preprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(target.memory()->cpu());
  }
}

auto BM_Resolution(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  cv::setNumThreads(1);
  const Size source_size{static_cast<int>(state.range(0)), static_cast<int>(state.range(1))};
  const Size target_size{static_cast<int>(state.range(2)), static_cast<int>(state.range(3))};
  cv::Mat pixels(source_size.height, source_size.width, CV_8UC3);
  cv::randu(pixels, 0, 256);
  const auto source =
      Image::wrap(MemoryType::Cpu, pixels.data, ImageFormat::InterBgr, source_size, DataType::Uint8, pixels.step);
  Image target(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, target_size);
  const auto type = state.range(4) == 0 ? ImagePreprocessorType::Cpu : ImagePreprocessorType::Metal;
  auto processor = ImagePreprocessor::create(type);
  state.SetLabel(type == ImagePreprocessorType::Cpu ? "cpu" : "metal");
  if (!processor) {
    state.SkipWithMessage("Requested preprocessing device is unavailable");
    return;
  }
  ImagePreprocessorParam param;
  param.resize_type = ResizeType::LetterBox;
  param.std = {255, 255, 255};
  param.default_color = {114, 114, 114};
  if (!source || !target.valid()) {
    state.SkipWithError("Cannot allocate preprocessing buffers");
    return;
  }
  for (int i = 0; i < 20; ++i) {
    if (!processor->process(target, *source, {}, param)) {
      state.SkipWithError("Warmup failed");
      return;
    }
  }
  for (auto _ : state) {
    if (!processor->process(target, *source, {}, param)) {
      state.SkipWithError("Preprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(target.memory()->cpu());
  }
}

auto resolution_cases(benchmark::Benchmark* bench) -> void {
  for (const int width : {1920, 3840}) {
    for (const auto size : {hrt::core::Size{640, 360}, hrt::core::Size{640, 640}, hrt::core::Size{960, 540},
                            hrt::core::Size{960, 960}, hrt::core::Size{1280, 720}, hrt::core::Size{1280, 1280}}) {
      bench->Args({width, width * 9 / 16, size.width, size.height, 0});
#ifdef __APPLE__
      bench->Args({width, width * 9 / 16, size.width, size.height, 1});
#endif
    }
  }
}

} // namespace

BENCHMARK(BM_Convert)->DenseRange(0, 8)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Task)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Resolution)
    ->Apply(resolution_cases)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MeasureProcessCPUTime();
