// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/preprocess_bench.cpp
// Description: Benchmarks image preprocessing: format conversions checked
//              against OpenCV, YOLO task and OCR configs, and video resolutions.
// -----------------------------------------------------------------------------

#include <hrt/core/context.hpp>

#include <benchmark/benchmark.h>
#include <cmath>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_fixture.hpp"

namespace {

// Establish a padded image layout, then reuse it with explicit or default strides.
auto BM_ImageLayout(benchmark::State& state) -> void {
  using namespace hrt::core;
  constexpr Size kSize{640, 480};
  constexpr std::size_t kRowPitch = 1984;
  hrt::Vector<std::uint8_t> pixels(kRowPitch * kSize.height);
  if (state.range(0) == 0) {
    state.SetLabel("wrap_padded_rgb_u8");
    for (auto _ : state) {
      auto image =
          Image::wrap(MemoryType::Cpu, pixels.data(), ImageFormat::InterRgb, kSize, DataType::Uint8, kRowPitch);
      if (!image) {
        state.SkipWithError("Image wrapping failed");
        break;
      }
      benchmark::DoNotOptimize(image->memory()->cpu());
    }
    return;
  }
  auto image = Image::wrap(MemoryType::Cpu, pixels.data(), ImageFormat::InterRgb, kSize, DataType::Uint8, kRowPitch);
  if (!image) {
    state.SkipWithError("Image wrapping failed");
    return;
  }
  if (state.range(0) == 1) {
    state.SetLabel("resize_same_explicit_stride");
    const auto shape = image->memory()->shape();
    const auto stride = image->stride();
    for (auto _ : state) {
      if (!image->memory()->resize(shape, stride)) {
        state.SkipWithError("Memory resize failed");
        break;
      }
      benchmark::DoNotOptimize(image->memory()->cpu());
    }
  } else {
    state.SetLabel("reshape_same_default_stride");
    for (auto _ : state) {
      if (!image->reshape(kSize)) {
        state.SkipWithError("Image reshape failed");
        break;
      }
      benchmark::DoNotOptimize(image->memory()->cpu());
    }
  }
}

// Conversion, layout and resize+normalize cases on the bus image.
auto BM_Convert(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  cv::setNumThreads(1);
  const auto bgr = cv::imread((std::filesystem::path(HRT_SOURCE_DIR) / "assets/test/images/bus.jpg").string());
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
  const auto entry = hrt::bench::algo_entry(task, hrt::bench::Provider::Cpu, false);
  const auto bgr = hrt::bench::task_image(task);
  const auto source = hrt::bench::wrap_bgr(bgr);
  if (!entry.is_object() || !source) {
    state.SkipWithError("assets/test/images is missing");
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

// OCR preprocess configs on the document image: 0 detection into 640x640, 1 recognition
// text_line of a 241x50 text line region into 48x320.
auto BM_Ocr(benchmark::State& state) -> void {
  cv::setNumThreads(1);
  const auto stage = static_cast<std::size_t>(state.range(0));
  state.SetLabel(stage == 0 ? "detection" : "recognition");
  const auto entry = hrt::bench::template_entry(hrt::bench::kOcrAlgos.at(stage), hrt::bench::Provider::Cpu, false);
  const auto bgr = hrt::bench::read_image(hrt::bench::kOcrImage);
  const auto source = hrt::bench::wrap_bgr(bgr);
  if (!entry.is_object() || !source) {
    state.SkipWithError("assets/test/images/ocr_document.png is missing");
    return;
  }
  const auto param = entry["config"]["preprocess"].get<hrt::preprocess::ImagePreprocessorParam>();
  hrt::core::Image target(hrt::core::MemoryType::Cpu, param.dst_image_format, hrt::core::DataType::Float32,
                          stage == 0 ? hrt::core::Size{640, 640} : hrt::core::Size{320, 48});
  auto processor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  for (auto _ : state) {
    const auto region = stage == 0 ? hrt::core::Rect{} : hrt::core::Rect{419, 16, 241, 50};
    if (!processor->process(target, *source, region, param)) {
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
  for (const int width : {1920, 2560, 3840}) {
    for (const auto size : {hrt::core::Size{640, 360}, hrt::core::Size{640, 640}, hrt::core::Size{960, 540},
                            hrt::core::Size{960, 960}, hrt::core::Size{1280, 720}, hrt::core::Size{1280, 1280}}) {
      bench->Args({width, width * 9 / 16, size.width, size.height, 0});
#ifdef __APPLE__
      bench->Args({width, width * 9 / 16, size.width, size.height, 1});
#endif
    }
  }
}

// CUDA letterbox into device memory: 0 uploads a host frame first, 1 starts from a device frame, 2 only uploads,
// 3 crops a 480x960 person-sized region of a host frame, as a cascaded classifier does.
auto BM_CudaResolution(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto context = Context::create(DeviceType::Cuda);
  if (context == nullptr) {
    state.SkipWithMessage("CUDA device is unavailable");
    return;
  }
  context->set_current();
  auto stream = context->create_stream();
  const Size source_size{static_cast<int>(state.range(0)), static_cast<int>(state.range(1))};
  const Size target_size{static_cast<int>(state.range(2)), static_cast<int>(state.range(3))};
  const auto mode = state.range(4);
  constexpr const char* kLabels[] = {"host_source", "device_source", "upload_only", "host_crop"};
  state.SetLabel(kLabels[mode]);
  cv::Mat pixels(source_size.height, source_size.width, CV_8UC3);
  cv::randu(pixels, 0, 256);
  const auto host =
      Image::wrap(MemoryType::Cpu, pixels.data, ImageFormat::InterBgr, source_size, DataType::Uint8, pixels.step);
  Image device(MemoryType::Cuda, ImageFormat::InterBgr, DataType::Uint8, source_size);
  Image target(MemoryType::Cuda, ImageFormat::PlanarRgb, DataType::Float32, target_size);
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Cuda);
  if (!host || !device.valid() || !target.valid() || !processor ||
      !Buffer::copy(*device.buffer(), *host->buffer(), CopyPolicy::DirectCopy, stream.get())) {
    state.SkipWithError("Cannot prepare CUDA preprocessing");
    return;
  }
  ImagePreprocessorParam param;
  param.resize_type = ResizeType::LetterBox;
  param.std = {255, 255, 255};
  param.default_color = {114, 114, 114};
  const auto& source = mode == 1 ? device : *host;
  const Rect crop = mode == 3 ? Rect{source_size.width / 2 - 240, source_size.height / 2 - 480, 480, 960} : Rect{};
  auto step = [&]() -> bool {
    const auto ok = mode == 2 ? Buffer::copy(*device.buffer(), *host->buffer(), CopyPolicy::DirectCopy, stream.get())
                              : processor->process(target, source, crop, param, stream.get()).has_value();
    stream->synchronize();
    return ok;
  };
  for (int i = 0; i < 20; ++i) {
    if (!step()) {
      state.SkipWithError("Warmup failed");
      return;
    }
  }
  for (auto _ : state) {
    if (!step()) {
      state.SkipWithError("Preprocessing failed");
      break;
    }
  }
}

// One batched call letterboxing 1920x1080 frames into 640x640 inputs; args: batch size, mode (0 CPU with one OpenCV
// thread, 1 CPU with four, 2 CUDA from host frames, 3 CUDA from device frames, 4 and 5 the same from NV12 frames,
// 6 RGA from host frames and 7 from dma-buf frames into uint8 RGB dma-bufs, as RKNN image inputs take them).
// Items are frames.
auto BM_Batch(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  const auto batch = static_cast<std::size_t>(state.range(0));
  const auto mode = state.range(1);
  constexpr const char* kLabels[] = {"cpu_1_thread",       "cpu_4_threads",  "cuda_host_source",
                                     "cuda_device_source", "cuda_host_nv12", "cuda_device_nv12",
                                     "rga_host_source",    "rga_dma_source", "metal_host_source"};
  state.SetLabel(kLabels[mode]);
  const auto cuda = mode >= 2 && mode <= 5;
  const auto rga = mode == 6 || mode == 7;
  const auto metal = mode == 8;
  cv::setNumThreads(mode == 1 ? 4 : 1);
  auto context = Context::create(cuda ? DeviceType::Cuda : DeviceType::Cpu);
  if (context == nullptr) {
    state.SkipWithMessage("CUDA device is unavailable");
    return;
  }
  context->set_current();
  auto stream = context->create_stream();
  auto processor = ImagePreprocessor::create(cuda    ? ImagePreprocessorType::Cuda
                                             : rga   ? ImagePreprocessorType::Rga
                                             : metal ? ImagePreprocessorType::Metal
                                                     : ImagePreprocessorType::Cpu);
  if (processor == nullptr) {
    state.SkipWithMessage("The preprocessor is not available in this build");
    return;
  }
  const Size source_size{1920, 1080};
  hrt::Vector<cv::Mat> frames(batch);
  hrt::Vector<Image> sources;
  hrt::Vector<Image> targets;
  for (auto& frame : frames) {
    const auto nv12 = mode == 4 || mode == 5;
    frame.create(nv12 ? source_size.height * 3 / 2 : source_size.height, source_size.width, nv12 ? CV_8UC1 : CV_8UC3);
    cv::randu(frame, 0, 256);
    const auto format = nv12 ? ImageFormat::Nv12 : ImageFormat::InterBgr;
    auto host = Image::wrap(MemoryType::Cpu, frame.data, format, source_size, DataType::Uint8, frame.step);
    if (!host || !processor) {
      state.SkipWithError("Cannot prepare batch preprocessing");
      return;
    }
    if (mode == 3 || mode == 5 || mode == 7) {
      Image device(mode == 7 ? MemoryType::DmaBuffer : MemoryType::Cuda, format, DataType::Uint8, source_size);
      if (!device.valid() || !Buffer::copy(*device.buffer(), *host->buffer(), CopyPolicy::DirectCopy, stream.get())) {
        state.SkipWithError("Cannot upload a frame");
        return;
      }
      sources.push_back(std::move(device));
    } else {
      sources.push_back(std::move(*host));
    }
    if (rga)
      targets.emplace_back(MemoryType::DmaBuffer, ImageFormat::InterRgb, DataType::Uint8, Size{640, 640});
    else
      targets.emplace_back(cuda ? MemoryType::Cuda : MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32,
                           Size{640, 640});
  }
  ImagePreprocessorParam param;
  param.resize_type = ResizeType::LetterBox;
  param.dst_image_format = rga ? ImageFormat::InterRgb : ImageFormat::PlanarRgb;
  param.std = rga ? hrt::Array<float, 3>{1, 1, 1} : hrt::Array<float, 3>{255, 255, 255};
  param.default_color = {114, 114, 114};
  auto step = [&]() -> bool {
    const auto ok = processor->process(targets, sources, {}, param, stream.get()).has_value();
    stream->synchronize();
    return ok;
  };
  for (int i = 0; i < 20; ++i) {
    if (!step()) {
      state.SkipWithError("Warmup failed");
      return;
    }
  }
  for (auto _ : state) {
    if (!step()) {
      state.SkipWithError("Preprocessing failed");
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
  cv::setNumThreads(1);
}

auto cuda_cases(benchmark::Benchmark* bench) -> void {
  for (const int width : {1920, 2560, 3840})
    for (const auto size : {hrt::core::Size{640, 640}, hrt::core::Size{1280, 1280}})
      for (const int mode : {0, 1, 2})
        bench->Args({width, width * 9 / 16, size.width, size.height, mode});
  bench->Args({3840, 2160, 224, 224, 0});
  bench->Args({3840, 2160, 224, 224, 3});
}

// RKNN image inputs stay interleaved uint8; their normalization belongs to the model.
auto BM_RgaResolution(benchmark::State& state) -> void {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  cv::setNumThreads(1);
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Rga);
  if (!processor) {
    state.SkipWithMessage("RGA preprocessing is unavailable");
    return;
  }
  const Size source_size{static_cast<int>(state.range(0)), static_cast<int>(state.range(1))};
  const Size target_size{static_cast<int>(state.range(2)), static_cast<int>(state.range(3))};
  const auto resident = state.range(4) != 0;
  state.SetLabel(resident ? "rga_dma_source_rgb_u8" : "rga_host_source_rgb_u8");
  cv::Mat pixels(source_size.height, source_size.width, CV_8UC3);
  cv::randu(pixels, 0, 256);
  auto host =
      Image::wrap(MemoryType::Cpu, pixels.data, ImageFormat::InterBgr, source_size, DataType::Uint8, pixels.step);
  Image device(MemoryType::DmaBuffer, ImageFormat::InterBgr, DataType::Uint8, source_size);
  Image target(MemoryType::DmaBuffer, ImageFormat::InterRgb, DataType::Uint8, target_size);
  if (!host || !device.valid() || !target.valid() ||
      !Buffer::copy(*device.buffer(), *host->buffer(), CopyPolicy::DirectCopy)) {
    state.SkipWithError("Cannot prepare RGA dma-buffers");
    return;
  }
  if (!processor->supports(ImageOperation::Fill, target, target) ||
      !processor->supports(ImageOperation::Resize, target, device)) {
    state.SkipWithError("RGA buffers do not satisfy hardware format, stride or address limits");
    return;
  }
  ImagePreprocessorParam param;
  param.dst_image_format = ImageFormat::InterRgb;
  param.resize_type = ResizeType::LetterBox;
  param.normalize_type = NormalizeType::None;
  param.default_color = {114, 114, 114};
  const auto& source = resident ? device : *host;
  // RGA operators use IM_SYNC; the returned dma-buffer is ready for its consumer.
  for (int i = 0; i < 20; ++i) {
    if (!processor->process(target, source, {}, param)) {
      state.SkipWithError("RGA warmup failed");
      return;
    }
  }
  for (auto _ : state) {
    if (!processor->process(target, source, {}, param)) {
      state.SkipWithError("RGA preprocessing failed");
      break;
    }
    benchmark::DoNotOptimize(target.memory()->cpu());
  }
}

auto rga_cases(benchmark::Benchmark* bench) -> void {
  for (const int width : {1920, 2560, 3840})
    for (const auto size : {hrt::core::Size{640, 640}, hrt::core::Size{1280, 1280}})
      for (const int resident : {0, 1})
        bench->Args({width, width * 9 / 16, size.width, size.height, resident});
}

} // namespace

BENCHMARK(BM_ImageLayout)->DenseRange(0, 2)->Unit(benchmark::kNanosecond)->UseRealTime();
BENCHMARK(BM_Convert)->DenseRange(0, 8)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Task)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Ocr)->DenseRange(0, 1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Resolution)
    ->Apply(resolution_cases)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MeasureProcessCPUTime();
BENCHMARK(BM_CudaResolution)->Apply(cuda_cases)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_RgaResolution)->Apply(rga_cases)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Batch)
    ->ArgsProduct({{1, 4}, {0, 1, 2, 3, 4, 5, 6, 7}})
    ->Args({1, 8})
    ->Args({4, 8})
    ->Args({8, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
