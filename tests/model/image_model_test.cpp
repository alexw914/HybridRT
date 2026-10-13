// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/model/image_model_test.cpp
// Description: Tests the pooled image model pipeline on ONNX Runtime and TensorRT.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#ifdef HRT_TEST_TENSORRT
#include <kitzoo/log/logger.hpp>
#endif
#include <kitzoo/os/filesys.hpp>

#ifdef HRT_TEST_TENSORRT
#include <NvInfer.h>
#include <algorithm>
#endif
#include <cmath>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>

#ifdef HRT_TEST_ORT
#include "../engine/onnx_model.hpp"

namespace {

// `type` names the postprocessor.
auto classify_algo(std::string_view preprocess, std::string_view type = "GeneralClassify") -> hrt::hrtc::AlgoParam {
  const auto config = kitzoo::json::Json{{"preprocess", kitzoo::json::Json::parse(preprocess)},
                                         {"postprocess", {{"type", type}, {"classify_postprocess_type", "none"}}}};
  return {
      .name = "classify",
      .engines = {{.engine_type = "ort",
                   .config = {},
                   .proto = {},
                   .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::test::unary_model("Identity", {1, 3, 2, 2}))}},
      .config = config.dump()};
}

class ImageModelTest : public ::testing::Test {
protected:
  hrt::SharedPtr<hrt::core::Context> cpu_ = hrt::core::Context::create(hrt::core::DeviceType::Cpu);
};

// Uniform BGR image: every input value is its channel / 255.
auto uniform_bgr(hrt::Vector<std::uint8_t>& pixels, std::uint8_t b, std::uint8_t g, std::uint8_t r,
                 hrt::core::Size size = {8, 8}) -> hrt::core::Image {
  pixels.resize(static_cast<std::size_t>(size.width * size.height * 3));
  for (std::size_t i = 0; i < pixels.size(); i += 3) {
    pixels[i] = b;
    pixels[i + 1] = g;
    pixels[i + 2] = r;
  }
  return *hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr, size);
}

TEST_F(ImageModelTest, RunsPreprocessInferenceAndPostprocess) {
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_);
  ASSERT_NE(model, nullptr);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1U);
  ASSERT_EQ(result->front().size(), 12U);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][4], 20.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][11], 10.0F / 255.0F);
}

TEST_F(ImageModelTest, CreatesFromAlgoJson) {
  const auto dir = std::filesystem::temp_directory_path() / "hybridrt_image_model_json";
  std::filesystem::create_directories(dir / "models");
  const auto model_bytes = hrt::test::unary_model("Identity", {1, 3, 2, 2});
  kitzoo::os::write_file(dir / "models" / "identity.onnx",
                         std::string_view(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size()));
  // One entry of a manifest's "algos"; an absolute model path is read as is.
  auto algo = kitzoo::json::Json::parse(R"({
    "name": "classify",
    "config": {
      "preprocess": {"std": [255, 255, 255]},
      "engine": {"backends": [{"engine_type": "ort", "config": {"intra_op_num_threads": 1}}]},
      "postprocess": {"type": "GeneralClassify"}
    }
  })");
  algo["config"]["engine"]["backends"][0]["data"] = (dir / "models" / "identity.onnx").string();
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(algo, cpu_);
  std::filesystem::remove_all(dir);
  ASSERT_NE(model, nullptr);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  ASSERT_EQ(result->front().size(), 12U);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][11], 10.0F / 255.0F);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  EXPECT_DEATH(Model::create(kitzoo::json::Json::array(), cpu_), "");
  auto missing = algo;
  missing["config"]["engine"]["backends"][0]["data"] = (dir / "models" / "missing.onnx").string();
  EXPECT_DEATH(Model::create(missing, cpu_), "");
}

TEST_F(ImageModelTest, ServesConcurrentCallsFromPools) {
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_,
      {.preprocess = 3, .engine = 2, .postprocess = 3});
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->options().engine, 2U);

  hrt::Vector<std::future<bool>> runs;
  for (int t = 0; t < 4; ++t) {
    runs.push_back(std::async(std::launch::async, [&model, t] {
      hrt::Vector<std::uint8_t> pixels;
      const auto value = static_cast<std::uint8_t>(40 * t);
      for (int i = 0; i < 20; ++i) {
        const auto result = model->process_sync(uniform_bgr(pixels, value, value, value, {3 + i + t, 5 + (i % 7)}));
        if (!result || !(std::abs((*result)[0][5] - static_cast<float>(value) / 255.0F) <= 1.0e-7F))
          return false;
      }
      return true;
    }));
  }
  for (auto& run : runs)
    EXPECT_TRUE(run.get());
}

TEST_F(ImageModelTest, HandlesFrameSizeChangesAndRecoversAfterInvalidInput) {
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_);
  for (const auto size : {hrt::core::Size{8, 8}, {17, 5}, {3, 21}, {8, 8}}) {
    hrt::core::Image invalid(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr);
    EXPECT_FALSE(model->process_sync(invalid));
    hrt::Vector<std::uint8_t> pixels(static_cast<std::size_t>(size.width * size.height * 3), 64);
    const auto image =
        hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr, size);
    ASSERT_TRUE(image);
    const auto result = model->process_sync(*image);
    ASSERT_TRUE(result);
    ASSERT_EQ(result->size(), 1U);
    ASSERT_EQ(result->front().size(), 12U);
    for (const auto value : result->front())
      EXPECT_FLOAT_EQ(value, 64.0F / 255.0F);
  }
}

TEST_F(ImageModelTest, PicksTheFirstUsableEngine) {
  // TensorRT is not usable here, so the model falls back to the next engine.
  auto algo = classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})");
  auto trt = algo.engines.front();
  trt.engine_type = "trt";
  algo.engines.insert(algo.engines.begin(), trt);
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(algo);
  EXPECT_EQ(model->context()->device_type(), hrt::core::DeviceType::Cpu);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
}

TEST_F(ImageModelTest, ChecksConfigWithoutThrowing) {
  using hrt::model::check_config;
  EXPECT_TRUE(check_config(classify_algo(R"({"resize_type": "center_keep_ratio"})")).empty());
  EXPECT_EQ(check_config(classify_algo(R"({"resize_typ": "normal"})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo(R"({"resize_type": "stretch"})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo(R"({"mean": [0, 0]})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo("{}", "Mystery")).size(), 1U);
  // Stage counts are ImageModelOptions and engine settings each engine's
  // config, so neither belongs in the algo config.
  const auto with = [](const char* section, const kitzoo::json::Json& value) -> hrt::hrtc::AlgoParam {
    auto algo = classify_algo("{}");
    auto config = kitzoo::json::Json::parse(algo.config);
    config[section] = value;
    algo.config = config.dump();
    return algo;
  };
  EXPECT_EQ(check_config(with("preprocess", {{"num", 2}})).size(), 1U);
  EXPECT_EQ(check_config(with("postprocess", {{"type", "GeneralClassify"}, {"num", 2}})).size(), 1U);
  EXPECT_EQ(check_config(with("engine", {{"num", 2}})).size(), 1U);
}

TEST_F(ImageModelTest, CreatesContextForEngineAndPreprocessor) {
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  // A CPU engine with CPU preprocessing gets a CPU context.
  const auto cpu_model = Model::create(classify_algo("{}"));
  ASSERT_NE(cpu_model, nullptr);
  EXPECT_EQ(cpu_model->context()->device_type(), hrt::core::DeviceType::Cpu);

  // A CPU engine takes the context of a device preprocessor.
  const auto cuda_algo = classify_algo(R"({"device": "cuda"})");
  if (hrt::core::Context::create(hrt::core::DeviceType::Cuda) == nullptr) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(Model::create(cuda_algo), "");
  } else {
    EXPECT_EQ(Model::create(cuda_algo)->context()->device_type(), hrt::core::DeviceType::Cuda);
  }
}

TEST_F(ImageModelTest, AbortsOnUnusableConfigurations) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  EXPECT_DEATH(Model::create(classify_algo("{}", "Mystery"), cpu_), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), cpu_, {.preprocess = 1, .engine = 0, .postprocess = 1}), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), cpu_, {.preprocess = 65, .engine = 1, .postprocess = 1}), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), hrt::SharedPtr<hrt::core::Context>{}), "");

  auto algo = classify_algo("{}");
  algo.engines[0].engine_type = "unknown";
  EXPECT_DEATH(Model::create(algo, cpu_), "");

  // Engine settings are checked when the engine loads.
  auto bad_config = classify_algo("{}");
  bad_config.engines[0].config = R"({"intra_op_threads": 2})";
  EXPECT_DEATH(Model::create(bad_config, cpu_), "");

  // An interleaved target cannot feed an NCHW input.
  EXPECT_DEATH(Model::create(classify_algo(R"({"dst_image_format": "inter_rgb"})"), cpu_), "");
}

} // namespace
#endif

#ifdef HRT_TEST_TENSORRT
namespace {

class TestLogger final : public nvinfer1::ILogger {
public:
  auto log(Severity severity, const char* message) noexcept -> void override {
    if (severity <= Severity::kERROR)
      KZ_LOG_ERROR("tensorrt test: {}", message);
  }
};

#ifndef HRT_TEST_ORT
class ImageModelTest : public ::testing::Test {};
#endif

TEST_F(ImageModelTest, KeepsCudaPreprocessorScratchAliveUntilStreamCompletes) {
  auto first_context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (first_context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  first_context->make_current();

  TestLogger logger;
  hrt::UniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  ASSERT_NE(builder, nullptr);
#if HRT_TENSORRT_VERSION < 100000
  constexpr auto kNetworkFlags =
      1U << static_cast<unsigned int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
  constexpr auto kNetworkFlags = 0U;
#endif
  hrt::UniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(kNetworkFlags));
  hrt::UniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  ASSERT_NE(network, nullptr);
  ASSERT_NE(config, nullptr);
  auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 3, 1024, 1024});
  ASSERT_NE(input, nullptr);
  auto* relu = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
  ASSERT_NE(relu, nullptr);
  network->markOutput(*relu->getOutput(0));
  hrt::UniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
  ASSERT_NE(serialized, nullptr);
  const auto* begin = static_cast<const std::uint8_t*>(serialized->data());
  const auto bytes = std::make_shared<const hrt::hrtc::Bytes>(begin, begin + serialized->size());
  serialized.reset();
  config.reset();
  network.reset();
  builder.reset();
  const hrt::hrtc::AlgoParam algo{
      .name = "relu",
      .engines = {{.engine_type = "trt", .config = {}, .proto = {}, .data = bytes}},
      .config = R"({"preprocess":{"std":[255,255,255]},"postprocess":{"type":"GeneralClassify"}})"};
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  auto cpu_preprocess_algo = algo;
  cpu_preprocess_algo.config =
      R"({"preprocess":{"device":"cpu","std":[255,255,255]},"postprocess":{"type":"GeneralClassify"}})";
  hrt::SharedPtr<hrt::core::Context> cpu_preprocess_context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (cpu_preprocess_context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  auto cpu_preprocess_model = Model::create(cpu_preprocess_algo, cpu_preprocess_context);
  ASSERT_NE(cpu_preprocess_model, nullptr);
  hrt::Vector<std::uint8_t> cpu_preprocess_pixels(1024 * 1024 * 3, 64);
  const auto cpu_preprocess_image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, cpu_preprocess_pixels.data(),
                                                           hrt::core::ImageFormat::InterBgr, {1024, 1024});
  ASSERT_TRUE(cpu_preprocess_image);
  const auto cpu_preprocess_result = cpu_preprocess_model->process_sync(*cpu_preprocess_image);
  ASSERT_TRUE(cpu_preprocess_result);
  EXPECT_TRUE(std::all_of(cpu_preprocess_result->front().begin(), cpu_preprocess_result->front().end(),
                          [](float value) -> bool { return std::abs(value - 64.0F / 255.0F) <= 0.00001F; }));

  for (int device_id : {0, 1}) {
    hrt::SharedPtr<hrt::core::Context> context = hrt::core::Context::create(hrt::core::DeviceType::Cuda, device_id);
    if (context == nullptr)
      continue;
    for (const auto options : {hrt::model::ImageModelOptions{1, 2, 2}, hrt::model::ImageModelOptions{3, 2, 3}}) {
      auto model = Model::create(algo, context, options);
      hrt::Vector<std::future<bool>> runs;
      for (int worker = 0; worker < 8; ++worker) {
        runs.push_back(std::async(std::launch::async, [&model, &context, worker]() -> bool {
          context->make_current();
          auto stream = context->create_stream();
          // Distinct frame values expose scratch reuse across streams.
          hrt::Vector<std::uint8_t> pixels(1024 * 1024 * 3, static_cast<std::uint8_t>(20 + worker * 20));
          const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(),
                                                    hrt::core::ImageFormat::InterBgr, {1024, 1024});
          if (!image)
            return false;
          hrt::core::Image device_image(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::InterBgr,
                                        hrt::core::DataType::Uint8, {1024, 1024});
          if (!hrt::core::Buffer::copy(*device_image.buffer(), *image->buffer(), hrt::core::CopyPolicy::DirectCopy,
                                       stream.get()))
            return false;
          stream->synchronize();
          const auto device_view = hrt::core::Image::wrap(
              hrt::core::MemoryType::Cuda, reinterpret_cast<void*>(device_image.buffer()->memory()->device()),
              hrt::core::ImageFormat::InterBgr, {1024, 1024});
          if (!device_view)
            return false;
          hrt::core::Image host_copy(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr,
                                     hrt::core::DataType::Uint8, {1024, 1024});
          if (!hrt::core::Buffer::copy(*host_copy.buffer(), *device_image.buffer(), hrt::core::CopyPolicy::DirectCopy,
                                       stream.get()))
            return false;
          stream->synchronize();
          const auto* copied_pixels = static_cast<const std::uint8_t*>(host_copy.buffer()->memory()->cpu());
          if (copied_pixels == nullptr || !std::equal(pixels.begin(), pixels.end(), copied_pixels))
            return false;
          const auto expected = static_cast<float>(20 + worker * 20) / 255.0F;
          const auto valid_result = [expected](const auto& result) -> bool {
            if (!result || result->size() != 1 || result->front().size() != 3 * 1024 * 1024)
              return false;
            return std::all_of(result->front().begin(), result->front().end(),
                               [expected](float value) -> bool { return std::abs(value - expected) <= 0.00001F; });
          };
          for (int frame = 0; frame < 20; ++frame) {
            const auto result = model->process_sync(frame % 2 == 0 ? *device_view : *image);
            if (!valid_result(result))
              return false;
          }
          for (int memory_type = 0; memory_type < 2; ++memory_type) {
            hrt::Vector<std::uint8_t> transient_pixels(1024 * 1024 * 3, static_cast<std::uint8_t>(20 + worker * 20));
            const auto transient_host = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, transient_pixels.data(),
                                                               hrt::core::ImageFormat::InterBgr, {1024, 1024});
            if (!transient_host)
              return false;
            if (memory_type == 0) {
              if (!valid_result(model->process_sync(*transient_host)))
                return false;
              continue;
            }
            hrt::core::Image transient_device(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::InterBgr,
                                              hrt::core::DataType::Uint8, {1024, 1024});
            if (!hrt::core::Buffer::copy(*transient_device.buffer(), *transient_host->buffer(),
                                         hrt::core::CopyPolicy::DirectCopy, stream.get()))
              return false;
            stream->synchronize();
            const auto transient_view = hrt::core::Image::wrap(
                hrt::core::MemoryType::Cuda, reinterpret_cast<void*>(transient_device.buffer()->memory()->device()),
                hrt::core::ImageFormat::InterBgr, {1024, 1024});
            if (!transient_view || !valid_result(model->process_sync(*transient_view)))
              return false;
          }
          return true;
        }));
      }
      for (auto& run : runs)
        EXPECT_TRUE(run.get()) << "device=" << device_id << ", preprocess=" << options.preprocess;
    }
  }
  first_context->make_current();
}

} // namespace
#endif
