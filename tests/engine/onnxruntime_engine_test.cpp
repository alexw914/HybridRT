// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/onnxruntime_engine_test.cpp
// Description: Tests ONNX Runtime execution, configuration and NHWC image inputs.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string_view>

#ifdef HRT_ORT_MODEL_TEST
#include <opencv2/imgcodecs.hpp>
#endif

#include "onnx_model.hpp"
#include "template_config.hpp"

namespace {

auto relu_param(std::string_view config = {}, std::initializer_list<std::int64_t> dims = {1, 4})
    -> hrt::hrtc::EngineParam {
  return {.engine_type = "ort",
          .config = hrt::String(config),
          .proto = {},
          .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::test::unary_model("Relu", dims))};
}

// Runs Relu on {-2, -0.5, 0, 3} with `config`, on a model whose dims fix to {1, 4}.
void expect_relu(std::string_view config, std::initializer_list<std::int64_t> dims = {1, 4}) {
  auto engine = hrt::engine::Engine::create("ORT");
  ASSERT_NE(engine, nullptr);
  ASSERT_TRUE(engine->initialize(relu_param(config, dims)));
  EXPECT_EQ(engine->input_protos()[0].shape, (hrt::core::Shape{1, 4}));
  ASSERT_EQ(engine->input_names(), hrt::Vector<hrt::String>{"x"});
  ASSERT_EQ(engine->output_protos().size(), 1U);
  EXPECT_EQ(engine->output_protos()[0].data_type, hrt::core::DataType::Float32);
  EXPECT_EQ(engine->output_protos()[0].shape, (hrt::core::Shape{1, 4}));

  auto input = std::make_shared<hrt::core::Buffer>(engine->input_protos()[0]);
  auto output = std::make_shared<hrt::core::Buffer>(engine->output_protos()[0]);
  ASSERT_TRUE(engine->set_inputs({input}));
  ASSERT_TRUE(engine->set_outputs({output}));

  const hrt::Array<float, 4> values{-2.0F, -0.5F, 0.0F, 3.0F};
  std::copy(values.begin(), values.end(), static_cast<float*>(input->memory()->mutable_cpu()));
  ASSERT_TRUE(engine->inference());
  const auto* result = static_cast<const float*>(output->memory()->cpu());
  EXPECT_FLOAT_EQ(result[0], 0.0F);
  EXPECT_FLOAT_EQ(result[1], 0.0F);
  EXPECT_FLOAT_EQ(result[2], 0.0F);
  EXPECT_FLOAT_EQ(result[3], 3.0F);
}

TEST(OnnxruntimeEngineTest, RunsReluModel) {
  expect_relu(hrt::test::template_config("ort"));
}

#ifdef __APPLE__
TEST(OnnxruntimeEngineTest, RunsOnCoreml) {
  expect_relu(R"({"execution_provider": "coreml", "provider_options": {"ModelFormat": "MLProgram"}})");
}

// The macOS package has no CUDA provider, and CoreML rejects the bad option.
TEST(OnnxruntimeEngineTest, FallsBackToCpuWithoutProvider) {
  expect_relu(R"({"execution_provider": "cuda", "provider_options": {"device_id": "0"}})");
  expect_relu(R"({"execution_provider": "coreml", "provider_options": {"ModelFormat": "Bad"}})");
}
#endif

#ifdef HRT_ORT_MODEL_TEST
TEST(OnnxruntimeEngineTest, RunsNhwcClassifierWithMetalPreprocessing) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto algo = hrt::test::template_algo("yolov8n_classify");
  if (!algo.is_object())
    GTEST_SKIP() << "Classifier template unavailable";
  auto path = hrt::test::template_model(algo);
  path.replace_filename(path.stem().string() + "Nhwc.onnx");
  const auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(path) || bgr.empty())
    GTEST_SKIP() << "NHWC classifier or bus.jpg unavailable";
  auto metal = ImagePreprocessor::create(ImagePreprocessorType::Metal);
  if (!metal)
    GTEST_SKIP() << "Metal GPU unavailable";
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  algo["config"]["engine"]["backends"][0]["data"] = path.string();
  const auto source =
      Image::wrap(MemoryType::Cpu, bgr.data, ImageFormat::InterBgr, {bgr.cols, bgr.rows}, DataType::Uint8, bgr.step);
  ASSERT_TRUE(source);
  auto param = algo["config"]["preprocess"].get<ImagePreprocessorParam>();
  param.dst_image_format = ImageFormat::InterRgb;
  param.interpolation_mode = InterpolationMode::Nearest;
  for (const auto provider : {"cpu", "coreml"}) {
    SCOPED_TRACE(provider);
    algo["config"]["engine"]["backends"][0]["config"]["execution_provider"] = provider;
    const auto loaded = hrt::hrtc::load_algo(algo);
    ASSERT_TRUE(loaded);
    auto engine = hrt::engine::Engine::create("ort");
    ASSERT_TRUE(engine->initialize(loaded->engines.front()));
    ASSERT_EQ(engine->input_protos().front().shape, (Shape{1, 640, 640, 3}));
    ASSERT_EQ(engine->output_protos().front().shape, (Shape{1, 1000}));
    auto input = std::make_shared<Buffer>(engine->input_protos().front());
    auto output = std::make_shared<Buffer>(engine->output_protos().front());
    Image target(input->memory()->ref(), ImageFormat::InterRgb, DataType::Float32, {640, 640});
    ASSERT_TRUE(target.valid());
    ASSERT_TRUE(metal->supports(ImageOperation::Fused, target, *source));
    ASSERT_TRUE(engine->set_inputs({input}));
    ASSERT_TRUE(engine->set_outputs({output}));
    ASSERT_TRUE(cpu->process(target, *source, {}, param));
    ASSERT_TRUE(engine->inference());
    const auto* values = static_cast<const float*>(output->memory()->cpu());
    const hrt::Vector<float> expected(values, values + output->volume());
    ASSERT_TRUE(metal->process(target, *source, {}, param));
    ASSERT_TRUE(engine->inference());
    values = static_cast<const float*>(output->memory()->cpu());
    float max_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      ASSERT_TRUE(std::isfinite(values[i]));
      max_error = std::max(max_error, std::abs(values[i] - expected[i]));
      EXPECT_NEAR(values[i], expected[i], 1.0e-5F);
    }
    const auto expected_class = std::max_element(expected.begin(), expected.end()) - expected.begin();
    const auto actual_class = std::max_element(values, values + expected.size()) - values;
    EXPECT_EQ(actual_class, expected_class);
    KZ_LOG_INFO("NHWC classifier {}: top-1 {}, max CPU/Metal error {}", provider, actual_class, max_error);
  }
}
#endif

TEST(OnnxruntimeEngineTest, FixesDynamicShapesAtLoad) {
  expect_relu({}, {-1, 4});
  expect_relu(R"({"input_shapes": {"x": [1, 4]}})", {-1, -1});

  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::Ort);
  EXPECT_FALSE(engine->initialize(relu_param({}, {-1, -1})));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"input_shapes": {"y": [1, 4]}})", {-1, -1})), "");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"input_shapes": {"x": [1, 4, 1]}})", {-1, -1})), "");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"input_shapes": {"x": [1, 5]}})")), "");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"input_shapes": {"x": [1, 0]}})")), "");
}

TEST(OnnxruntimeEngineTest, ChecksConfig) {
  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::Ort);
  // Operator libraries are optional: a missing one is skipped.
  EXPECT_TRUE(engine->initialize(relu_param(R"({"custom_op_libraries": ["/nonexistent/libops.so"]})")));

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"execution_provider": "gpu"})")), "");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"provider_options": {"ModelFormat": "MLProgram"}})")), "");
  EXPECT_DEATH(engine->initialize(relu_param(R"({"intra_op_threads": 2})")), "");
  EXPECT_DEATH(engine->initialize(relu_param("[1]")), "");
  EXPECT_DEATH(engine->initialize(relu_param("{")), "");
}

TEST(OnnxruntimeEngineTest, RejectsBadModelsAndBuffers) {
  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::Ort);
  ASSERT_NE(engine, nullptr);
  hrt::hrtc::EngineParam bad{.engine_type = "ort",
                             .config = {},
                             .proto = {},
                             .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::hrtc::Bytes{1, 2, 3})};
  EXPECT_FALSE(engine->initialize(bad));

  ASSERT_TRUE(engine->initialize(relu_param()));
  auto wrong = std::make_shared<hrt::core::Buffer>(hrt::core::MemoryType::Cpu, hrt::core::DataType::Float32,
                                                   hrt::core::Shape{1, 3});
  EXPECT_FALSE(engine->set_inputs({wrong}));
}

} // namespace
