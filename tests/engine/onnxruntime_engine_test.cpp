// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/onnxruntime_engine_test.cpp
// Description: Tests the ONNX Runtime backend on a generated Relu model.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>

#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string_view>

#include "onnx_model.hpp"
#include "relu.hpp"
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
  ASSERT_NO_FATAL_FAILURE(hrt::test::expect_relu(*engine, *input, *output, values));
}

TEST(OnnxruntimeEngineTest, RunsReluModel) {
  expect_relu(hrt::test::template_config("ort"));
}

#ifdef __APPLE__
TEST(OnnxruntimeEngineTest, RunsOnCoreml) {
  expect_relu(R"({"execution_provider": "coreml", "provider_options": {"ModelFormat": "MLProgram"}})");
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
