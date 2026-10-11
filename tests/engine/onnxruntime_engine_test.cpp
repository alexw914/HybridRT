// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/onnxruntime_engine_test.cpp
// Description: Tests the ONNX Runtime backend on a generated Relu model.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/hrtc/hrtc.hpp>

#include <gtest/gtest.h>
#include <string_view>

#include "onnx_model.hpp"
#include "template_config.hpp"

namespace {

auto relu_model() -> hrt::SharedPtr<const hrt::hrtc::Bytes> {
  return std::make_shared<const hrt::hrtc::Bytes>(hrt::test::unary_model("Relu", {1, 4}));
}

auto relu_param(std::string_view config = {}) -> hrt::hrtc::EngineParam {
  return {.engine_type = "ort", .config = hrt::String(config), .proto = {}, .data = relu_model()};
}

// Runs Relu on {-2, -0.5, 0, 3} with `config`.
void expect_relu(std::string_view config) {
  auto engine = hrt::engine::Engine::create("ORT");
  ASSERT_NE(engine, nullptr);
  ASSERT_TRUE(engine->initialize(relu_param(config)));
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
#endif

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
