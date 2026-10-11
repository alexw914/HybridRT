// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/ncnn_engine_test.cpp
// Description: Tests the ncnn backend on an inline ReLU model.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>

#include <gtest/gtest.h>
#include <string_view>

#include "template_config.hpp"

namespace {

// data [c=1, h=1, w=4] -> ReLU -> output; blob shapes come from -23330 hints.
constexpr std::string_view kReluParam = "7767517\n"
                                        "2 2\n"
                                        "Input data 0 1 data 0=4 1=1 2=1 -23330=4,3,4,1,1\n"
                                        "ReLU relu 1 1 data output -23330=4,3,4,1,1\n";

TEST(NcnnEngineTest, RunsReluModel) {
  auto engine = hrt::engine::Engine::create("ncnn");
  ASSERT_NE(engine, nullptr);
  hrt::hrtc::EngineParam param{.engine_type = "ncnn",
                               .config = hrt::test::template_config("ncnn"),
                               .proto = std::make_shared<const hrt::hrtc::Bytes>(kReluParam.begin(), kReluParam.end()),
                               .data = {}};
  ASSERT_TRUE(engine->initialize(param));
  ASSERT_EQ(engine->input_protos().size(), 1U);
  EXPECT_EQ(engine->input_protos()[0].shape, (hrt::core::Shape{1, 1, 1, 4}));
  ASSERT_EQ(engine->output_protos().size(), 1U);
  EXPECT_EQ(engine->output_protos()[0].shape, (hrt::core::Shape{1, 1, 1, 4}));

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
  EXPECT_FLOAT_EQ(result[3], 3.0F);
}

TEST(NcnnEngineTest, RejectsBrokenParam) {
  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::Ncnn);
  ASSERT_NE(engine, nullptr);
  constexpr std::string_view kBroken = "not a param file";
  hrt::hrtc::EngineParam param{.engine_type = "ncnn",
                               .config = {},
                               .proto = std::make_shared<const hrt::hrtc::Bytes>(kBroken.begin(), kBroken.end()),
                               .data = {}};
  EXPECT_FALSE(engine->initialize(param));
}

} // namespace
