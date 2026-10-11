// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/mnn_engine_test.cpp
// Description: Tests the MNN backend on a ReLU model built with MNN Express.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>

#include <kitzoo/os/filesys.hpp>

#include <MNN/expr/ExprCreator.hpp>
#include <filesystem>
#include <gtest/gtest.h>

#include "template_config.hpp"

namespace {

// y = Relu(x), x float [1, 4].
auto relu_model() -> hrt::SharedPtr<const hrt::hrtc::Bytes> {
  using namespace MNN::Express;
  auto x = _Input({1, 4}, NCHW, halide_type_of<float>());
  x->setName("x");
  auto y = _Relu(x);
  y->setName("y");
  const auto path = std::filesystem::temp_directory_path() / "hybridrt_mnn_relu.mnn";
  Variable::save({y}, path.c_str());
  const auto content = kitzoo::os::read_file(path);
  std::filesystem::remove(path);
  return std::make_shared<const hrt::hrtc::Bytes>(content.begin(), content.end());
}

TEST(MnnEngineTest, RunsReluModel) {
  auto engine = hrt::engine::Engine::create("mnn");
  ASSERT_NE(engine, nullptr);
  hrt::hrtc::EngineParam param{
      .engine_type = "mnn", .config = hrt::test::template_config("mnn"), .proto = {}, .data = relu_model()};
  ASSERT_TRUE(engine->initialize(param));
  ASSERT_EQ(engine->input_names(), hrt::Vector<hrt::String>{"x"});
  ASSERT_EQ(engine->output_protos().size(), 1U);
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
  EXPECT_FLOAT_EQ(result[3], 3.0F);
}

} // namespace
