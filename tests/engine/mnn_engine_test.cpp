// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/mnn_engine_test.cpp
// Description: Tests MNN CPU and optional Metal inference on an Express ReLU model.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>

#include <kitzoo/os/filesys.hpp>

#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
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

auto run_relu_model(const hrt::String& config) -> void {
  auto engine = hrt::engine::Engine::create("mnn");
  ASSERT_NE(engine, nullptr);
  hrt::hrtc::EngineParam param{.engine_type = "mnn", .config = config, .proto = {}, .data = relu_model()};
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
  EXPECT_FLOAT_EQ(result[2], 0.0F);
  EXPECT_FLOAT_EQ(result[3], 3.0F);
  EXPECT_TRUE(std::equal(values.begin(), values.end(), static_cast<const float*>(input->memory()->cpu())));

  const hrt::Array<float, 4> updated{4.0F, -1.0F, 2.0F, -3.0F};
  std::copy(updated.begin(), updated.end(), static_cast<float*>(input->memory()->mutable_cpu()));
  ASSERT_TRUE(engine->inference());
  EXPECT_FLOAT_EQ(result[0], 4.0F);
  EXPECT_FLOAT_EQ(result[1], 0.0F);
  EXPECT_FLOAT_EQ(result[2], 2.0F);
  EXPECT_FLOAT_EQ(result[3], 0.0F);

  ASSERT_TRUE(engine->initialize(param));
  ASSERT_EQ(engine->input_protos().size(), 1U);
  ASSERT_EQ(engine->output_protos().size(), 1U);
  ASSERT_TRUE(engine->set_inputs({input}));
  ASSERT_TRUE(engine->set_outputs({output}));
  ASSERT_TRUE(engine->inference());
  EXPECT_FLOAT_EQ(result[0], 4.0F);
  EXPECT_FLOAT_EQ(result[3], 0.0F);
}

TEST(MnnEngineTest, RunsReluModel) {
  run_relu_model(hrt::test::template_config("mnn"));
}

#ifdef __APPLE__
TEST(MnnEngineTest, RunsOnMetal) {
  MNN::ScheduleConfig schedule;
  schedule.type = MNN_FORWARD_METAL;
  const auto runtime = MNN::Interpreter::createRuntime({schedule});
  if (runtime.first.count(MNN_FORWARD_METAL) == 0)
    GTEST_SKIP() << "MNN Metal runtime is unavailable";

  const auto model = relu_model();
  const hrt::UniquePtr<MNN::Interpreter> interpreter{MNN::Interpreter::createFromBuffer(model->data(), model->size())};
  ASSERT_NE(interpreter, nullptr);
  auto* session = interpreter->createSession(schedule, runtime);
  ASSERT_NE(session, nullptr);
  hrt::Array<int, 2> backends{-1, -1};
  ASSERT_TRUE(interpreter->getSessionInfo(session, MNN::Interpreter::BACKENDS, backends.data()));
  ASSERT_EQ(backends[0], MNN_FORWARD_METAL);
  interpreter->releaseSession(session);

  run_relu_model(R"({"forward_type":"metal","num_thread":4})");
}
#endif

TEST(MnnEngineTest, RejectsEmptyModel) {
  auto engine = hrt::engine::Engine::create("mnn");
  ASSERT_NE(engine, nullptr);
  hrt::hrtc::EngineParam param{.engine_type = "mnn", .config = {}, .proto = {}, .data = {}};
  EXPECT_FALSE(engine->initialize(param));
  EXPECT_TRUE(engine->input_protos().empty());
  EXPECT_TRUE(engine->output_protos().empty());
}

} // namespace
