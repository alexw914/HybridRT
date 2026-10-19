// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/embedding_test.cpp
// Description: Tests that Embedding returns its engine output unchanged.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(EmbeddingTest, ReturnsTheOutputAsFloat) {
  EXPECT_EQ(hrt::model::task_of("Embedding"), hrt::model::Task::Embedding);
  const auto output = proto(DataType::Int8, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor =
      hrt::postprocess::create_embedding_postprocessor("Embedding", kitzoo::json::Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {0, 10, 20});
  buffer->scale = 0.1F;
  buffer->zero_point = 10;
  hrt::postprocess::EmbeddingResult result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_EQ(result.size(), 3U);
  EXPECT_FLOAT_EQ(result[0], -1.0F);
  EXPECT_FLOAT_EQ(result[1], 0.0F);
  EXPECT_FLOAT_EQ(result[2], 1.0F);
}

TEST(EmbeddingTest, NeedsOneOutput) {
  const auto output = proto(DataType::Float32, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output, output});
  EXPECT_DEATH(hrt::postprocess::create_embedding_postprocessor("Embedding", kitzoo::json::Json::object(), engine), "");
}

TEST(EmbeddingTest, TakesNoParameters) {
  EXPECT_TRUE(hrt::postprocess::check_config("Embedding", kitzoo::json::Json::object()).empty());
  EXPECT_FALSE(
      hrt::postprocess::check_config("Embedding", kitzoo::json::Json::parse(R"({"classify_postprocess_type": "none"})"))
          .empty());
}

} // namespace
