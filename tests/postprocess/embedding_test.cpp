// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/embedding_test.cpp
// Description: Tests that Embedding returns its engine output as packed floats.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/postprocessor_factory.hpp>

#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(EmbeddingTest, ReturnsTheOutputAsFloat) {
  EXPECT_EQ(hrt::model::task_of("Embedding"), hrt::model::Task::Embedding);
  const auto output = proto(DataType::Int8, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::EmbeddingResult>(
      "Embedding", kitzoo::json::Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  auto buffer = hrt::test::create_buffer<std::int8_t>(output, {0, 10, 20});
  buffer->set_quantization(0.1F, 10);
  hrt::postprocess::EmbeddingResult result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_EQ(result.size(), 3U);
  EXPECT_FLOAT_EQ(result[0], -1.0F);
  EXPECT_FLOAT_EQ(result[1], 0.0F);
  EXPECT_FLOAT_EQ(result[2], 1.0F);
}

TEST(EmbeddingTest, DequantizesStridedRowsInOrder) {
  const auto output = proto(DataType::Int8, {1, 2, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::EmbeddingResult>(
      "Embedding", kitzoo::json::Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // Rows of 3 bytes padded to 4; the padding must be skipped.
  auto buffer = std::make_shared<hrt::core::Buffer>(hrt::core::MemoryType::Cpu, DataType::Int8,
                                                    hrt::core::Shape{1, 2, 3}, hrt::core::Stride{8, 8, 4});
  ASSERT_FALSE(buffer->is_contiguous());
  const hrt::Array<std::int8_t, 8> bytes{1, 2, 3, 99, 4, 5, 6, 99};
  std::copy(bytes.begin(), bytes.end(), static_cast<std::int8_t*>(buffer->memory()->mutable_cpu()));
  buffer->set_quantization(0.5F, 0);
  hrt::postprocess::EmbeddingResult result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(8, 8)));
  EXPECT_EQ(result, (hrt::Vector<float>{0.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F}));
}

// Every int8 value plus a tail shorter than one 16-value vector step must
// match the scalar formula exactly on the vector path of the target.
TEST(EmbeddingTest, DequantizesEveryInt8ValueLikeTheScalarFormula) {
  constexpr std::int64_t kCount = 256 + 5;
  const auto output = proto(DataType::Int8, {1, kCount});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::EmbeddingResult>(
      "Embedding", kitzoo::json::Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::Vector<std::int8_t> values(kCount);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<std::int8_t>(static_cast<int>(i % 256) - 128);
  auto buffer = hrt::test::create_buffer<std::int8_t>(output, values);
  buffer->set_quantization(0.037F, -5);
  hrt::postprocess::EmbeddingResult result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_EQ(result.size(), values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(result[i], 0.037F * (static_cast<float>(values[i]) - -5.0F)) << i;
}

TEST(EmbeddingTest, NeedsOneOutput) {
  const auto output = proto(DataType::Float32, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output, output});
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::EmbeddingResult>(
                   "Embedding", kitzoo::json::Json::object(), engine),
               "");
}

TEST(EmbeddingTest, TakesNoParameters) {
  EXPECT_TRUE(hrt::postprocess::check_postprocess_config("Embedding", kitzoo::json::Json::object()).empty());
  EXPECT_FALSE(hrt::postprocess::check_postprocess_config(
                   "Embedding", kitzoo::json::Json::parse(R"({"classify_postprocess_type": "none"})"))
                   .empty());
}

} // namespace
