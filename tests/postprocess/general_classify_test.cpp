// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/general_classify_test.cpp
// Description: Tests classification score activation.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/classify.hpp>

#include <cmath>
#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(GeneralClassifyTest, AppliesSoftmax) {
  const auto output = proto(DataType::Float32, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_classify_postprocessor(
      "GeneralClassify", kitzoo::json::Json::parse(R"({"classify_postprocess_type": "softmax"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::postprocess::ClassifyResult result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::create_buffer<float>(output, {0.0F, std::log(3.0F), 0.0F})},
                                     hrt::test::identity(8, 8)));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_NEAR(result[0][0], 0.2F, 1e-6F);
  EXPECT_NEAR(result[0][1], 0.6F, 1e-6F);
}

TEST(GeneralClassifyTest, DequantizesInt8Scores) {
  const auto output = proto(DataType::Int8, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor =
      hrt::postprocess::create_classify_postprocessor("GeneralClassify", kitzoo::json::Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  auto buffer = hrt::test::create_buffer<std::int8_t>(output, {0, 10, 20});
  buffer->set_quantization(0.1F, 10);
  hrt::postprocess::ClassifyResult result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_EQ(result.at(0).size(), 3U);
  EXPECT_FLOAT_EQ(result[0][0], -1.0F);
  EXPECT_FLOAT_EQ(result[0][1], 0.0F);
  EXPECT_FLOAT_EQ(result[0][2], 1.0F);
}

TEST(GeneralClassifyTest, DecodesEachSampleOfABatch) {
  const auto output = proto(DataType::Float32, {2, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {2, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_classify_postprocessor(
      "GeneralClassify", kitzoo::json::Json::parse(R"({"classify_postprocess_type": "softmax"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  const auto buffer = hrt::test::create_buffer<float>(output, {0.0F, std::log(3.0F), 0.0F, std::log(2.0F), 0.0F, 0.0F});
  hrt::Vector<hrt::postprocess::ClassifyResult> results;
  ASSERT_TRUE(postprocessor->process(results, {buffer}, {hrt::test::identity(8, 8), hrt::test::identity(8, 8)}));
  ASSERT_EQ(results.size(), 2U);
  ASSERT_EQ(results[0].at(0).size(), 3U);
  EXPECT_NEAR(results[0][0][1], 0.6F, 1e-6F);
  ASSERT_EQ(results[1].at(0).size(), 3U);
  EXPECT_NEAR(results[1][0][0], 0.5F, 1e-6F);

  // Fewer transforms than samples decode only the leading samples.
  ASSERT_TRUE(postprocessor->process(results, {buffer}, {hrt::test::identity(8, 8)}));
  ASSERT_EQ(results.size(), 1U);
  EXPECT_NEAR(results[0][0][1], 0.6F, 1e-6F);
}

} // namespace
