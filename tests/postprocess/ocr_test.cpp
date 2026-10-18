// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/ocr_test.cpp
// Description: Tests CTC sequence decoding, DB regions and config validation.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/ocr.hpp>

#include <gtest/gtest.h>
#include <limits>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(OcrTest, DecodesUtf8DuplicatesBlanksAndSpace) {
  const auto output = proto(DataType::Float32, {1, 7, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 48, 320})}, {output});
  const kitzoo::json::Json config{{"stage", "recognition"}, {"dictionary", "string:中\na\n"}};
  auto decoder = hrt::postprocess::create_ocr_postprocessor(config, engine);
  hrt::Vector<float> scores(28, 0.01F);
  const hrt::Array<int, 7> indexes{1, 1, 0, 1, 3, 2, 0};
  for (std::size_t t = 0; t < indexes.size(); ++t)
    scores[t * 4 + static_cast<std::size_t>(indexes[t])] = 0.9F;
  auto buffer = hrt::test::buffer_of<float>(output, scores);
  hrt::postprocess::OcrResult result;
  ASSERT_TRUE(decoder->process(result, {buffer}, hrt::test::identity(320, 48)));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].text, "中中 a");
  EXPECT_NEAR(result[0].recognition_score, 0.9F, 1e-6F);
  std::fill(scores.begin(), scores.end(), 0.0F);
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of<float>(output, scores)}, hrt::test::identity(320, 48)));
  EXPECT_TRUE(result[0].text.empty());
  scores[5] = 3.5F; // A logit.
  EXPECT_FALSE(decoder->process(result, {hrt::test::buffer_of<float>(output, scores)}, hrt::test::identity(320, 48)));
  scores[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(decoder->process(result, {hrt::test::buffer_of<float>(output, scores)}, hrt::test::identity(320, 48)));
}

TEST(OcrTest, ExpandsScoresAndMapsDbRegions) {
  const auto output = proto(DataType::Float32, {1, 1, 32, 64});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 64})}, {output});
  auto decoder = hrt::postprocess::create_ocr_postprocessor({{"stage", "detection"}}, engine);
  hrt::Vector<float> scores(32 * 64, 0);
  for (int y = 10; y <= 20; ++y)
    for (int x = 10; x <= 40; ++x)
      scores[static_cast<std::size_t>(y * 64 + x)] = 0.9F;
  hrt::postprocess::OcrResult result;
  const hrt::preprocess::ImageTransform transform{{0, 0, 64, 32}, {20, 30, 128, 64}, {64, 32}};
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of<float>(output, scores)}, transform));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_NEAR(result[0].detection_score, 0.9F, 1e-6F);
  const auto& rect = result[0].rect;
  EXPECT_LT(rect.x, 40);
  EXPECT_GT(rect.x + rect.width, 100);
  EXPECT_GE(rect.x, 20);
  EXPECT_GE(rect.y, 30);
  EXPECT_LE(rect.x + rect.width, 147);
  EXPECT_LE(rect.y + rect.height, 93);
  std::fill(scores.begin(), scores.end(), 0);
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of<float>(output, scores)}, transform));
  EXPECT_TRUE(result.empty());
}

TEST(OcrTest, RejectsInvalidConfigAndDictionaryMismatch) {
  const auto output = proto(DataType::Float32, {1, 2, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 48, 320})}, {output});
  EXPECT_DEATH(hrt::postprocess::create_ocr_postprocessor({{"stage", "detection"}, {"unknown", true}}, engine), "");
  EXPECT_DEATH(
      hrt::postprocess::create_ocr_postprocessor({{"stage", "recognition"}, {"dictionary", "string:a\n"}}, engine), "");
  EXPECT_DEATH(hrt::postprocess::create_ocr_postprocessor({{"stage", "detection"}, {"threshold", 2}}, engine), "");
}

} // namespace
