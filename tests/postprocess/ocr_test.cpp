// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/ocr_test.cpp
// Description: Tests CTC sequence decoding, DB regions and config validation.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

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
  auto decoder = hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>("Ocr", config, engine);
  hrt::Vector<float> scores(28, 0.01F);
  const hrt::Array<int, 7> indexes{1, 1, 0, 1, 3, 2, 0};
  for (std::size_t t = 0; t < indexes.size(); ++t)
    scores[t * 4 + static_cast<std::size_t>(indexes[t])] = 0.9F;
  auto buffer = hrt::test::create_buffer<float>(output, scores);
  hrt::postprocess::OcrResult result;
  ASSERT_TRUE(decoder->process(result, {buffer}, hrt::test::identity(320, 48)));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].text, "中中 a");
  EXPECT_NEAR(result[0].recognition_score, 0.9F, 1e-6F);
  std::fill(scores.begin(), scores.end(), 0.0F);
  ASSERT_TRUE(
      decoder->process(result, {hrt::test::create_buffer<float>(output, scores)}, hrt::test::identity(320, 48)));
  EXPECT_TRUE(result[0].text.empty());
  scores[5] = 3.5F; // A logit.
  EXPECT_FALSE(
      decoder->process(result, {hrt::test::create_buffer<float>(output, scores)}, hrt::test::identity(320, 48)));
  scores[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(
      decoder->process(result, {hrt::test::create_buffer<float>(output, scores)}, hrt::test::identity(320, 48)));
}

TEST(OcrTest, ExpandsScoresAndMapsDbRegions) {
  const auto output = proto(DataType::Float32, {1, 1, 32, 64});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 64})}, {output});
  auto decoder =
      hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>("Ocr", {{"stage", "detection"}}, engine);
  hrt::Vector<float> scores(32 * 64, 0);
  for (int y = 10; y <= 20; ++y)
    for (int x = 10; x <= 40; ++x)
      scores[static_cast<std::size_t>(y * 64 + x)] = 0.9F;
  hrt::postprocess::OcrResult result;
  const hrt::preprocess::ImageTransform transform{{0, 0, 64, 32}, {20, 30, 128, 64}, {64, 32}};
  ASSERT_TRUE(decoder->process(result, {hrt::test::create_buffer<float>(output, scores)}, transform));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_NEAR(result[0].detection_score, 0.9F, 1e-6F);
  const auto& rect = result[0].rect;
  EXPECT_LT(rect.x, 40);
  EXPECT_GT(rect.x + rect.width, 100);
  EXPECT_GE(rect.x, 20);
  EXPECT_GE(rect.y, 30);
  EXPECT_LE(rect.x + rect.width, 147);
  EXPECT_LE(rect.y + rect.height, 93);
  std::fill(scores.begin(), scores.end(), 0.0F);
  ASSERT_TRUE(decoder->process(result, {hrt::test::create_buffer<float>(output, scores)}, transform));
  EXPECT_TRUE(result.empty());
}

TEST(OcrTest, DecodesInt8MapsInPlace) {
  // Probabilities q / 255 + 128 / 255, as an RKNN DB map: 0.9 is stored as 102.
  const auto map = proto(DataType::Int8, {1, 1, 32, 64});
  hrt::test::FakeEngine detection_engine({proto(DataType::Float32, {1, 3, 32, 64})}, {map});
  auto detector = hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>("Ocr", {{"stage", "detection"}},
                                                                                      detection_engine);
  hrt::Vector<std::int8_t> scores(32 * 64, -128);
  for (int y = 10; y <= 20; ++y)
    for (int x = 10; x <= 40; ++x)
      scores[static_cast<std::size_t>(y * 64 + x)] = 102;
  auto buffer = hrt::test::create_buffer(map, scores);
  buffer->set_quantization(1.0F / 255.0F, -128);
  hrt::postprocess::OcrResult result;
  const hrt::preprocess::ImageTransform transform{{0, 0, 64, 32}, {20, 30, 128, 64}, {64, 32}};
  ASSERT_TRUE(detector->process(result, {buffer}, transform));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_NEAR(result[0].detection_score, 230.0F / 255.0F, 1e-5F);
  // The threshold 0.2 lies between 51 / 255 (stored -77, kept out) and 52 / 255.
  std::fill(scores.begin(), scores.end(), std::int8_t{-77});
  buffer = hrt::test::create_buffer(map, scores);
  buffer->set_quantization(1.0F / 255.0F, -128);
  ASSERT_TRUE(detector->process(result, {buffer}, transform));
  EXPECT_TRUE(result.empty());
  // A value below zero is rejected like a float one: -77 dequantizes to -27 / 255.
  buffer->set_quantization(1.0F / 255.0F, -50);
  EXPECT_FALSE(detector->process(result, {buffer}, transform));

  const auto logits = proto(DataType::Int8, {1, 3, 3});
  hrt::test::FakeEngine recognition_engine({proto(DataType::Float32, {1, 3, 48, 320})}, {logits});
  auto recognizer = hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>(
      "Ocr", {{"stage", "recognition"}, {"dictionary", "string:a\n"}}, recognition_engine);
  buffer = hrt::test::create_buffer<std::int8_t>(logits, {-128, 102, -128, -128, 102, -128, -128, -128, 77});
  buffer->set_quantization(1.0F / 255.0F, -128);
  ASSERT_TRUE(recognizer->process(result, {buffer}, hrt::test::identity(320, 48)));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].text, "a ");
  EXPECT_NEAR(result[0].recognition_score, (230.0F + 205.0F) / 2.0F / 255.0F, 1e-5F);
}

TEST(OcrTest, RejectsInvalidConfigAndDictionaryMismatch) {
  const auto output = proto(DataType::Float32, {1, 2, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 48, 320})}, {output});
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>(
                   "Ocr", {{"stage", "detection"}, {"unknown", true}}, engine),
               "");
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>(
                   "Ocr", {{"stage", "recognition"}, {"dictionary", "string:a\n"}}, engine),
               "");
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>(
                   "Ocr", {{"stage", "detection"}, {"threshold", 2}}, engine),
               "");
}

} // namespace
