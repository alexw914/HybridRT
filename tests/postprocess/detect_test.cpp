// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/detect_test.cpp
// Description: Tests detection decoding, NMS and coordinate mapping.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/detect.hpp>

#include <cmath>
#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::FakeEngine;
using hrt::test::proto;
using Json = kitzoo::json::Json;

auto sigmoid(float x) -> float {
  return 1.0F / (1.0F + std::exp(-x));
}

// One stride-8 head over a 16x16 input: 2x2 cells, reg_max 2, one class.
// Cell (x=0, y=1) scores logit 3 with every distance ~1 cell.
auto yolov8_values(bool nhwc, float scale) -> hrt::Vector<float> {
  constexpr int kChannels = 9;
  hrt::Vector<float> values(kChannels * 4, -10.0F / scale);
  auto at = [&](int c, int y, int x) -> float& {
    return nhwc ? values[static_cast<std::size_t>((y * 2 + x) * kChannels + c)]
                : values[static_cast<std::size_t>(c * 4 + y * 2 + x)];
  };
  for (int side = 0; side < 4; ++side) {
    at(side * 2, 1, 0) = 0.0F;
    at(side * 2 + 1, 1, 0) = 20.0F / scale;
  }
  at(8, 1, 0) = 3.0F / scale;
  return values;
}

auto yolov8_config() -> Json {
  return Json::parse(R"({"class_num": 1, "strides": [8], "reg_max": 2,
                         "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
}

void expect_yolov8_box(const hrt::postprocess::DetectResultMap& result) {
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& box = result.at(0)[0];
  // Spans (-4, 4) to (12, 20) before clipping to the 16x16 source.
  EXPECT_NEAR(box.rect.x, 0.0F, 1e-3F);
  EXPECT_NEAR(box.rect.y, 4.0F, 1e-3F);
  EXPECT_NEAR(box.rect.width, 12.0F, 1e-3F);
  EXPECT_NEAR(box.rect.height, 12.0F, 1e-3F);
  EXPECT_NEAR(box.score, sigmoid(3.0F), 1e-5F);
}

TEST(DetectYoloV8Test, DecodesNchwFloat) {
  const auto output = proto(DataType::Float32, {1, 9, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("YoloV8Detect", yolov8_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, yolov8_values(false, 1.0F))},
                                     hrt::test::identity(16, 16)));
  expect_yolov8_box(result);
}

TEST(DetectYoloV8Test, DecodesNhwcInt8) {
  const auto output = proto(DataType::Int8, {1, 2, 2, 9});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output}, hrt::engine::DataLayout::Nhwc);
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("YoloV8Detect", yolov8_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  const auto real = yolov8_values(true, 0.2F);
  hrt::Vector<std::int8_t> quantized(real.begin(), real.end());
  auto buffer = hrt::test::buffer_of(output, quantized);
  buffer->scale = 0.2F;

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(16, 16)));
  expect_yolov8_box(result);
}

TEST(DetectYoloV8Test, FiltersInt8AtThreshold) {
  const auto output = proto(DataType::Int8, {1, 9, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("YoloV8Detect", yolov8_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // scale 0.2, zero point 10: q = real / 0.2 + 10. Threshold 0.5 is logit 0,
  // so cell (y=1, x=0) at q=10 is kept and cell (y=0, x=1) at q=9 is not.
  hrt::Vector<std::int8_t> quantized(9 * 4, -40);
  const auto at = [&](int c, int y, int x) -> std::int8_t& {
    return quantized[static_cast<std::size_t>(c * 4 + y * 2 + x)];
  };
  for (int side = 0; side < 4; ++side) {
    at(side * 2, 1, 0) = 10;
    at(side * 2 + 1, 1, 0) = 110;
  }
  at(8, 1, 0) = 10;
  at(8, 0, 1) = 9;
  auto buffer = hrt::test::buffer_of(output, quantized);
  buffer->scale = 0.2F;
  buffer->zero_point = 10;

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(16, 16)));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 1U);
  EXPECT_NEAR(result.at(0)[0].score, 0.5F, 1e-5F);
}

TEST(DetectYoloV8Test, AbortsOnMismatchedConfig) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, {1, 9, 2, 2})});
  auto config = yolov8_config();
  config["class_num"] = 2;
  EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("YoloV8Detect", config, engine), "");
  EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("Unknown", yolov8_config(), engine), "");
  auto negative = yolov8_config();
  negative["confidence_threshold"] = {-0.1};
  EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("YoloV8Detect", negative, engine), "");
}

TEST(DetectYoloV5Test, KeepsLowObjectnessAboveThreshold) {
  // Objectness 0.4 (logit < 0.25) with a confident class still clears 0.25.
  const auto output = proto(DataType::Float32, {1, 6, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto config = Json::parse(
      R"({"class_num": 1, "strides": [8], "anchors": [[8, 8]], "confidence_threshold": [0.25], "iou_threshold": [0.45]})");
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("YoloV5Detect", config, engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::Vector<float> values(24, -10.0F);
  for (int c = 0; c < 4; ++c)
    values[static_cast<std::size_t>(c * 4)] = 0.0F;
  values[4 * 4] = std::log(0.4F / 0.6F);
  values[5 * 4] = 10.0F;

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, values)}, hrt::test::identity(16, 16)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 0.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 8.0F);
  EXPECT_NEAR(box.score, 0.4F * sigmoid(10.0F), 1e-5F);
}

TEST(DetectYoloV5Test, TakesOneStrideAndAnchorListPerHead) {
  const auto output = proto(DataType::Float32, {1, 12, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  // Two anchors in the head: 2 * (class_num + 5) channels.
  const auto good = Json::parse(
      R"({"class_num": 1, "confidence_threshold": [0.25], "iou_threshold": [0.45], "strides": [8], "anchors": [[8, 8, 16, 16]]})");
  EXPECT_TRUE(hrt::postprocess::check_config("YoloV5Detect", good).empty());
  EXPECT_NE(hrt::postprocess::create_detect_postprocessor("YoloV5Detect", good, engine), nullptr);

  // The removed head_num, anchors_per_head and strides_per_head are unknown keys.
  const auto old = Json::parse(
      R"({"class_num": 1, "confidence_threshold": [0.25], "iou_threshold": [0.45], "head_num": 1, "anchors_per_head": [2], "strides_per_head": [8],
                                   "anchors": [[8, 8, 16, 16]]})");
  EXPECT_EQ(hrt::postprocess::check_config("YoloV5Detect", old).size(), 3U);

  for (
      const auto* bad :
      {R"({"class_num": 1, "confidence_threshold": [0.25], "iou_threshold": [0.45], "strides": [8, 16], "anchors": [[8, 8, 16, 16]]})",
       R"({"class_num": 1, "confidence_threshold": [0.25], "iou_threshold": [0.45], "strides": [8], "anchors": [[8, 8, 16]]})",
       R"({"class_num": 1, "confidence_threshold": [0.25], "iou_threshold": [0.45], "strides": [8], "anchors": [[8, 8]]})"})
    EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("YoloV5Detect", Json::parse(bad), engine), "") << bad;
}

TEST(DetectGeneralTest, AppliesNmsAndMapsToSource) {
  const auto output = proto(DataType::Float32, {1, 5, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("GeneralDetect", config, engine);
  ASSERT_NE(postprocessor, nullptr);

  // Rows cx, cy, w, h, score for two overlapping boxes: (10, 10)-(30, 30) and
  // (12, 12)-(30, 30).
  const hrt::Vector<float> values{20, 21, 20, 21, 20, 18, 20, 18, 0.9F, 0.8F};
  // Network region (0, 10, 40x20) came from source region (0, 0, 80x40).
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {0, 0, 80, 40}, {40, 40}};

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, values)}, transform));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 20.0F);
  EXPECT_FLOAT_EQ(box.rect.y, 0.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 40.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 40.0F);
  EXPECT_FLOAT_EQ(box.score, 0.9F);
}

TEST(DetectGeneralTest, DecodesCenterSizeBoxes) {
  const auto output = proto(DataType::Float32, {1, 5, 1});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("GeneralDetect", config, engine);
  ASSERT_NE(postprocessor, nullptr);

  // Center (20, 15), size 10 x 6.
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of<float>(output, {20, 15, 10, 6, 0.9F})},
                                     hrt::test::identity(40, 40)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 15.0F);
  EXPECT_FLOAT_EQ(box.rect.y, 12.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 10.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 6.0F);
}

TEST(DetectGeneralTest, DecodesInt8) {
  const auto output = proto(DataType::Int8, {1, 5, 1});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  auto postprocessor = hrt::postprocess::create_detect_postprocessor("GeneralDetect", config, engine);
  ASSERT_NE(postprocessor, nullptr);

  // real = 0.25 * (q - 4): cx, cy, w, h = 20, 20, 20, 20 and score 0.75.
  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {84, 84, 84, 84, 7});
  buffer->scale = 0.25F;
  buffer->zero_point = 4;

  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(40, 40)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 10.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 20.0F);
  EXPECT_FLOAT_EQ(box.score, 0.75F);
}

TEST(DetectGeneralTest, ChecksConfigWithoutThrowing) {
  const auto problems = [](std::string_view config) -> hrt::Vector<hrt::String> {
    return hrt::postprocess::check_config("GeneralDetect", Json::parse(config));
  };
  EXPECT_TRUE(problems(R"({"type": "GeneralDetect", "class_num": 1})").empty());
  EXPECT_EQ(problems(R"({"type": "GeneralDetect", "num": 2})").size(), 1U);
  EXPECT_EQ(problems(R"({"class_num": "one"})").size(), 1U);
  EXPECT_EQ(problems(R"({"class_num": -1})").size(), 1U);
  EXPECT_EQ(problems(R"({"confidence_treshold": [0.5]})").size(), 1U);
  EXPECT_EQ(problems(R"({"confidence_threshold": ["high"]})").size(), 1U);
  EXPECT_EQ(problems(R"({"box_format": "cxcywh"})").size(), 1U);
  EXPECT_EQ(hrt::postprocess::check_config("Mystery", Json::object()).size(), 1U);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {proto(DataType::Float32, {1, 5, 2})});
  EXPECT_DEATH(
      hrt::postprocess::create_detect_postprocessor("GeneralDetect", Json::parse(R"({"class_num": "one"})"), engine),
      "");
}

} // namespace
