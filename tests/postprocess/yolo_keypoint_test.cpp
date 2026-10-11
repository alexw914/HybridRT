// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/yolo_keypoint_test.cpp
// Description: Tests decoding of single-output pose heads.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/keypoint.hpp>

#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::postprocess::KeypointVisible;
using hrt::test::proto;
using Json = kitzoo::json::Json;

auto config() -> Json {
  return Json::parse(R"({"class_num": 1, "keypoint_num": 2, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
}

TEST(YoloKeypointTest, KeepsTheBestBoxWithItsKeypoints) {
  // [1, 4 + 1 class + 2 keypoints x 3, 2 candidates], one row per value.
  const auto output = proto(DataType::Float32, {1, 11, 2});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor("YoloKeypoint", config(), engine);

  // Candidate 0: box (10, 10)-(30, 30), score 0.9, keypoints (12, 14) visible
  // and (28, 26) hidden. Candidate 1 overlaps it with a lower score.
  const hrt::Vector<float> values{20, 21, 20, 21,   20, 18, 20, 18, 0.9F, 0.8F, 12,
                                  0,  14, 0,  0.9F, 0,  28, 0,  26, 0,    0.2F, 0};
  // Network region (0, 10, 40x20) came from source region (0, 0, 80x40).
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {0, 0, 80, 40}, {40, 40}};
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, values)}, transform));

  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& object = result.at(0)[0];
  EXPECT_FLOAT_EQ(object.box.score, 0.9F);
  EXPECT_FLOAT_EQ(object.box.rect.x, 20.0F);
  EXPECT_FLOAT_EQ(object.box.rect.y, 0.0F);
  EXPECT_FLOAT_EQ(object.box.rect.width, 40.0F);
  ASSERT_EQ(object.keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.x, 24.0F);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.y, 8.0F);
  EXPECT_EQ(object.keypoints[0].visible, KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(object.keypoints[0].score, 0.9F);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.x, 56.0F);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.y, 32.0F);
  EXPECT_EQ(object.keypoints[1].visible, KeypointVisible::Invisible);
}

TEST(YoloKeypointTest, DecodesKeypointsWithoutVisibility) {
  // [1, 4 + 1 class + 2 keypoints x 2, 1 candidate].
  const auto output = proto(DataType::Float32, {1, 9, 1});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor("YoloKeypoint", config(), engine);

  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result,
                                     {hrt::test::buffer_of<float>(output, {20, 20, 20, 20, 0.9F, 12, 14, 28, 26})},
                                     hrt::test::identity(40, 40)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& keypoints = result.at(0)[0].keypoints;
  ASSERT_EQ(keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(keypoints[1].point.x, 28.0F);
  EXPECT_FLOAT_EQ(keypoints[1].point.y, 26.0F);
  EXPECT_EQ(keypoints[1].visible, KeypointVisible::Visible);
}

TEST(YoloKeypointTest, AbortsOnOutputsThatDoNotFit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {proto(DataType::Float32, {1, 10, 4})});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloKeypoint", config(), engine), "");
  EXPECT_EQ(hrt::postprocess::check_config("YoloKeypoint", Json::parse(R"({"keypoints": 17})")).size(), 1U);
}

} // namespace
