// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/general_keypoint_test.cpp
// Description: Tests keypoint decoding and coordinate mapping.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/keypoint.hpp>

#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(GeneralKeypointTest, MapsNormalizedPointsToSource) {
  const auto output = proto(DataType::Float32, {1, 2, 2});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "point_channel_last"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  // The 16x16 input holds a 32x32 source crop at (100, 50).
  const hrt::preprocess::ImageTransform transform{{0, 0, 16, 16}, {100, 50, 32, 32}, {16, 16}};
  hrt::postprocess::KeypointResultMap result;
  result[0].emplace_back();
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::buffer_of<float>(output, {0.5F, 0.25F, 1.0F, 1.0F})}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 1U);
  ASSERT_EQ(result.at(0)[0].keypoints.size(), 2U);
  EXPECT_FALSE(result.at(0)[0].box.rect.empty());
  EXPECT_EQ(result.at(0)[0].box.rect.x, 100);
  EXPECT_EQ(result.at(0)[0].box.rect.y, 50);
  EXPECT_EQ(result.at(0)[0].box.rect.width, 32);
  EXPECT_EQ(result.at(0)[0].box.rect.height, 32);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.x, 116.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.y, 58.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[1].point.x, 132.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[1].point.y, 82.0F);
}

TEST(GeneralKeypointTest, DequantizesInt8Points) {
  const auto output = proto(DataType::Int8, {1, 1, 2});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "point_channel_last"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  // real = (q + 100) / 200: (0.5, 0.25).
  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {0, -50});
  buffer->scale = 0.005F;
  buffer->zero_point = -100;
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(16, 16)));
  ASSERT_EQ(result.at(0).size(), 1U);
  EXPECT_EQ(result.at(0)[0].box.rect.width, 16);
  EXPECT_EQ(result.at(0)[0].box.rect.height, 16);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.x, 8.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.y, 4.0F);
}

TEST(GeneralKeypointTest, RefinesHeatmapPeak) {
  // A symmetric peak at (2, 1) stays at its cell after DARK refinement.
  const auto output = proto(DataType::Float32, {1, 1, 4, 5});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 10})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "heatmap"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::Vector<float> heatmap(20, 0.1F);
  heatmap[1 * 5 + 2] = 0.9F;
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, heatmap)}, hrt::test::identity(10, 8)));
  ASSERT_EQ(result.at(0).size(), 1U);
  EXPECT_NEAR(result.at(0)[0].keypoints[0].point.x, 4.0F, 0.5F);
  EXPECT_NEAR(result.at(0)[0].keypoints[0].point.y, 2.0F, 0.5F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].score, 0.9F);
}

TEST(GeneralKeypointTest, AbortsOnInvalidShapes) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  struct Case {
    const char* type;
    hrt::core::Shape shape;
  };

  const hrt::Vector<Case> cases{
      {"heatmap", {1, 2}},
      {"heatmap", {2, 1, 4, 4}},
      {"heatmap", {1, 1, 0, 4}},
      {"point_channel_first", {1, 3, 4}},
      {"point_channel_last", {1, 4, 3}},
      {"point_channel_last", {2, 4, 2}},
  };
  for (const auto& item : cases) {
    hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {proto(DataType::Float32, item.shape)});
    const kitzoo::json::Json config{{"general_keypoint_type", item.type}};
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine), "");
  }
  hrt::test::FakeEngine empty({proto(DataType::Float32, {1, 3, 8, 8})}, {});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", kitzoo::json::Json::object(), empty),
               "");
}

TEST(GeneralKeypointTest, AbortsOnInvalidVisibilityOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto config = kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_with_visible"})");
  const auto x = proto(DataType::Float32, {1, 2, 4});
  for (const auto& visibility : hrt::Vector<hrt::core::Shape>{{1, 2, 2}, {1, 2, 4}, {1, 1, 3}, {2, 2, 3}}) {
    hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})},
                                 {x, x, proto(DataType::Float32, visibility)});
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine), "");
  }
  hrt::test::FakeEngine missing({proto(DataType::Float32, {1, 3, 8, 8})}, {x, x});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, missing), "");
}

TEST(GeneralKeypointTest, DecodesChannelFirstPoints) {
  const auto output = proto(DataType::Float32, {1, 2, 1});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_channel_first"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::buffer_of<float>(output, {0.25F, 0.5F})}, hrt::test::identity(8, 8)));
  ASSERT_EQ(result.at(0)[0].keypoints.size(), 1U);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.x, 2.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.y, 4.0F);
}

TEST(GeneralKeypointTest, DecodesVisibilityDistributions) {
  const auto x = proto(DataType::Float32, {1, 1, 4});
  const auto y = proto(DataType::Float32, {1, 1, 2});
  const auto visible = proto(DataType::Float32, {1, 1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {x, y, visible});
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_with_visible"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(
      postprocessor->process(result,
                             {hrt::test::buffer_of<float>(x, {0, 1, 0, 0}), hrt::test::buffer_of<float>(y, {0, 1}),
                              hrt::test::buffer_of<float>(visible, {0.1F, 0.2F, 0.7F})},
                             hrt::test::identity(8, 8)));
  ASSERT_EQ(result.at(0)[0].keypoints.size(), 1U);
  const auto& point = result.at(0)[0].keypoints[0];
  EXPECT_FLOAT_EQ(point.point.x, 2.0F);
  EXPECT_FLOAT_EQ(point.point.y, 4.0F);
  EXPECT_EQ(point.visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(point.score, 0.7F);
}

} // namespace
