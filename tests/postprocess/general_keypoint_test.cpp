// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/general_keypoint_test.cpp
// Description: Tests heatmap, SimCC and coordinate keypoint decoding and
//              coordinate mapping.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/keypoint.hpp>

#include <cmath>
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

// A sigma-2 Gaussian centered at (cx, cy) on a width x height map.
auto gaussian_map(int width, int height, float cx, float cy, float peak) -> hrt::Vector<float> {
  hrt::Vector<float> map(static_cast<std::size_t>(width * height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto dx = static_cast<float>(x) - cx;
      const auto dy = static_cast<float>(y) - cy;
      map[static_cast<std::size_t>(y * width + x)] = peak * std::exp(-(dx * dx + dy * dy) / 8.0F);
    }
  }
  return map;
}

TEST(GeneralKeypointTest, RefinesHeatmapToSubpixelPeak) {
  // 48 x 64 heatmaps of a 192 x 256 input: 4 input pixels per cell.
  const auto output = proto(DataType::Float32, {1, 2, 64, 48});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 256, 192})}, {output});
  auto heatmap = gaussian_map(48, 64, 20.3F, 30.7F, 0.9F);
  // Near the border DARK has no room for its stencil and keeps the argmax.
  const auto edge = gaussian_map(48, 64, 1.3F, 30.0F, 0.9F);
  heatmap.insert(heatmap.end(), edge.begin(), edge.end());
  for (const auto* refine : {"dark", "none"}) {
    const kitzoo::json::Json config{{"general_keypoint_type", "heatmap"}, {"heatmap_refine", refine}};
    auto postprocessor = hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine);
    hrt::postprocess::KeypointResultMap result;
    ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, heatmap)}, hrt::test::identity(192, 256)));
    const auto& points = result.at(0)[0].keypoints;
    ASSERT_EQ(points.size(), 2U);
    if (refine == hrt::String("dark")) {
      EXPECT_NEAR(points[0].point.x, 20.3F * 4.0F, 0.1F);
      EXPECT_NEAR(points[0].point.y, 30.7F * 4.0F, 0.1F);
    } else {
      EXPECT_FLOAT_EQ(points[0].point.x, 80.0F);
      EXPECT_FLOAT_EQ(points[0].point.y, 124.0F);
    }
    EXPECT_FLOAT_EQ(points[1].point.x, 4.0F);
    EXPECT_FLOAT_EQ(points[1].point.y, 120.0F);
  }
}

TEST(GeneralKeypointTest, KeepsArgmaxAndHidesLogitHeatmapPeaks) {
  // Logit heatmaps are mostly negative: the blurred map has no positive log
  // to expand, so the peak stays, and a peak at or below 0 is not visible.
  const auto output = proto(DataType::Float32, {1, 2, 16, 16});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 64, 64})}, {output});
  hrt::Vector<float> heatmap(512, -8.0F);
  heatmap[7 * 16 + 9] = 1.5F;
  heatmap[256 + 5 * 16 + 4] = -0.5F;
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "heatmap"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, heatmap)}, hrt::test::identity(64, 64)));
  const auto& points = result.at(0)[0].keypoints;
  EXPECT_FLOAT_EQ(points[0].point.x, 36.0F);
  EXPECT_FLOAT_EQ(points[0].point.y, 28.0F);
  EXPECT_EQ(points[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(points[1].point.x, 16.0F);
  EXPECT_FLOAT_EQ(points[1].point.y, 20.0F);
  EXPECT_EQ(points[1].visible, hrt::postprocess::KeypointVisible::Invisible);
}

TEST(GeneralKeypointTest, DecodesSimccDistributions) {
  // RTMPose layout: split ratio 2 on an 8 x 4 input, so 16 x and 8 y bins.
  const auto x = proto(DataType::Float32, {1, 2, 16});
  const auto y = proto(DataType::Float32, {1, 2, 8});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 4, 8})}, {x, y});
  hrt::Vector<float> xs(32, 0.0F);
  hrt::Vector<float> ys(16, 0.0F);
  xs[5] = 0.8F;
  ys[3] = 0.6F;
  xs[16 + 12] = 0.2F;
  ys[8 + 1] = 0.9F;
  const kitzoo::json::Json config{{"general_keypoint_type", "simcc"}, {"keypoint_threshold", 0.3F}};
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(x, xs), hrt::test::buffer_of(y, ys)},
                                     hrt::test::identity(8, 4)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& points = result.at(0)[0].keypoints;
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[0].point.x, 2.5F);
  EXPECT_FLOAT_EQ(points[0].point.y, 1.5F);
  EXPECT_FLOAT_EQ(points[0].score, 0.6F);
  EXPECT_EQ(points[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(points[1].point.x, 6.0F);
  EXPECT_FLOAT_EQ(points[1].point.y, 0.5F);
  EXPECT_FLOAT_EQ(points[1].score, 0.2F);
  EXPECT_EQ(points[1].visible, hrt::postprocess::KeypointVisible::Invisible);
}

TEST(GeneralKeypointTest, AppliesSigmoidToScores) {
  const auto x = proto(DataType::Float32, {1, 1, 4});
  const auto y = proto(DataType::Float32, {1, 1, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 2, 2})}, {x, y});
  const kitzoo::json::Json config{
      {"general_keypoint_type", "simcc"}, {"score_activation", "sigmoid"}, {"keypoint_threshold", 0.5F}};
  auto postprocessor = hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(
      result, {hrt::test::buffer_of<float>(x, {-1, -3, -2, -2}), hrt::test::buffer_of<float>(y, {-2, -2, -0.5F, -3})},
      hrt::test::identity(2, 2)));
  const auto& point = result.at(0)[0].keypoints[0];
  EXPECT_FLOAT_EQ(point.score, 1.0F / (1.0F + std::exp(1.0F)));
  EXPECT_EQ(point.visible, hrt::postprocess::KeypointVisible::Invisible);
}

TEST(GeneralKeypointTest, AbortsOnInvalidSimccOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto config = kitzoo::json::Json::parse(R"({"general_keypoint_type":"simcc"})");
  const auto x = proto(DataType::Float32, {1, 2, 8});
  const auto input = proto(DataType::Float32, {1, 3, 4, 4});
  for (const auto& outputs : hrt::Vector<hrt::Vector<hrt::core::BufferProto>>{
           {x}, {x, x, x}, {x, proto(DataType::Float32, {1, 3, 8})}, {x, proto(DataType::Float32, {2, 2, 8})}}) {
    hrt::test::FakeEngine engine({input}, outputs);
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("GeneralKeypoint", config, engine), "");
  }
  hrt::test::FakeEngine engine({input}, {x, x});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor(
                   "GeneralKeypoint", kitzoo::json::Json::parse(R"({"heatmap_refine":"taylor"})"), engine),
               "");
}

} // namespace
