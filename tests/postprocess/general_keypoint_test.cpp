// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/general_keypoint_test.cpp
// Description: Tests batched heatmap, SimCC and coordinate keypoint decoding
//              and source coordinate mapping.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

#include <cmath>
#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(GeneralKeypointTest, MapsNormalizedPointsToSource) {
  const auto output = proto(DataType::Float32, {1, 2, 2});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "point_channel_last"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  // The 16x16 input holds a 32x32 source crop at (100, 50).
  const hrt::preprocess::ImageTransform transform{{0, 0, 16, 16}, {100, 50, 32, 32}, {16, 16}};
  hrt::postprocess::KeypointResultMap result;
  result[0].emplace_back();
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::create_buffer<float>(output, {0.5F, 0.25F, 1.0F, 1.0F})}, transform));
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
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "point_channel_last"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  // real = (q + 100) / 200: (0.5, 0.25).
  auto buffer = hrt::test::create_buffer<std::int8_t>(output, {0, -50});
  buffer->set_quantization(0.005F, -100);
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
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "heatmap"})"), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::Vector<float> heatmap(20, 0.1F);
  heatmap[1 * 5 + 2] = 0.9F;
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::create_buffer(output, heatmap)}, hrt::test::identity(10, 8)));
  ASSERT_EQ(result.at(0).size(), 1U);
  EXPECT_NEAR(result.at(0)[0].keypoints[0].point.x, 4.0F, 0.5F);
  EXPECT_NEAR(result.at(0)[0].keypoints[0].point.y, 2.0F, 0.5F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].score, 0.9F);
}

TEST(GeneralKeypointTest, DecodesIndependentBatchSamples) {
  for (const auto* mode :
       {"heatmap", "heatmap_udp", "simcc", "point_with_visible", "point_channel_first", "point_channel_last"}) {
    for (const auto type : {DataType::Float32, DataType::Int8}) {
      for (const std::int64_t batch : {4, 8}) {
        SCOPED_TRACE(::testing::Message() << mode << " batch=" << batch << " type=" << static_cast<int>(type));
        const hrt::String name(mode);
        const auto heatmap = name == "heatmap" || name == "heatmap_udp";
        const auto distributions = name == "simcc" || name == "point_with_visible";
        hrt::Vector<hrt::core::BufferProto> outputs;
        if (heatmap) {
          outputs.push_back(proto(type, {batch, 1, 4, 4}));
        } else if (distributions) {
          outputs.assign(2, proto(type, {batch, 1, 4}));
          if (name == "point_with_visible")
            outputs.push_back(proto(type, {batch, 1, 3}));
        } else {
          outputs.push_back(proto(type, name == "point_channel_first" ? hrt::core::Shape{batch, 2, 1}
                                                                      : hrt::core::Shape{batch, 1, 2}));
        }
        hrt::test::FakeEngine engine({proto(DataType::Float32, {batch, 3, 16, 16})}, outputs);
        const kitzoo::json::Json config{
            {"general_keypoint_type", mode}, {"heatmap_refine", "none"}, {"keypoint_threshold", 0.3F}};
        auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
            "GeneralKeypoint", config, engine);
        hrt::Vector<hrt::preprocess::ImageTransform> transforms;
        for (int i = 0; i < batch; ++i)
          transforms.push_back({{0, 0, 16, 16}, {100 + i * 10, 50 + i * 7, 32, 48}, {16, 16}});
        hrt::Vector<hrt::postprocess::KeypointResultMap> results;
        for (int round = 0; round < 2; ++round) {
          hrt::Vector<hrt::Vector<float>> values;
          for (const auto& output : outputs) {
            std::int64_t volume = 1;
            for (const auto dimension : output.shape)
              volume *= dimension;
            values.emplace_back(static_cast<std::size_t>(volume), 0.0F);
          }
          for (std::int64_t i = 0; i < batch; ++i) {
            const auto x = (i + round) % 4, y = (i * 2 + round + 1) % 4;
            const auto score = i % 2 == 0 ? 0.75F : 0.25F;
            if (heatmap) {
              values[0][static_cast<std::size_t>(i * 16 + y * 4 + x)] = score;
            } else if (distributions) {
              values[0][static_cast<std::size_t>(i * 4 + x)] = score;
              values[1][static_cast<std::size_t>(i * 4 + y)] = score;
              if (name == "point_with_visible")
                values[2][static_cast<std::size_t>(i * 3 + 2)] = score;
            } else {
              values[0][static_cast<std::size_t>(i * 2)] = static_cast<float>(x) / 4.0F;
              values[0][static_cast<std::size_t>(i * 2 + 1)] = static_cast<float>(y) / 4.0F;
            }
          }
          hrt::core::Buffers buffers;
          for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (type == DataType::Float32) {
              buffers.push_back(hrt::test::create_buffer(outputs[i], values[i]));
            } else {
              hrt::Vector<std::int8_t> quantized;
              for (const auto value : values[i])
                quantized.push_back(static_cast<std::int8_t>(value * 4.0F - 4.0F));
              auto buffer = hrt::test::create_buffer(outputs[i], quantized);
              buffer->set_quantization(0.25F, -4);
              buffers.push_back(std::move(buffer));
            }
          }
          // A repeated call with a partial batch must discard the previous tail result.
          if (round == 1)
            transforms.pop_back();
          ASSERT_TRUE(postprocessor->process(results, buffers, transforms));
          ASSERT_EQ(results.size(), transforms.size());
          for (std::size_t i = 0; i < results.size(); ++i) {
            ASSERT_EQ(results[i].size(), 1U);
            ASSERT_EQ(results[i].at(0).size(), 1U);
            const auto& object = results[i].at(0)[0];
            ASSERT_EQ(object.keypoints.size(), 1U);
            const auto& point = object.keypoints[0];
            const auto cells = name == "heatmap_udp" ? 3.0F : 4.0F;
            const auto x = static_cast<float>((i + static_cast<std::size_t>(round)) % 4);
            const auto y = static_cast<float>((i * 2 + static_cast<std::size_t>(round) + 1) % 4);
            EXPECT_NEAR(point.point.x, static_cast<float>(100 + i * 10) + x / cells * 32.0F, 1e-4F);
            EXPECT_NEAR(point.point.y, static_cast<float>(50 + i * 7) + y / cells * 48.0F, 1e-4F);
            EXPECT_FLOAT_EQ(object.box.rect.x, static_cast<float>(100 + i * 10));
            EXPECT_FLOAT_EQ(object.box.rect.y, static_cast<float>(50 + i * 7));
            const auto scored = heatmap || distributions;
            const auto score = scored ? (i % 2 == 0 ? 0.75F : 0.25F) : 1.0F;
            EXPECT_FLOAT_EQ(point.score, score);
            EXPECT_EQ(point.visible, (heatmap || name == "simcc") && score <= 0.3F
                                         ? hrt::postprocess::KeypointVisible::Invisible
                                         : hrt::postprocess::KeypointVisible::Visible);
          }
        }
      }
    }
  }
}

TEST(GeneralKeypointTest, KeepsUnbatchedHeatmapLayout) {
  const auto output = proto(DataType::Float32, {2, 4, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", {{"heatmap_refine", "none"}}, engine);
  hrt::Vector<float> heatmaps(32, 0.0F);
  heatmaps[6] = 0.75F;
  heatmaps[29] = 0.5F;
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::create_buffer(output, heatmaps)}, hrt::test::identity(16, 16)));
  const auto& points = result.at(0)[0].keypoints;
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[0].point.x, 8.0F);
  EXPECT_FLOAT_EQ(points[0].point.y, 4.0F);
  EXPECT_FLOAT_EQ(points[1].point.x, 4.0F);
  EXPECT_FLOAT_EQ(points[1].point.y, 12.0F);
}

TEST(GeneralKeypointTest, DequantizesEveryUnbatchedHeatmap) {
  const auto output = proto(DataType::Int8, {2, 4, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", {{"heatmap_refine", "none"}}, engine);
  hrt::Vector<std::int8_t> heatmaps(32, 0);
  heatmaps[6] = 3;
  heatmaps[29] = 2;
  auto buffer = hrt::test::create_buffer(output, heatmaps);
  buffer->set_quantization(0.25F, 0);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(16, 16)));
  const auto& points = result.at(0)[0].keypoints;
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[1].point.x, 4.0F);
  EXPECT_FLOAT_EQ(points[1].point.y, 12.0F);
  EXPECT_FLOAT_EQ(points[1].score, 0.5F);
}

TEST(GeneralKeypointTest, RejectsMultipleSetsWithinBatchSample) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto& shape : hrt::Vector<hrt::core::Shape>{{4, 2, 1, 4, 4}, {4, 0, 4, 4}}) {
    hrt::test::FakeEngine engine({proto(DataType::Float32, {4, 3, 16, 16})}, {proto(DataType::Float32, shape)});
    EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
                     "GeneralKeypoint", kitzoo::json::Json::object(), engine),
                 "");
  }
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
    EXPECT_DEATH(
        hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine),
        "");
  }
  hrt::test::FakeEngine empty({proto(DataType::Float32, {1, 3, 8, 8})}, {});
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
                   "GeneralKeypoint", kitzoo::json::Json::object(), empty),
               "");
}

TEST(GeneralKeypointTest, AbortsOnInvalidVisibilityOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto config = kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_with_visible"})");
  const auto x = proto(DataType::Float32, {1, 2, 4});
  for (const auto& visibility : hrt::Vector<hrt::core::Shape>{{1, 2, 2}, {1, 2, 4}, {1, 1, 3}, {2, 2, 3}}) {
    hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})},
                                 {x, x, proto(DataType::Float32, visibility)});
    EXPECT_DEATH(
        hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine),
        "");
  }
  hrt::test::FakeEngine missing({proto(DataType::Float32, {1, 3, 8, 8})}, {x, x});
  EXPECT_DEATH(
      hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, missing),
      "");
}

TEST(GeneralKeypointTest, DecodesChannelFirstPoints) {
  const auto output = proto(DataType::Float32, {1, 2, 1});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_channel_first"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::create_buffer<float>(output, {0.25F, 0.5F})},
                                     hrt::test::identity(8, 8)));
  ASSERT_EQ(result.at(0)[0].keypoints.size(), 1U);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.x, 2.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.y, 4.0F);
}

TEST(GeneralKeypointTest, DecodesVisibilityDistributions) {
  const auto x = proto(DataType::Float32, {1, 1, 4});
  const auto y = proto(DataType::Float32, {1, 1, 2});
  const auto visible = proto(DataType::Float32, {1, 1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {x, y, visible});
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type":"point_with_visible"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result,
                                     {hrt::test::create_buffer<float>(x, {0, 1, 0, 0}),
                                      hrt::test::create_buffer<float>(y, {0, 1}),
                                      hrt::test::create_buffer<float>(visible, {0.1F, 0.2F, 0.7F})},
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
    auto postprocessor =
        hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine);
    hrt::postprocess::KeypointResultMap result;
    ASSERT_TRUE(
        postprocessor->process(result, {hrt::test::create_buffer(output, heatmap)}, hrt::test::identity(192, 256)));
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

TEST(GeneralKeypointTest, DecodesUdpHeatmapsAgainstCornerCells) {
  // UDP maps cell 0 and cell W - 1 onto the input edges: 48 x 64 maps of a
  // 192 x 256 input scale by 192 / 47 and 256 / 63.
  const auto output = proto(DataType::Float32, {1, 2, 64, 48});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 256, 192})}, {output});
  auto heatmap = gaussian_map(48, 64, 20.3F, 30.7F, 0.9F);
  const auto corner = gaussian_map(48, 64, 47.0F, 63.0F, 0.9F);
  heatmap.insert(heatmap.end(), corner.begin(), corner.end());
  for (const auto* refine : {"dark", "none"}) {
    const kitzoo::json::Json config{{"general_keypoint_type", "heatmap_udp"}, {"heatmap_refine", refine}};
    auto postprocessor =
        hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine);
    hrt::postprocess::KeypointResultMap result;
    ASSERT_TRUE(
        postprocessor->process(result, {hrt::test::create_buffer(output, heatmap)}, hrt::test::identity(192, 256)));
    const auto& points = result.at(0)[0].keypoints;
    ASSERT_EQ(points.size(), 2U);
    if (refine == hrt::String("dark")) {
      EXPECT_NEAR(points[0].point.x, 20.3F * 192.0F / 47.0F, 0.1F);
      EXPECT_NEAR(points[0].point.y, 30.7F * 256.0F / 63.0F, 0.1F);
    } else {
      EXPECT_FLOAT_EQ(points[0].point.x, 20.0F * 192.0F / 47.0F);
      EXPECT_FLOAT_EQ(points[0].point.y, 31.0F * 256.0F / 63.0F);
    }
    // The corner cell is the input corner; DARK may move an edge peak by less
    // than one cell, as mmpose's edge-padded expansion does.
    const auto tolerance = refine == hrt::String("dark") ? 192.0F / 47.0F : 0.0F;
    EXPECT_NEAR(points[1].point.x, 192.0F, tolerance);
    EXPECT_NEAR(points[1].point.y, 256.0F, tolerance * 256.0F / 192.0F);
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
  auto postprocessor = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
      "GeneralKeypoint", kitzoo::json::Json::parse(R"({"general_keypoint_type": "heatmap"})"), engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::create_buffer(output, heatmap)}, hrt::test::identity(64, 64)));
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
  auto postprocessor =
      hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::create_buffer(x, xs), hrt::test::create_buffer(y, ys)},
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

TEST(GeneralKeypointTest, DecodesInt8SimccDistributions) {
  // The float distributions of DecodesSimccDistributions, quantized at 0.01 with zero point -100.
  const auto x = proto(DataType::Int8, {1, 2, 16});
  const auto y = proto(DataType::Int8, {1, 2, 8});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 4, 8})}, {x, y});
  hrt::Vector<std::int8_t> xs(32, -100);
  hrt::Vector<std::int8_t> ys(16, -100);
  xs[5] = -20;
  ys[3] = -40;
  xs[16 + 12] = -80;
  ys[8 + 1] = -10;
  const kitzoo::json::Json config{{"general_keypoint_type", "simcc"}, {"keypoint_threshold", 0.3F}};
  auto postprocessor =
      hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine);
  auto x_buffer = hrt::test::create_buffer(x, xs);
  auto y_buffer = hrt::test::create_buffer(y, ys);
  x_buffer->set_quantization(0.01F, -100);
  y_buffer->set_quantization(0.01F, -100);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {x_buffer, y_buffer}, hrt::test::identity(8, 4)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& points = result.at(0)[0].keypoints;
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[0].point.x, 2.5F);
  EXPECT_FLOAT_EQ(points[0].point.y, 1.5F);
  EXPECT_NEAR(points[0].score, 0.6F, 1e-6F);
  EXPECT_EQ(points[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(points[1].point.x, 6.0F);
  EXPECT_FLOAT_EQ(points[1].point.y, 0.5F);
  EXPECT_NEAR(points[1].score, 0.2F, 1e-6F);
  EXPECT_EQ(points[1].visible, hrt::postprocess::KeypointVisible::Invisible);
}

TEST(GeneralKeypointTest, AppliesSigmoidToScores) {
  const auto x = proto(DataType::Float32, {1, 1, 4});
  const auto y = proto(DataType::Float32, {1, 1, 4});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 2, 2})}, {x, y});
  const kitzoo::json::Json config{
      {"general_keypoint_type", "simcc"}, {"score_activation", "sigmoid"}, {"keypoint_threshold", 0.5F}};
  auto postprocessor =
      hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(postprocessor->process(
      result,
      {hrt::test::create_buffer<float>(x, {-1, -3, -2, -2}), hrt::test::create_buffer<float>(y, {-2, -2, -0.5F, -3})},
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
    EXPECT_DEATH(
        hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("GeneralKeypoint", config, engine),
        "");
  }
  hrt::test::FakeEngine engine({input}, {x, x});
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>(
                   "GeneralKeypoint", kitzoo::json::Json::parse(R"({"heatmap_refine":"taylor"})"), engine),
               "");
}

} // namespace
