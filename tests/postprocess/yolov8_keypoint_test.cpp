// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/yolov8_keypoint_test.cpp
// Description: Tests raw YOLOv8 pose heads: distance bins, keypoint offsets,
//              visibility and any keypoint count.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/keypoint.hpp>

#include <cmath>
#include <gtest/gtest.h>

#include "fake_engine.hpp"

#ifdef HRT_ORT_MODEL_TEST
#include <opencv2/imgcodecs.hpp>

#include "../engine/template_config.hpp"
#endif

namespace {

using hrt::core::DataType;
using hrt::test::FakeEngine;
using hrt::test::proto;
using Json = kitzoo::json::Json;

// One 4 x 4 head of a 32 x 32 input (stride 8) with reg_max 2, one class and
// two keypoints with visibility: 4 x 2 + 1 + 2 x 3 = 15 channels.
constexpr int kChannels = 15;

auto head_values() -> hrt::Vector<float> {
  hrt::Vector<float> values(kChannels * 16, 0.0F);
  auto at = [&](int channel, int y, int x) -> float& {
    return values[static_cast<std::size_t>(channel * 16 + y * 4 + x)];
  };
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x)
      at(8, y, x) = -10.0F;
  }
  // Cell (x 1, y 2): every side one cell from the center (bins 0 and 1 at
  // -20 and 20), score logit 3, keypoints (0.25, -0.25) visible and (0, 0)
  // hidden.
  for (int side = 0; side < 4; ++side) {
    at(side * 2, 2, 1) = -20.0F;
    at(side * 2 + 1, 2, 1) = 20.0F;
  }
  at(8, 2, 1) = 3.0F;
  at(9, 2, 1) = 0.25F;
  at(10, 2, 1) = -0.25F;
  at(11, 2, 1) = 2.0F;
  at(14, 2, 1) = -2.0F;
  return values;
}

auto config() -> Json {
  return Json::parse(R"({"class_num": 1, "keypoint_num": 2,
                         "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
}

TEST(YoloV8KeypointTest, InfersAndValidatesKeypointCountsAcrossPoseFormats) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto* type : {"GeneralYoloPose", "E2EKeypoint", "Yolo26Keypoint", "YoloV8Keypoint"}) {
    const hrt::String name(type);
    auto options = Json::parse(R"({"class_num":1,"confidence_threshold":[0.5]})");
    if (name != "E2EKeypoint")
      options["iou_threshold"] = {0.45};
    if (name == "YoloV8Keypoint")
      options["reg_max"] = 2;
    for (const auto channels : {4, 9, 6}) {
      hrt::core::Shape shape{1, 5 + channels, 1};
      if (name == "E2EKeypoint")
        shape = {1, 1, 6 + channels};
      else if (name == "Yolo26Keypoint" || name == "YoloV8Keypoint")
        shape = {1, (name == "YoloV8Keypoint" ? 9 : 5) + channels, 2, 2};
      FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, shape)});
      auto automatic = options;
      if (channels == 6)
        EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor(type, automatic, engine), "");
      else
        EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor(type, automatic, engine), nullptr);
      automatic["keypoint_num"] = channels == 9 ? 3 : 2;
      EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor(type, automatic, engine), nullptr);
      for (const auto count : {0, 7}) {
        automatic["keypoint_num"] = count;
        EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor(type, automatic, engine), "");
      }
    }
  }
}

TEST(YoloV8KeypointTest, RequiresTwoExplicitShapeParameters) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, {1, 24, 2, 2})});
  auto options = Json::parse(R"({"class_num":1,"confidence_threshold":[0.5],"iou_threshold":[0.45]})");
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, engine), "");
  EXPECT_FALSE(hrt::postprocess::check_config("YoloV8Keypoint", options).empty());
  options["reg_max"] = 2;
  EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, engine), nullptr);
  options["keypoint_num"] = 5;
  EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, engine), nullptr);
  options["keypoint_num"] = 4;
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, engine), "");
}

TEST(YoloV8KeypointTest, InfersMissingClassCountAndChecksExplicitValues) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine unique({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, {1, 13, 2, 2})});
  auto options = Json::parse(R"({"reg_max":2,"keypoint_num":2,"confidence_threshold":[0.5],"iou_threshold":[0.45]})");
  EXPECT_TRUE(hrt::postprocess::check_config("YoloV8Keypoint", options).empty());
  auto decoder = hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, unique);
  hrt::Vector<float> values(13 * 4, -20.0F);
  values[8 * 4] = 3.0F;
  values[9 * 4] = 0.25F;
  values[10 * 4] = 0.5F;
  hrt::postprocess::KeypointResultMap result;
  const auto output = proto(DataType::Float32, {1, 13, 2, 2});
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, hrt::test::identity(16, 16)));
  ASSERT_EQ(result.at(0).size(), 1U);
  ASSERT_EQ(result.at(0)[0].keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].point.x, 4.0F);
  EXPECT_FLOAT_EQ(result.at(0)[0].keypoints[0].score, 1.0F);
  FakeEngine ambiguous({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, {1, 15, 2, 2})});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, ambiguous), "");
  options["class_num"] = 1;
  EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, ambiguous), nullptr);
  for (const auto classes : {0, 2, 7}) {
    options["class_num"] = classes;
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", options, ambiguous), "");
  }
  for (const auto* key : {"class_num", "reg_max", "keypoint_num"}) {
    auto one = Json::parse(R"({"confidence_threshold":[0.5],"iou_threshold":[0.45]})");
    one[key] = 1;
    EXPECT_FALSE(hrt::postprocess::check_config("YoloV8Keypoint", one).empty());
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", one, unique), "");
  }
  auto empty = Json::parse(R"({"confidence_threshold":[0.5],"iou_threshold":[0.45]})");
  EXPECT_FALSE(hrt::postprocess::check_config("YoloV8Keypoint", empty).empty());
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", empty, unique), "");
}

TEST(YoloV8KeypointTest, RequiresRegMaxForAmbiguousKeypointDimensions) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {proto(DataType::Float32, {1, 21, 2, 2})});
  auto config = Json::parse(R"({"class_num": 1, "keypoint_num": 4,
                               "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", config, engine), "");
  for (const auto bins : {2, 3}) {
    config["reg_max"] = bins;
    EXPECT_NE(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", config, engine), nullptr);
  }
  config["reg_max"] = 1;
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", config, engine), "");
}

TEST(YoloV8KeypointTest, DecodesBinsAndKeypointOffsets) {
  const auto output = proto(DataType::Float32, {1, kChannels, 4, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 32})}, {output});
  auto decoder = hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", config(), engine);
  ASSERT_NE(decoder, nullptr);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, head_values())}, hrt::test::identity(32, 32)));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& object = result.at(0)[0];
  // Center (1.5, 2.5) cells, one cell to each side.
  EXPECT_NEAR(object.box.rect.x, 4.0F, 1e-3F);
  EXPECT_NEAR(object.box.rect.y, 12.0F, 1e-3F);
  EXPECT_NEAR(object.box.rect.width, 16.0F, 1e-3F);
  EXPECT_NEAR(object.box.rect.height, 16.0F, 1e-3F);
  EXPECT_NEAR(object.box.score, 1.0F / (1.0F + std::exp(-3.0F)), 1e-6F);
  ASSERT_EQ(object.keypoints.size(), 2U);
  // ((raw * 2 + cell) * stride): (0.5 + 1) * 8 and (-0.5 + 2) * 8.
  EXPECT_FLOAT_EQ(object.keypoints[0].point.x, 12.0F);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.y, 12.0F);
  EXPECT_EQ(object.keypoints[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.x, 8.0F);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.y, 16.0F);
  EXPECT_EQ(object.keypoints[1].visible, hrt::postprocess::KeypointVisible::Invisible);
  EXPECT_EQ(hrt::model::task_of("YoloV8Keypoint"), hrt::model::Task::Keypoint);
}

TEST(YoloV8KeypointTest, AbortsOnHeadsThatDoNotFit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 32, 32});
  const auto output = proto(DataType::Float32, {1, kChannels, 4, 4});
  for (const auto& shape : hrt::Vector<hrt::core::Shape>{{1, kChannels + 1, 4, 4}, {1, kChannels, 2, 4}}) {
    FakeEngine engine({input}, {proto(DataType::Float32, shape)});
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", config(), engine), "");
  }
  FakeEngine valid({input}, {output});
  for (const char* key : {"keypoint_num", "strides"}) {
    auto broken = config();
    broken[key] = key == hrt::String("strides") ? Json::array({8, 16}) : Json(0);
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("YoloV8Keypoint", broken, valid), "");
  }
}

#ifdef HRT_ORT_MODEL_TEST
TEST(YoloV8KeypointTest, RunsFaceLandmarkModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolov8n_face");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/zidane.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLOv8n-face ONNX fixture or zidane.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::KeypointResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // zidane.jpg: two faces with five landmarks each, inside their boxes.
  ASSERT_TRUE(result->contains(0));
  ASSERT_EQ(result->at(0).size(), 2U);
  for (const auto& object : result->at(0)) {
    ASSERT_EQ(object.keypoints.size(), 5U);
    for (const auto& keypoint : object.keypoints) {
      EXPECT_GE(keypoint.point.x, object.box.rect.x);
      EXPECT_LE(keypoint.point.x, object.box.rect.x + object.box.rect.width);
    }
  }
}
#endif

} // namespace
