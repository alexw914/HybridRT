// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/e2e_keypoint_test.cpp
// Description: Tests Ultralytics end-to-end pose rows, keypoint visibility and
//              source mapping.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/keypoint.hpp>

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

TEST(E2EKeypointTest, MapsRowsAndKeypointsWithoutNms) {
  // Two keypoints with visibility: 6 + 2 x 3 columns.
  const auto output = proto(DataType::Float32, {1, 3, 12});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  const auto config =
      Json::parse(R"({"class_num": 1, "keypoint_num": 2, "confidence_threshold": [0.5], "keypoint_threshold": 0.4})");
  auto decoder = hrt::postprocess::create_keypoint_postprocessor("E2EKeypoint", config, engine);
  const hrt::Vector<float> values{
      10,
      10,
      30,
      20,
      0.9F,
      0,
      20,
      15,
      0.9F,
      12,
      18,
      0.2F,
      // Overlapping, but already deduplicated by the export, so it stays.
      11,
      10,
      30,
      20,
      0.8F,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      // Padding row.
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
  };
  // The 40 x 20 band at y 10 of the input holds the 80 x 40 source region at (100, 200).
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {100, 200, 80, 40}, {40, 40}};
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 2U);
  const auto& object = result.at(0)[0];
  EXPECT_FLOAT_EQ(object.box.rect.x, 120.0F);
  EXPECT_FLOAT_EQ(object.box.rect.y, 200.0F);
  EXPECT_FLOAT_EQ(object.box.rect.width, 40.0F);
  EXPECT_FLOAT_EQ(object.box.score, 0.9F);
  ASSERT_EQ(object.keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.x, 140.0F);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.y, 210.0F);
  EXPECT_EQ(object.keypoints[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_FLOAT_EQ(object.keypoints[1].score, 0.2F);
  EXPECT_EQ(object.keypoints[1].visible, hrt::postprocess::KeypointVisible::Invisible);
  EXPECT_EQ(hrt::model::task_of("E2EKeypoint"), hrt::model::Task::Keypoint);
}

TEST(E2EKeypointTest, ReadsKeypointsWithoutVisibility) {
  const auto output = proto(DataType::Float32, {1, 1, 8});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  const auto config = Json::parse(R"({"class_num": 2, "keypoint_num": 1, "confidence_threshold": [0.25]})");
  auto decoder = hrt::postprocess::create_keypoint_postprocessor("E2EKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of<float>(output, {10, 10, 30, 20, 0.9F, 1, 20, 15})},
                               hrt::test::identity(40, 40)));
  ASSERT_TRUE(result.contains(1));
  const auto& point = result.at(1)[0].keypoints[0];
  EXPECT_FLOAT_EQ(point.point.x, 20.0F);
  EXPECT_FLOAT_EQ(point.score, 1.0F);
  EXPECT_EQ(point.visible, hrt::postprocess::KeypointVisible::Visible);
}

TEST(E2EKeypointTest, AbortsOnInvalidConfigOrOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 8, 8});
  const auto config = Json::parse(R"({"class_num": 1, "keypoint_num": 17, "confidence_threshold": [0.25]})");
  for (const auto& shape : hrt::Vector<hrt::core::Shape>{{1, 56, 8400}, {1, 300, 56}, {2, 300, 57}, {300, 57}}) {
    FakeEngine engine({input}, {proto(DataType::Float32, shape)});
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("E2EKeypoint", config, engine), "");
  }
  FakeEngine valid({input}, {proto(DataType::Float32, {1, 300, 57})});
  for (const char* text : {R"({"class_num": 1, "keypoint_num": 0, "confidence_threshold": [0.25]})",
                           R"({"class_num": 1, "keypoint_num": 17})",
                           R"({"class_num": 1, "keypoint_num": 17, "confidence_threshold": [0.25],
                               "iou_threshold": [0.45]})"})
    EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("E2EKeypoint", Json::parse(text), valid), "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(E2EKeypointTest, RunsUltralyticsEndToEndExportWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_e2e_pose");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLO26n end-to-end pose ONNX fixture or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::KeypointResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // bus.jpg: four people.
  ASSERT_TRUE(result->contains(0));
  EXPECT_EQ(result->at(0).size(), 4U);
  for (const auto& object : result->at(0))
    EXPECT_EQ(object.keypoints.size(), 17U);
}
#endif

} // namespace
