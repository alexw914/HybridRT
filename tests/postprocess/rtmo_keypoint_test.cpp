// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/rtmo_keypoint_test.cpp
// Description: Tests RTMO candidate thresholds, NMS, keypoint visibility and
//              source mapping.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/postprocessor_factory.hpp>

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

TEST(RtmoKeypointTest, KeepsBestCandidatesAndMapsKeypoints) {
  const auto boxes = proto(DataType::Float32, {1, 3, 5});
  const auto keypoints = proto(DataType::Float32, {1, 3, 2, 3});
  // Keypoints first: outputs are matched by rank, not order.
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {keypoints, boxes});
  auto decoder = hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("RtmoKeypoint",
                                                                                             Json::object(), engine);
  const hrt::Vector<float> box_values{
      10,
      10,
      30,
      20,
      0.9F,
      // Overlaps the first: removed by NMS.
      11,
      10,
      30,
      20,
      0.8F,
      // Below the default 0.15 score threshold.
      0,
      0,
      5,
      5,
      0.1F,
  };
  const hrt::Vector<float> point_values{
      20, 15, 0.9F, 12, 18, 0.2F, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  };
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {100, 200, 80, 40}, {40, 40}};
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(
      result, {hrt::test::create_buffer(keypoints, point_values), hrt::test::create_buffer(boxes, box_values)},
      transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 1U);
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
  EXPECT_EQ(hrt::model::task_type("RtmoKeypoint"), hrt::model::Task::Keypoint);
}

TEST(RtmoKeypointTest, AppliesConfiguredThresholds) {
  const auto boxes = proto(DataType::Float32, {1, 2, 5});
  const auto keypoints = proto(DataType::Float32, {1, 2, 1, 3});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {boxes, keypoints});
  const auto config = Json::parse(R"({"confidence_threshold":[0.05],"iou_threshold":[0.99],"keypoint_threshold":0.1})");
  auto decoder =
      hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("RtmoKeypoint", config, engine);
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(result,
                               {hrt::test::create_buffer<float>(boxes, {10, 10, 30, 20, 0.9F, 11, 10, 30, 20, 0.1F}),
                                hrt::test::create_buffer<float>(keypoints, {20, 15, 0.2F, 21, 15, 0.05F})},
                               hrt::test::identity(40, 40)));
  ASSERT_EQ(result.at(0).size(), 2U);
  EXPECT_EQ(result.at(0)[0].keypoints[0].visible, hrt::postprocess::KeypointVisible::Visible);
  EXPECT_EQ(result.at(0)[1].keypoints[0].visible, hrt::postprocess::KeypointVisible::Invisible);
}

TEST(RtmoKeypointTest, AbortsOnInvalidConfigOrOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 8, 8});
  const auto boxes = proto(DataType::Float32, {1, 4, 5});
  const auto keypoints = proto(DataType::Float32, {1, 4, 17, 3});
  for (const auto& outputs :
       hrt::Vector<hrt::Vector<hrt::core::BufferProto>>{{boxes},
                                                        {boxes, proto(DataType::Float32, {1, 5, 17, 3})},
                                                        {boxes, proto(DataType::Float32, {1, 4, 17, 2})},
                                                        {proto(DataType::Float32, {1, 4, 4}), keypoints},
                                                        {proto(DataType::Float32, {2, 4, 5}), keypoints}}) {
    FakeEngine engine({input}, outputs);
    EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("RtmoKeypoint",
                                                                                             Json::object(), engine),
                 "");
  }
  FakeEngine valid({input}, {boxes, keypoints});
  for (const char* text : {R"({"confidence_threshold":[0.1,0.2]})", R"({"iou_threshold":[1.5]})", R"({"class_num":1})"})
    EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::KeypointResultMap>("RtmoKeypoint",
                                                                                             Json::parse(text), valid),
                 "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(RtmoKeypointTest, RunsFixedRtmoModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("rtmo_s_pose");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "RTMO-s ONNX fixture with fixed outputs or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::KeypointResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->contains(0));
  EXPECT_EQ(result->at(0).size(), 5U);
  for (const auto& object : result->at(0))
    EXPECT_EQ(object.keypoints.size(), 17U);
}
#endif

} // namespace
