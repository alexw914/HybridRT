// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/yolo26_keypoint_test.cpp
// Description: Tests raw YOLO26 pose heads: distances, keypoint offsets and
//              visibility.
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

TEST(Yolo26KeypointTest, DecodesDistancesAndKeypointOffsets) {
  // Distances in cells and keypoints at (raw + cell + 0.5).
  constexpr int channels = 4 + 1 + 2 * 3;
  const auto output = proto(DataType::Float32, {1, channels, 4, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 32})}, {output});
  const auto config = Json::parse(R"({"class_num": 1, "keypoint_num": 2, "strides": [8],
                                      "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  auto decoder = hrt::postprocess::create_keypoint_postprocessor("Yolo26Keypoint", config, engine);
  hrt::Vector<float> values(channels * 16, 0.0F);
  auto at = [&](int channel, int y, int x) -> float& {
    return values[static_cast<std::size_t>(channel * 16 + y * 4 + x)];
  };
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x)
      at(4, y, x) = -10.0F;
  }
  // Cell (x 1, y 2): one cell to each side, score logit 3, keypoint (0.25, -0.25).
  for (int side = 0; side < 4; ++side)
    at(side, 2, 1) = 1.0F;
  at(4, 2, 1) = 3.0F;
  at(5, 2, 1) = 0.25F;
  at(6, 2, 1) = -0.25F;
  at(7, 2, 1) = 2.0F;
  hrt::postprocess::KeypointResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, hrt::test::identity(32, 32)));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& object = result.at(0)[0];
  EXPECT_NEAR(object.box.rect.x, 4.0F, 1e-4F);
  EXPECT_NEAR(object.box.rect.width, 16.0F, 1e-4F);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.x, 14.0F);
  EXPECT_FLOAT_EQ(object.keypoints[0].point.y, 18.0F);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.x, 12.0F);
  EXPECT_FLOAT_EQ(object.keypoints[1].point.y, 20.0F);
}

TEST(Yolo26KeypointTest, AbortsOnHeadsThatDoNotFit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto config = Json::parse(R"({"class_num": 1, "keypoint_num": 17, "strides": [8],
                                      "confidence_threshold": [0.25], "iou_threshold": [0.45]})");
  // A YOLOv8 head with distance bins: 64 + 1 + 51 channels.
  FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 32})}, {proto(DataType::Float32, {1, 116, 4, 4})});
  EXPECT_DEATH(hrt::postprocess::create_keypoint_postprocessor("Yolo26Keypoint", config, engine), "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(Yolo26KeypointTest, RunsRawHeadsWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_pose_raw");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLO26n-pose raw-head ONNX fixture or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::KeypointResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // bus.jpg: four people with 17 keypoints each.
  ASSERT_TRUE(result->contains(0));
  ASSERT_EQ(result->at(0).size(), 4U);
  for (const auto& object : result->at(0))
    EXPECT_EQ(object.keypoints.size(), 17U);
}
#endif

} // namespace
