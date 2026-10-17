// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/rotated_test.cpp
// Description: Tests rotated-box decoding, end-to-end OBB rows and rotated
//              NMS.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/rotated.hpp>

#include <gtest/gtest.h>
#include <numbers>

#include "fake_engine.hpp"

#ifdef HRT_ORT_MODEL_TEST
#include <opencv2/imgcodecs.hpp>

#include "../engine/template_config.hpp"
#endif

namespace {

using hrt::core::DataType;
using hrt::test::proto;

TEST(RotatedTest, NmsMergesOverlappingBoxes) {
  hrt::postprocess::RotatedResult boxes{
      {{{10, 10}, {10, 4}, 30}, 0.8F, 0, -1},
      {{{10, 10}, {10, 4}, 31}, 0.9F, 0, -1},
      {{{40, 40}, {10, 4}, 0}, 0.7F, 0, -1},
  };
  hrt::postprocess::nms_rotated(boxes, 0.5F);
  ASSERT_EQ(boxes.size(), 2U);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9F);
  EXPECT_FLOAT_EQ(boxes[1].score, 0.7F);
}

TEST(RotatedTest, NmsUsesAreaRatioForContainedBoxes) {
  const hrt::postprocess::RotatedResult original{
      {{{50, 50}, {100, 100}, 30}, 0.9F, 0, -1},
      {{{50, 50}, {10, 10}, 30}, 0.8F, 0, -1},
  };
  auto boxes = original;
  hrt::postprocess::nms_rotated(boxes, 0.5F);
  EXPECT_EQ(boxes.size(), 2U);
  boxes = original;
  hrt::postprocess::nms_rotated(boxes, 0.005F);
  ASSERT_EQ(boxes.size(), 1U);
  EXPECT_FLOAT_EQ(boxes.front().score, 0.9F);
  boxes = {original.front(), original.front()};
  hrt::postprocess::nms_rotated(boxes, 0.5F);
  EXPECT_EQ(boxes.size(), 1U);
}

TEST(RotatedTest, GeneralDecodesObb) {
  const auto output = proto(DataType::Float32, {1, 7, 1});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 64, 64})}, {output});
  auto config =
      kitzoo::json::Json::parse(R"({"class_num": 2, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  auto postprocessor = hrt::postprocess::create_rotated_postprocessor("GeneralRotated", config, engine);

  // cx, cy, w, h, class 0 and 1 scores, angle pi / 12.
  const hrt::Vector<float> values{20, 15, 20, 10, 0.1F, 0.9F, std::numbers::pi_v<float> / 12.0F};
  hrt::postprocess::RotatedResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, values)}, hrt::test::identity(64, 64)));
  EXPECT_FALSE(result.contains(0));
  ASSERT_EQ(result.at(1).size(), 1U);
  const auto& rect = result.at(1)[0].rect;
  EXPECT_FLOAT_EQ(rect.center.x, 20.0F);
  EXPECT_FLOAT_EQ(rect.center.y, 15.0F);
  EXPECT_FLOAT_EQ(rect.size.width, 20.0F);
  EXPECT_FLOAT_EQ(rect.size.height, 10.0F);
  EXPECT_NEAR(rect.angle, 15.0F, 1e-4F);

  EXPECT_EQ(hrt::postprocess::check_config("GeneralRotated", kitzoo::json::Json::parse(R"({"unknown_key": 1})")).size(),
            1U);
}

TEST(RotatedTest, RejectsNonuniformSourceMapping) {
  const auto output = proto(DataType::Float32, {1, 6, 1});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 64, 64})}, {output});
  const auto config =
      kitzoo::json::Json::parse(R"({"class_num":1,"confidence_threshold":[0.5],"iou_threshold":[0.45]})");
  auto postprocessor = hrt::postprocess::create_rotated_postprocessor("GeneralRotated", config, engine);
  const auto buffer = hrt::test::buffer_of<float>(output, {20, 15, 20, 10, 0.9F, std::numbers::pi_v<float> / 4.0F});
  hrt::postprocess::RotatedResultMap result;
  const hrt::preprocess::ImageTransform stretched{{0, 0, 64, 64}, {0, 0, 128, 64}, {64, 64}};
  EXPECT_FALSE(postprocessor->process(result, {buffer}, stretched));
  EXPECT_TRUE(result.empty());
  const hrt::preprocess::ImageTransform rounded{{0, 50, 640, 379}, {0, 0, 1280, 759}, {640, 480}};
  EXPECT_TRUE(postprocessor->process(result, {buffer}, rounded));
  EXPECT_FALSE(result.empty());
  const hrt::preprocess::ImageTransform uniform{{0, 0, 64, 64}, {10, 20, 128, 128}, {64, 64}};
  ASSERT_TRUE(postprocessor->process(result, {buffer}, uniform));
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& rect = result.at(0)[0].rect;
  EXPECT_FLOAT_EQ(rect.center.x, 50.0F);
  EXPECT_FLOAT_EQ(rect.center.y, 50.0F);
  EXPECT_FLOAT_EQ(rect.size.width, 40.0F);
  EXPECT_FLOAT_EQ(rect.size.height, 20.0F);
  EXPECT_NEAR(rect.angle, 45.0F, 1e-4F);
}

TEST(RotatedTest, E2EDecodesRowsWithoutNms) {
  const auto output = proto(DataType::Float32, {1, 3, 7});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 64, 64})}, {output});
  const auto config = kitzoo::json::Json::parse(R"({"class_num": 2, "confidence_threshold": [0.5]})");
  auto postprocessor = hrt::postprocess::create_rotated_postprocessor("E2ERotated", config, engine);
  const auto angle = std::numbers::pi_v<float> / 12.0F;
  // cx, cy, w, h, score, class, angle: two identical boxes both stay (the
  // export already deduplicated them) and a padding row is skipped.
  const hrt::Vector<float> values{
      20, 15, 20, 10, 0.9F, 1, angle, 20, 15, 20, 10, 0.8F, 1, angle, 0, 0, 0, 0, 0, 0, 0,
  };
  // The 64 x 64 input holds a 128 x 128 source at (10, 20).
  const hrt::preprocess::ImageTransform transform{{0, 0, 64, 64}, {10, 20, 128, 128}, {64, 64}};
  hrt::postprocess::RotatedResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, values)}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(1).size(), 2U);
  const auto& box = result.at(1)[0];
  EXPECT_FLOAT_EQ(box.rect.center.x, 50.0F);
  EXPECT_FLOAT_EQ(box.rect.center.y, 50.0F);
  EXPECT_FLOAT_EQ(box.rect.size.width, 40.0F);
  EXPECT_FLOAT_EQ(box.rect.size.height, 20.0F);
  EXPECT_NEAR(box.rect.angle, 15.0F, 1e-4F);
  EXPECT_FLOAT_EQ(box.score, 0.9F);
  EXPECT_EQ(hrt::model::task_of("E2ERotated"), hrt::model::Task::Rotated);
}

TEST(RotatedTest, E2EAbortsOnInvalidConfigOrOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 8, 8});
  const auto config = kitzoo::json::Json::parse(R"({"class_num": 15, "confidence_threshold": [0.25]})");
  for (const auto& shape : hrt::Vector<hrt::core::Shape>{{1, 20, 21504}, {1, 300, 6}, {2, 300, 7}}) {
    hrt::test::FakeEngine engine({input}, {proto(DataType::Float32, shape)});
    EXPECT_DEATH(hrt::postprocess::create_rotated_postprocessor("E2ERotated", config, engine), "");
  }
  hrt::test::FakeEngine valid({input}, {proto(DataType::Float32, {1, 300, 7})});
  EXPECT_DEATH(hrt::postprocess::create_rotated_postprocessor(
                   "E2ERotated", kitzoo::json::Json::parse(R"({"class_num": 15, "iou_threshold": [0.45]})"), valid),
               "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(RotatedTest, RunsUltralyticsEndToEndExportWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_e2e_obb");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/airport.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLO26n end-to-end OBB ONNX fixture or airport.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::RotatedResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  std::size_t boxes = 0;
  for (const auto& [cls, items] : *result)
    boxes += items.size();
  // airport.jpg: ten boxes, as Ultralytics reports for the same export.
  EXPECT_EQ(boxes, 10U);
}

TEST(RotatedTest, RunsRawObbHeadsWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolov8n_obb_raw");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/airport.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLOv8n-obb raw-head ONNX fixture or airport.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::RotatedResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // airport.jpg: eleven planes.
  ASSERT_TRUE(result->contains(0));
  EXPECT_EQ(result->at(0).size(), 11U);
}
#endif

} // namespace
