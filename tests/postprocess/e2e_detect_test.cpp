// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/e2e_detect_test.cpp
// Description: Tests Ultralytics end-to-end detection rows, thresholds and
//              source mapping.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/detect.hpp>

#include <gtest/gtest.h>
#include <limits>

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

TEST(E2EDetectTest, MapsRowsWithoutNms) {
  const auto output = proto(DataType::Float32, {1, 5, 6});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  const auto config = Json::parse(R"({"class_num":2,"confidence_threshold":[0.5,0.7]})");
  auto decoder = hrt::postprocess::create_detect_postprocessor("E2EDetect", config, engine);
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const hrt::Vector<float> values{
      // Two overlapping boxes both stay: the export already deduplicated them.
      10,
      10,
      30,
      20,
      0.9F,
      0,
      12,
      10,
      30,
      20,
      0.6F,
      0,
      // Below the class 1 threshold, a class id out of range and a padding row.
      10,
      10,
      30,
      20,
      0.6F,
      1,
      10,
      10,
      30,
      20,
      0.9F,
      nan,
      0,
      0,
      0,
      0,
      0,
      0,
  };
  // The 40 x 20 band at y 10 of the input holds the 80 x 40 source region at (100, 200).
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {100, 200, 80, 40}, {40, 40}};
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 2U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 120.0F);
  EXPECT_FLOAT_EQ(box.rect.y, 200.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 40.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 20.0F);
  EXPECT_FLOAT_EQ(box.score, 0.9F);
  EXPECT_EQ(box.type, 0);
  EXPECT_EQ(hrt::model::task_of("E2EDetect"), hrt::model::Task::Detect);
}

TEST(E2EDetectTest, DequantizesAndClipsToSourceRegion) {
  const auto output = proto(DataType::Int8, {1, 1, 6});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 100, 100})}, {output});
  auto decoder = hrt::postprocess::create_detect_postprocessor("E2EDetect", Json::parse(R"({"class_num":2,
      "confidence_threshold":[0.25]})"),
                                                               engine);
  // real = q: box (-20, 30)-(60, 90), score 1 after scale, class 1.
  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {-20, 30, 60, 90, 100, 1});
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {buffer}, hrt::test::identity(100, 100)));
  ASSERT_TRUE(result.contains(1));
  const auto& box = result.at(1)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 0.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 60.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 60.0F);
}

TEST(E2EDetectTest, AbortsOnInvalidConfigOrOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 8, 8});
  const auto config = Json::parse(R"({"class_num":80,"confidence_threshold":[0.25]})");
  for (const auto& shape : hrt::Vector<hrt::core::Shape>{{1, 84, 8400}, {1, 300, 7}, {2, 300, 6}, {300, 6}}) {
    FakeEngine engine({input}, {proto(DataType::Float32, shape)});
    EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("E2EDetect", config, engine), "");
  }
  FakeEngine valid({input}, {proto(DataType::Float32, {1, 300, 6})});
  for (const char* text : {R"({"class_num":0,"confidence_threshold":[0.25]})", R"({"class_num":80})",
                           R"({"class_num":80,"confidence_threshold":[0.25],"iou_threshold":[0.45]})"})
    EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("E2EDetect", Json::parse(text), valid), "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(E2EDetectTest, RunsTemplateModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_e2e_detect");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "Template end-to-end ONNX fixture or bus.jpg is unavailable";
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto model = hrt::model::ImageModel<hrt::postprocess::DetectResultMap>::create(algo);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // bus.jpg: four people and the bus.
  ASSERT_TRUE(result->contains(0));
  EXPECT_EQ(result->at(0).size(), 4U);
  ASSERT_TRUE(result->contains(5));
  EXPECT_EQ(result->at(5).size(), 1U);
}

#endif

} // namespace
