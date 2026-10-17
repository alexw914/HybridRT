// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/detr_detect_test.cpp
// Description: Tests DETR query formats, score activation and source mapping.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/detect.hpp>

#include <cmath>
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

TEST(DetrDetectTest, SelectsOneClassAndKeepsOverlappingQueries) {
  const auto output = proto(DataType::Float32, {1, 4, 6});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  const auto config = Json::parse(R"({"class_num":2,"confidence_threshold":[0.5,0.7]})");
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", config, engine);
  const hrt::Vector<float> values{
      0.5F, 0.5F, 0.5F, 0.25F, 0.9F,  0.8F, 0.5F, 0.5F, 0.5F, 0.25F, 0.7F, 0.1F,
      0.5F, 0.5F, 0.5F, 0.25F, 0.55F, 0.6F, 0.5F, 0.5F, 0.5F, 0.25F, 0.1F, 0.7F,
  };
  const hrt::preprocess::ImageTransform transform{{0, 10, 40, 20}, {100, 200, 80, 40}, {40, 40}};
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(0).size(), 2U);
  const auto& box = result.at(0)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 120.0F);
  EXPECT_FLOAT_EQ(box.rect.y, 210.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 40.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 20.0F);
  EXPECT_FLOAT_EQ(box.score, 0.9F);
  EXPECT_EQ(box.type, 0);
  EXPECT_EQ(hrt::model::task_of("DetrDetect"), hrt::model::Task::Detect);
}

TEST(DetrDetectTest, DequantizesAndClipsToSourceRegion) {
  const auto output = proto(DataType::Int8, {1, 1, 6});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 100, 100})}, {output});
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(R"({"class_num":2})"), engine);
  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {50, 50, 100, 100, 10, 90});
  buffer->scale = 0.01F;
  hrt::postprocess::DetectResultMap result;
  const hrt::preprocess::ImageTransform transform{{0, 25, 100, 50}, {10, 20, 200, 100}, {100, 100}};
  ASSERT_TRUE(decoder->process(result, {buffer}, transform));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(1).size(), 1U);
  const auto& box = result.at(1)[0];
  EXPECT_FLOAT_EQ(box.rect.x, 10.0F);
  EXPECT_FLOAT_EQ(box.rect.y, 20.0F);
  EXPECT_FLOAT_EQ(box.rect.width, 200.0F);
  EXPECT_FLOAT_EQ(box.rect.height, 100.0F);
  EXPECT_NEAR(box.score, 0.9F, 1e-6F);
}

TEST(DetrDetectTest, RejectsNonfiniteAndEmptyBoxesAndClearsResults) {
  const auto output = proto(DataType::Float32, {1, 4, 5});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(R"({"class_num":1})"), engine);
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto inf = std::numeric_limits<float>::infinity();
  const hrt::Vector<float> values{nan,  0.5F, 0.2F, 0.2F, 0.9F, 0.5F, 0.5F, -0.2F, 0.2F, 0.9F,
                                  0.5F, 0.5F, 0.2F, 0.2F, inf,  2.0F, 2.0F, 0.2F,  0.2F, 0.9F};
  hrt::postprocess::DetectResultMap result{{0, {{{0, 0, 1, 1}, 1.0F}}}};
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of(output, values)}, hrt::test::identity(40, 40)));
  EXPECT_TRUE(result.empty());
  EXPECT_FALSE(decoder->process(result, {}, hrt::test::identity(40, 40)));
  const auto wrong = proto(DataType::Float32, {1, 1, 5});
  EXPECT_FALSE(decoder->process(result, {hrt::test::buffer_of<float>(wrong, {0.5F, 0.5F, 0.2F, 0.2F, 0.9F})},
                                hrt::test::identity(40, 40)));
}

TEST(DetrDetectTest, ChecksConfigAndOutputContract) {
  EXPECT_TRUE(hrt::postprocess::check_config("DetrDetect", Json::parse(R"({"class_num":2})")).empty());
  for (const auto* text : {R"({"class_num":"two"})", R"({"class_num":-1})", R"({"class_num":2,"iou_threshold":[0.5]})",
                           R"({"class_num":2,"confidence_threshold":["high"]})"})
    EXPECT_FALSE(hrt::postprocess::check_config("DetrDetect", Json::parse(text)).empty());
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})}, {proto(DataType::Float32, {1, 2, 6})});
  for (const auto* text :
       {R"({"class_num":0})", R"({"class_num":3})", R"({"class_num":2,"confidence_threshold":[-0.1]})",
        R"({"class_num":2,"confidence_threshold":[0.1,0.2,0.3]})"})
    EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(text), engine), "");
  for (const auto& output : {proto(DataType::Float32, {1, 6, 2}), proto(DataType::Float32, {2, 2, 6}),
                             proto(DataType::Float32, {1, 0, 6}), proto(DataType::Int32, {1, 2, 6})}) {
    FakeEngine invalid({proto(DataType::Float32, {1, 3, 40, 40})}, {output});
    EXPECT_DEATH(
        hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(R"({"class_num":2})"), invalid), "");
  }
}

TEST(DetrDetectTest, AppliesStableSoftmaxIncludingBackgroundOnSeparateOutputs) {
  const auto scores = proto(DataType::Float32, {1, 2, 3});
  const auto boxes = proto(DataType::Float32, {1, 2, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 100, 100})}, {boxes, scores});
  const auto config = Json::parse(R"({"class_num":2,"output_format":"separate","score_activation":"softmax",
                                       "background_class":2,
                                       "confidence_threshold":[0.1]})");
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", config, engine);
  const auto box_buffer = hrt::test::buffer_of<float>(boxes, {0.5F, 0.5F, 0.2F, 0.4F, 0.5F, 0.5F, 0.2F, 0.4F});
  const auto score_buffer = hrt::test::buffer_of<float>(scores, {1000, 1002, 1004, 1001, 1000, 1010});
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {box_buffer, score_buffer}, hrt::test::identity(100, 100)));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(1).size(), 1U);
  EXPECT_NEAR(result.at(1)[0].score, std::exp(2.0F) / (1.0F + std::exp(2.0F) + std::exp(4.0F)), 1e-6F);
  EXPECT_FLOAT_EQ(result.at(1)[0].rect.x, 40.0F);
  EXPECT_FLOAT_EQ(result.at(1)[0].rect.y, 30.0F);
}

TEST(DetrDetectTest, AppliesSigmoidAndRemapsClassesAroundBackground) {
  const auto output = proto(DataType::Float32, {1, 1, 7});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 100, 100})}, {output});
  const auto config = Json::parse(R"({"class_num":2,"score_activation":"sigmoid","background_class":0})");
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", config, engine);
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {hrt::test::buffer_of<float>(output, {0.5F, 0.5F, 0.2F, 0.4F, 10, -2, 2})},
                               hrt::test::identity(100, 100)));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(1).size(), 1U);
  EXPECT_NEAR(result.at(1)[0].score, 1.0F / (1.0F + std::exp(-2.0F)), 1e-6F);
  EXPECT_EQ(result.at(1)[0].type, 1);
}

TEST(DetrDetectTest, PacksStridedQueries) {
  auto output = proto(DataType::Float32, {1, 2, 5});
  output.stride = {64, 64, 32};
  FakeEngine engine({proto(DataType::Float32, {1, 3, 100, 100})}, {output});
  auto decoder = hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(R"({"class_num":1})"), engine);
  auto buffer = std::make_shared<hrt::core::Buffer>(output);
  auto* data = static_cast<float*>(buffer->memory()->mutable_cpu());
  for (const auto offset : {0, 8}) {
    data[offset] = 0.5F;
    data[offset + 1] = 0.5F;
    data[offset + 2] = 0.2F;
    data[offset + 3] = 0.4F;
    data[offset + 4] = 0.9F;
  }
  hrt::postprocess::DetectResultMap result;
  ASSERT_TRUE(decoder->process(result, {buffer}, hrt::test::identity(100, 100)));
  ASSERT_EQ(result.at(0).size(), 2U);
  EXPECT_FLOAT_EQ(result.at(0)[1].rect.x, 40.0F);
}

TEST(DetrDetectTest, RejectsMismatchedSeparateQueriesAndRemovedOutputKeys) {
  for (const auto* text : {R"({"class_num":2,"score_activation":"auto"})", R"({"class_num":2,"output_format":"auto"})"})
    EXPECT_FALSE(hrt::postprocess::check_config("DetrDetect", Json::parse(text)).empty());
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 40, 40})},
                    {proto(DataType::Float32, {1, 2, 4}), proto(DataType::Float32, {1, 3, 2})});
  EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor(
                   "DetrDetect", Json::parse(R"({"class_num":2,"output_format":"separate"})"), engine),
               "");
  FakeEngine reversed({proto(DataType::Float32, {1, 3, 40, 40})},
                      {proto(DataType::Float32, {1, 2, 2}), proto(DataType::Float32, {1, 2, 4})});
  EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor(
                   "DetrDetect", Json::parse(R"({"class_num":2,"output_format":"separate"})"), reversed),
               "");
  for (const auto* key : {"boxes_output", "scores_output"}) {
    auto removed = Json::parse(R"({"class_num":2,"output_format":"separate"})");
    removed[key] = 0;
    EXPECT_FALSE(hrt::postprocess::check_config("DetrDetect", removed).empty());
  }
  FakeEngine valid({proto(DataType::Float32, {1, 3, 40, 40})},
                   {proto(DataType::Float32, {1, 2, 4}), proto(DataType::Float32, {1, 2, 2})});
  for (const auto* text : {R"({"class_num":2,"output_format":"separate","boxes_output":0})",
                           R"({"class_num":2,"output_format":"separate","scores_output":1})",
                           R"({"class_num":2,"output_format":"separate","background_class":3})",
                           R"({"class_num":2,"output_format":"separate","background_class":-2})"})
    EXPECT_DEATH(hrt::postprocess::create_detect_postprocessor("DetrDetect", Json::parse(text), valid), "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(DetrDetectTest, RunsUltralyticsModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("rtdetr_l_detect");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "Ultralytics RT-DETR-L ONNX fixture or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::DetectResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  ASSERT_TRUE(result->contains(5));
  EXPECT_FALSE(result->at(5).empty());
}
#endif

} // namespace
