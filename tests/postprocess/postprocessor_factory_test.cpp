// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/postprocessor_factory_test.cpp
// Description: Tests built-in postprocessor lookup, typed construction and
//              validation of incompatible requests.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

#include <gtest/gtest.h>

#include "fake_engine.hpp"

namespace {

using hrt::core::DataType;
using hrt::test::proto;

template <typename Result, std::size_t N>
auto expect_types(const hrt::Array<std::string_view, N>& types) -> void {
  for (auto type : types)
    EXPECT_TRUE(hrt::postprocess::supports_postprocessor<Result>(type)) << type;
  EXPECT_FALSE(hrt::postprocess::supports_postprocessor<Result>(""));
  EXPECT_FALSE(hrt::postprocess::supports_postprocessor<Result>("Unknown"));
}

TEST(PostprocessorFactoryTest, RecognizesAllBuiltInResultTypes) {
  using namespace hrt::postprocess;
  expect_types<DetectResultMap>(hrt::Array<std::string_view, 6>{"YoloV5Detect", "YoloV8Detect", "Yolo26Detect",
                                                                "GeneralDetect", "DetrDetect", "E2EDetect"});
  expect_types<RotatedResultMap>(hrt::Array<std::string_view, 5>{"YoloV5Rotated", "YoloV8Rotated", "Yolo26Rotated",
                                                                 "GeneralRotated", "E2ERotated"});
  expect_types<ClassifyResult>(hrt::Array<std::string_view, 1>{"GeneralClassify"});
  expect_types<EmbeddingResult>(hrt::Array<std::string_view, 1>{"Embedding"});
  expect_types<KeypointResultMap>(hrt::Array<std::string_view, 6>{
      "GeneralKeypoint", "GeneralYoloPose", "YoloV8Keypoint", "Yolo26Keypoint", "RtmoKeypoint", "E2EKeypoint"});
  expect_types<SegmentResultMap>(hrt::Array<std::string_view, 5>{"GeneralYoloSegment", "YoloV8Segment", "Yolo26Segment",
                                                                 "E2ESegment", "SemanticSegment"});
  expect_types<OcrResult>(hrt::Array<std::string_view, 1>{"Ocr"});
  EXPECT_FALSE(supports_postprocessor<ClassifyResult>("Embedding"));
  EXPECT_FALSE(supports_postprocessor<EmbeddingResult>("GeneralClassify"));
  EXPECT_FALSE(supports_postprocessor<OcrResult>("GeneralKeypoint"));
}

TEST(PostprocessorFactoryTest, CreatesTypedProcessors) {
  const auto output = proto(DataType::Float32, {1, 3});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {output});
  const auto config = kitzoo::json::Json::object();
  auto classify =
      hrt::postprocess::create_postprocessor<hrt::postprocess::ClassifyResult>("GeneralClassify", config, engine);
  auto embedding =
      hrt::postprocess::create_postprocessor<hrt::postprocess::EmbeddingResult>("Embedding", config, engine);
  ASSERT_NE(classify, nullptr);
  ASSERT_NE(embedding, nullptr);
  const auto buffer = hrt::test::create_buffer<float>(output, {1.0F, 2.0F, 3.0F});
  hrt::postprocess::ClassifyResult classes;
  hrt::postprocess::EmbeddingResult features;
  ASSERT_TRUE(classify->process(classes, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_TRUE(embedding->process(features, {buffer}, hrt::test::identity(8, 8)));
  ASSERT_EQ(classes.size(), 1U);
  EXPECT_EQ(classes.front(), (hrt::Vector<float>{1.0F, 2.0F, 3.0F}));
  EXPECT_EQ(features, classes.front());
}

TEST(PostprocessorFactoryTest, CreatesOcrThroughTheSameFactory) {
  const auto output = proto(DataType::Float32, {1, 2, 2});
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 48, 320})}, {output});
  const kitzoo::json::Json config{{"stage", "recognition"}, {"dictionary", "string:a\n"}, {"append_space", false}};
  EXPECT_TRUE(hrt::postprocess::check_postprocess_config("Ocr", config).empty());
  auto processor = hrt::postprocess::create_postprocessor<hrt::postprocess::OcrResult>("Ocr", config, engine);
  ASSERT_NE(processor, nullptr);
  hrt::postprocess::OcrResult result;
  const auto buffer = hrt::test::create_buffer<float>(output, {0.1F, 0.9F, 0.9F, 0.1F});
  ASSERT_TRUE(processor->process(result, {buffer}, hrt::test::identity(320, 48)));
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result.front().text, "a");
  EXPECT_NEAR(result.front().recognition_score, 0.9F, 1.0e-6F);
}

TEST(PostprocessorFactoryTest, RejectsUnknownAndIncompatibleResultTypes) {
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {proto(DataType::Float32, {1, 3})});
  const auto config = kitzoo::json::Json::object();
  EXPECT_EQ(hrt::postprocess::check_postprocess_config("Unknown", config).size(), 1U);
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::ClassifyResult>("Unknown", config, engine), "");
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::ClassifyResult>("Embedding", config, engine),
               "");
}

TEST(PostprocessorFactoryTest, RejectsInvalidConfiguration) {
  hrt::test::FakeEngine engine({proto(DataType::Float32, {1, 3, 8, 8})}, {proto(DataType::Float32, {1, 3})});
  EXPECT_DEATH(hrt::postprocess::create_postprocessor<hrt::postprocess::ClassifyResult>("GeneralClassify",
                                                                                        {{"extra", 1}}, engine),
               "");
}

} // namespace
