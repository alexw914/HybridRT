// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/segment_test.cpp
// Description: Tests instance mask, end-to-end instance mask and semantic
//              label decoding.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/segment.hpp>

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

// 16x16 input, prototypes [1, 2, 4, 4]: channel 0 is +10 on the left half
// and -10 on the right, channel 1 is zero.
auto prototypes() -> hrt::Vector<float> {
  hrt::Vector<float> values(2 * 4 * 4, 0.0F);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x)
      values[static_cast<std::size_t>(y * 4 + x)] = x < 2 ? 10.0F : -10.0F;
  }
  return values;
}

auto segment_engine(DataType detections) -> FakeEngine {
  return FakeEngine({proto(DataType::Float32, {1, 3, 16, 16})},
                    {proto(detections, {1, 7, 1}), proto(DataType::Float32, {1, 2, 4, 4})});
}

auto segment_config() -> Json {
  return Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
}

auto expect_left_half_mask(const hrt::postprocess::SegmentResultMap& result) -> void {
  ASSERT_EQ(result.at(0).size(), 1U);
  const auto& object = result.at(0)[0];
  // Box center (8, 8), size 8 x 8 in a 16x16 source.
  EXPECT_FLOAT_EQ(object.box.rect.x, 4.0F);
  EXPECT_FLOAT_EQ(object.box.rect.width, 8.0F);
  EXPECT_EQ(object.mask.rect, (hrt::core::Rect{4, 4, 8, 8}));
  ASSERT_EQ(object.mask.data.size(), 64U);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x)
      EXPECT_EQ(object.mask.data[static_cast<std::size_t>(y * 8 + x)], x < 4 ? 1 : 0) << "at " << x << "," << y;
  }
  EXPECT_FALSE(result.contains(99));
}

TEST(SegmentTest, DecodesYoloInstanceMask) {
  auto engine = segment_engine(DataType::Float32);
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("InstanceSegment", segment_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // cx, cy, w, h, score, coefficient 0, coefficient 1.
  const hrt::Vector<float> detections{8, 8, 8, 8, 0.9F, 1, 0};
  hrt::postprocess::SegmentResultMap result;
  result[99].emplace_back();
  ASSERT_TRUE(postprocessor->process(result,
                                     {hrt::test::buffer_of(proto(DataType::Float32, {1, 7, 1}), detections),
                                      hrt::test::buffer_of(proto(DataType::Float32, {1, 2, 4, 4}), prototypes())},
                                     hrt::test::identity(16, 16)));
  expect_left_half_mask(result);
}

TEST(SegmentTest, DecodesInt8Detections) {
  auto engine = segment_engine(DataType::Int8);
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("InstanceSegment", segment_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // real = 0.1 * q: 8, 8, 8, 8, 0.9, 1, 0.
  auto detections = hrt::test::buffer_of<std::int8_t>(proto(DataType::Int8, {1, 7, 1}), {80, 80, 80, 80, 9, 10, 0});
  detections->scale = 0.1F;
  hrt::postprocess::SegmentResultMap result;
  ASSERT_TRUE(postprocessor->process(
      result, {detections, hrt::test::buffer_of(proto(DataType::Float32, {1, 2, 4, 4}), prototypes())},
      hrt::test::identity(16, 16)));
  expect_left_half_mask(result);
}

TEST(SegmentTest, KeepsNonzeroClassId) {
  const auto detections = proto(DataType::Float32, {1, 8, 1});
  const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {detections, masks});
  auto config = segment_config();
  config["class_num"] = 2;
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("InstanceSegment", config, engine);
  hrt::postprocess::SegmentResultMap result;
  ASSERT_TRUE(postprocessor->process(result,
                                     {hrt::test::buffer_of<float>(detections, {8, 8, 8, 8, 0.1F, 0.9F, 1, 0}),
                                      hrt::test::buffer_of(masks, prototypes())},
                                     hrt::test::identity(16, 16)));
  ASSERT_EQ(result.size(), 1U);
  ASSERT_EQ(result.at(1).size(), 1U);
  EXPECT_EQ(result.at(1)[0].box.type, 1);
}

TEST(SegmentTest, AbortsOnMismatchedOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto engine = segment_engine(DataType::Float32);
  auto config = segment_config();
  config["class_num"] = 2;
  EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("InstanceSegment", config, engine), "");
  EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("Mystery", segment_config(), engine), "");
  EXPECT_EQ(hrt::postprocess::check_config("SemanticSegment", Json::parse(R"({"class_num": 3})")).size(), 1U);
}

TEST(SegmentTest, DecodesSemanticLabels) {
  const auto output = proto(DataType::Float32, {1, 3, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 4, 4})}, {output});
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("SemanticSegment", Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // Winning class per 2x2 output cell: 0, 1 / 2, 1.
  const hrt::Vector<float> scores{
      9, 0, 0, 0, // class 0
      0, 9, 0, 9, // class 1
      0, 0, 9, 0, // class 2
  };
  hrt::postprocess::SegmentResultMap result;
  result[0].emplace_back();
  ASSERT_TRUE(postprocessor->process(result, {hrt::test::buffer_of(output, scores)}, hrt::test::identity(4, 4)));
  const hrt::Vector<std::uint16_t> expected{0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1};
  ASSERT_EQ(result.size(), 3U);
  for (std::int64_t cls = 0; cls < 3; ++cls) {
    ASSERT_EQ(result.at(cls).size(), 1U);
    const auto& object = result.at(cls)[0];
    EXPECT_EQ(object.box.type, cls);
    EXPECT_FLOAT_EQ(object.box.rect.x, 0.0F);
    EXPECT_FLOAT_EQ(object.box.rect.y, 0.0F);
    EXPECT_FLOAT_EQ(object.box.rect.width, 4.0F);
    EXPECT_FLOAT_EQ(object.box.rect.height, 4.0F);
    EXPECT_EQ(object.mask.rect, (hrt::core::Rect{0, 0, 4, 4}));
    ASSERT_EQ(object.mask.data.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(object.mask.data[i], expected[i] == cls ? 1 : 0);
  }
}

TEST(SegmentTest, PreservesInputRegionAndIntegerClassIds) {
  const auto output = proto(DataType::Int32, {1, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 2, 2})}, {output});
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("SemanticSegment", Json::object(), engine);
  hrt::postprocess::SegmentResultMap result;
  const hrt::preprocess::ImageTransform transform{{0, 0, 2, 2}, {10, 20, 2, 2}, {2, 2}};
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::buffer_of<std::int32_t>(output, {0, 1, 256, 65535})}, transform));
  ASSERT_EQ(result.size(), 4U);
  const hrt::Vector<std::int64_t> classes{0, 1, 256, 65535};
  for (std::size_t i = 0; i < classes.size(); ++i) {
    ASSERT_EQ(result.at(classes[i]).size(), 1U);
    const auto& object = result.at(classes[i])[0];
    EXPECT_EQ(object.box.type, classes[i]);
    EXPECT_FLOAT_EQ(object.box.rect.x, 10.0F);
    EXPECT_FLOAT_EQ(object.box.rect.y, 20.0F);
    EXPECT_FLOAT_EQ(object.box.rect.width, 2.0F);
    EXPECT_FLOAT_EQ(object.box.rect.height, 2.0F);
    EXPECT_EQ(object.mask.rect, transform.src_roi);
    hrt::Vector<std::uint8_t> expected(4, 0);
    expected[i] = 1;
    EXPECT_EQ(object.mask.data, expected);
  }
}

TEST(SegmentTest, DecodesEndToEndRowsWithoutNms) {
  const auto detections = proto(DataType::Float32, {1, 3, 8});
  const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {masks, detections});
  const auto config = Json::parse(R"({"class_num": 2, "confidence_threshold": [0.5]})");
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("E2ESegment", config, engine);
  ASSERT_NE(postprocessor, nullptr);
  // x1, y1, x2, y2, score, class, coefficient 0, coefficient 1: the left-half
  // mask of class 0, a class 1 row below its threshold and a padding row.
  const hrt::Vector<float> rows{
      4, 4, 12, 12, 0.9F, 0, 1, 0, 4, 4, 12, 12, 0.4F, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  hrt::postprocess::SegmentResultMap result;
  result[99].emplace_back();
  ASSERT_TRUE(postprocessor->process(
      result, {hrt::test::buffer_of(masks, prototypes()), hrt::test::buffer_of(detections, rows)},
      hrt::test::identity(16, 16)));
  ASSERT_EQ(result.size(), 1U);
  expect_left_half_mask(result);
  EXPECT_EQ(hrt::model::task_of("E2ESegment"), hrt::model::Task::Segment);
}

TEST(SegmentTest, AbortsOnMismatchedEndToEndOutputs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 16, 16});
  const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
  const auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5]})");
  for (const auto& outputs :
       hrt::Vector<hrt::Vector<hrt::core::BufferProto>>{{proto(DataType::Float32, {1, 3, 8})},
                                                        {proto(DataType::Float32, {1, 3, 9}), masks},
                                                        {proto(DataType::Float32, {1, 7, 1}), masks}}) {
    FakeEngine engine({input}, outputs);
    EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("E2ESegment", config, engine), "");
  }
  FakeEngine valid({input}, {proto(DataType::Float32, {1, 3, 8}), masks});
  EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor(
                   "E2ESegment", Json::parse(R"({"class_num": 1, "iou_threshold": [0.5]})"), valid),
               "");
}

#ifdef HRT_ORT_MODEL_TEST
TEST(SegmentTest, RunsUltralyticsEndToEndExportWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_e2e_segment");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLO26n end-to-end segment ONNX fixture or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::SegmentResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // bus.jpg: three people and the bus, as Ultralytics reports for the same export.
  ASSERT_TRUE(result->contains(0));
  EXPECT_EQ(result->at(0).size(), 3U);
  ASSERT_TRUE(result->contains(5));
  EXPECT_EQ(result->at(5).size(), 1U);
  for (const auto& [cls, objects] : *result) {
    for (const auto& object : objects)
      EXPECT_FALSE(object.mask.data.empty());
  }
}

TEST(SegmentTest, RunsSemanticModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("lraspp_voc_segment");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/yolo/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "LRASPP-MobileNetV3 ONNX fixture or bus.jpg is unavailable";
  const auto model = hrt::model::ImageModel<hrt::postprocess::SegmentResultMap>::create(algo);
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  // bus.jpg: VOC background, bus and person.
  hrt::Vector<std::int64_t> classes;
  for (const auto& [cls, objects] : *result)
    classes.push_back(cls);
  EXPECT_EQ(classes, (hrt::Vector<std::int64_t>{0, 6, 15}));
}
#endif

} // namespace
