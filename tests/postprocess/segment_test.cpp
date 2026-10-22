// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/postprocess/segment_test.cpp
// Description: Tests YOLO instance masks (single output, raw YOLOv8 and
//              YOLO26 heads, end to end) and semantic label decoding.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>
#include <hrt/postprocess/segment.hpp>

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
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("GeneralYoloSegment", segment_config(), engine);
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
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("GeneralYoloSegment", segment_config(), engine);
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

TEST(SegmentTest, AccumulatesInt8PrototypesWithZeroPoint) {
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})},
                    {proto(DataType::Float32, {1, 7, 1}), proto(DataType::Int8, {1, 2, 4, 4})});
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("GeneralYoloSegment", segment_config(), engine);
  ASSERT_NE(postprocessor, nullptr);

  // real = 0.5 * (q - 40). Both coefficients are 1, so ignoring the zero
  // point would add 40 to every logit and fill the mask.
  hrt::Vector<std::int8_t> quantized;
  for (const auto value : prototypes())
    quantized.push_back(static_cast<std::int8_t>(value * 2.0F + 40.0F));
  auto masks = hrt::test::buffer_of(proto(DataType::Int8, {1, 2, 4, 4}), quantized);
  masks->scale = 0.5F;
  masks->zero_point = 40;
  const hrt::Vector<float> detections{8, 8, 8, 8, 0.9F, 1, 1};
  hrt::postprocess::SegmentResultMap result;
  ASSERT_TRUE(postprocessor->process(result,
                                     {hrt::test::buffer_of(proto(DataType::Float32, {1, 7, 1}), detections), masks},
                                     hrt::test::identity(16, 16)));
  expect_left_half_mask(result);
}

TEST(SegmentTest, KeepsNonzeroClassId) {
  const auto detections = proto(DataType::Float32, {1, 8, 1});
  const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {detections, masks});
  auto config = segment_config();
  config["class_num"] = 2;
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("GeneralYoloSegment", config, engine);
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
  EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("GeneralYoloSegment", config, engine), "");
  EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("Mystery", segment_config(), engine), "");
  EXPECT_EQ(hrt::postprocess::check_postprocess_config("SemanticSegment", Json::parse(R"({"class_num": 3})")).size(),
            1U);
}

// One stride-8 head of the 16x16 input (2x2 cells) whose cell (0, 0) holds
// the box (4, 4)-(12, 12) with score logit 3 and coefficients (1, 0); `bins`
// distance bins per side, the first meaning 0 cells and the last 1 cell.
auto raw_segment_head(std::size_t bins) -> hrt::Vector<float> {
  const auto channels = bins * 4 + 1 + 2;
  hrt::Vector<float> values(channels * 4, 0.0F);
  auto at = [&](std::size_t channel) -> float& { return values[channel * 4]; };
  for (std::size_t side = 0; side < 4; ++side) {
    const auto distance = side < 2 ? 0.0F : 1.0F;
    if (bins == 1) {
      at(side) = distance;
    } else {
      at(side * bins) = distance == 0.0F ? 20.0F : -20.0F;
      at(side * bins + bins - 1) = distance == 0.0F ? -20.0F : 20.0F;
    }
  }
  for (std::size_t cell = 1; cell < 4; ++cell)
    values[bins * 4 * 4 + cell] = -10.0F;
  at(bins * 4) = 3.0F;
  at(bins * 4 + 1) = 1.0F;
  return values;
}

TEST(SegmentTest, DecodesRawYoloV8AndYolo26Heads) {
  for (const auto& [type, bins] :
       hrt::Vector<std::pair<const char*, std::size_t>>{{"YoloV8Segment", 2}, {"Yolo26Segment", 1}}) {
    const auto head = proto(DataType::Float32, {1, static_cast<std::int64_t>(bins * 4 + 3), 2, 2});
    const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
    // Prototypes first: they are found by shape, not position.
    FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {masks, head});
    auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5],
                                  "iou_threshold": [0.45]})");
    auto postprocessor = hrt::postprocess::create_segment_postprocessor(type, config, engine);
    ASSERT_NE(postprocessor, nullptr) << type;
    hrt::postprocess::SegmentResultMap result;
    ASSERT_TRUE(postprocessor->process(
        result, {hrt::test::buffer_of(masks, prototypes()), hrt::test::buffer_of(head, raw_segment_head(bins))},
        hrt::test::identity(16, 16)))
        << type;
    expect_left_half_mask(result);
    EXPECT_EQ(hrt::model::task_of(type), hrt::model::Task::Segment);
  }
}

// Channel-major [channels, cells] values as cell-major [cells, channels].
auto channels_last(const hrt::Vector<float>& values, std::size_t channels) -> hrt::Vector<float> {
  const auto cells = values.size() / channels;
  hrt::Vector<float> out(values.size());
  for (std::size_t c = 0; c < channels; ++c) {
    for (std::size_t i = 0; i < cells; ++i)
      out[i * channels + c] = values[c * cells + i];
  }
  return out;
}

TEST(SegmentTest, DecodesNhwcRawHeadsAndPrototypes) {
  for (const auto& [type, bins] :
       hrt::Vector<std::pair<const char*, std::size_t>>{{"YoloV8Segment", 2}, {"Yolo26Segment", 1}}) {
    const auto channels = bins * 4 + 3;
    const auto head = proto(DataType::Float32, {1, 2, 2, static_cast<std::int64_t>(channels)});
    const auto masks = proto(DataType::Float32, {1, 4, 4, 2});
    FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})}, {masks, head}, hrt::engine::DataLayout::Nhwc);
    auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5],
                                  "iou_threshold": [0.45]})");
    auto postprocessor = hrt::postprocess::create_segment_postprocessor(type, config, engine);
    ASSERT_NE(postprocessor, nullptr) << type;
    hrt::postprocess::SegmentResultMap result;
    ASSERT_TRUE(postprocessor->process(result,
                                       {hrt::test::buffer_of(masks, channels_last(prototypes(), 2)),
                                        hrt::test::buffer_of(head, channels_last(raw_segment_head(bins), channels))},
                                       hrt::test::identity(16, 16)))
        << type;
    expect_left_half_mask(result);
  }
}

TEST(SegmentTest, ChecksExplicitRawHeadStrides) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto& type : {"YoloV8Segment", "Yolo26Segment"}) {
    FakeEngine engine({proto(DataType::Float32, {1, 3, 32, 64})},
                      {proto(DataType::Float32, {1, 7, 2, 4}), proto(DataType::Float32, {1, 2, 8, 16}),
                       proto(DataType::Float32, {1, 7, 4, 8})});
    auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
    if (hrt::String(type) == "YoloV8Segment")
      config["reg_max"] = 1;
    EXPECT_NE(hrt::postprocess::create_segment_postprocessor(type, config, engine), nullptr);
    config["strides"] = {16, 8};
    EXPECT_FALSE(hrt::postprocess::check_postprocess_config(type, config).empty());
  }
}

TEST(SegmentTest, ChecksRawYoloV8RegMax) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeEngine engine({proto(DataType::Float32, {1, 3, 16, 16})},
                    {proto(DataType::Float32, {1, 11, 2, 2}), proto(DataType::Float32, {1, 2, 4, 4})});
  auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  config["reg_max"] = 2;
  EXPECT_NE(hrt::postprocess::create_segment_postprocessor("YoloV8Segment", config, engine), nullptr);
  for (const auto bins : {0, 1, 16}) {
    config["reg_max"] = bins;
    EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("YoloV8Segment", config, engine), "");
  }
}

TEST(SegmentTest, AbortsOnRawHeadsThatDoNotFit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto input = proto(DataType::Float32, {1, 3, 16, 16});
  const auto masks = proto(DataType::Float32, {1, 2, 4, 4});
  const auto config = Json::parse(R"({"class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.45]})");
  for (const auto& outputs : hrt::Vector<hrt::Vector<hrt::core::BufferProto>>{
           {proto(DataType::Float32, {1, 7, 2, 2})}, {proto(DataType::Float32, {1, 8, 2, 2}), masks}}) {
    FakeEngine engine({input}, outputs);
    EXPECT_DEATH(hrt::postprocess::create_segment_postprocessor("Yolo26Segment", config, engine), "");
  }
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

// 35 pixels cover two 16-pixel vector steps and a scalar tail; ties go to the
// lower class and NaN never wins, as in a plain strictly-greater argmax.
TEST(SegmentTest, MatchesScalarArgmaxAcrossVectorStepsAndTail) {
  constexpr int kChannels = 4;
  constexpr int kHeight = 5;
  constexpr int kWidth = 7;
  constexpr std::size_t kPlane = kHeight * kWidth;
  const auto output = proto(DataType::Float32, {1, kChannels, kHeight, kWidth});
  FakeEngine engine({proto(DataType::Float32, {1, 3, kHeight, kWidth})}, {output});
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("SemanticSegment", Json::object(), engine);
  ASSERT_NE(postprocessor, nullptr);

  hrt::Vector<float> scores(kChannels * kPlane);
  for (std::size_t i = 0; i < scores.size(); ++i)
    scores[i] = static_cast<float>((i * 37) % 11);
  scores[1 * kPlane + 3] = std::numeric_limits<float>::quiet_NaN();
  scores[0 * kPlane + 20] = std::numeric_limits<float>::quiet_NaN();
  hrt::Vector<std::uint16_t> expected(kPlane);
  for (std::size_t i = 0; i < kPlane; ++i) {
    auto best = scores[i];
    for (std::size_t c = 1; c < kChannels; ++c) {
      if (scores[c * kPlane + i] > best) {
        best = scores[c * kPlane + i];
        expected[i] = static_cast<std::uint16_t>(c);
      }
    }
  }

  hrt::postprocess::SegmentResultMap result;
  ASSERT_TRUE(
      postprocessor->process(result, {hrt::test::buffer_of(output, scores)}, hrt::test::identity(kWidth, kHeight)));
  for (std::size_t i = 0; i < kPlane; ++i) {
    for (const auto& [cls, objects] : result)
      EXPECT_EQ(objects[0].mask.data[i], expected[i] == cls ? 1 : 0) << "pixel " << i << ", class " << cls;
  }
  EXPECT_EQ(result.at(0)[0].mask.data[20], 1) << "a NaN maximum keeps label 0";
}

TEST(SegmentTest, DecodesInt8SemanticScoresWithoutDequantizing) {
  const auto output = proto(DataType::Int8, {1, 3, 2, 2});
  FakeEngine engine({proto(DataType::Float32, {1, 3, 4, 4})}, {output});
  auto postprocessor = hrt::postprocess::create_segment_postprocessor("SemanticSegment", Json::object(), engine);
  // Quantized scores keep their order; ties go to the lower class: 0, 1 / 2, 0.
  auto buffer = hrt::test::buffer_of<std::int8_t>(output, {
                                                              90,
                                                              -5,
                                                              0,
                                                              7, // class 0
                                                              -10,
                                                              9,
                                                              0,
                                                              7, // class 1
                                                              -10,
                                                              -5,
                                                              90,
                                                              7, // class 2
                                                          });
  buffer->scale = 0.1F;
  buffer->zero_point = -100;
  hrt::postprocess::SegmentResultMap result;
  ASSERT_TRUE(postprocessor->process(result, {buffer}, hrt::test::identity(2, 2)));
  ASSERT_EQ(result.size(), 3U);
  EXPECT_EQ(result.at(0)[0].mask.data, (hrt::Vector<std::uint8_t>{1, 0, 0, 1}));
  EXPECT_EQ(result.at(1)[0].mask.data, (hrt::Vector<std::uint8_t>{0, 1, 0, 0}));
  EXPECT_EQ(result.at(2)[0].mask.data, (hrt::Vector<std::uint8_t>{0, 0, 1, 0}));
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
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
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
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
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

TEST(SegmentTest, RunsRawSegmentHeadsWhenAvailable) {
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  bool ran = false;

  struct Expected {
    const char* name;
    std::size_t people;
  };

  // The 640x480 YOLOv8 and 640x640 YOLO26 fixtures retain different people on bus.jpg.
  for (const auto& [name, people] : hrt::Array<Expected, 2>{{{"yolov8n_segment_raw", 3}, {"yolo26n_segment_raw", 4}}}) {
    const auto algo = hrt::test::template_algo(name);
    if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
      continue;
    const auto model = hrt::model::ImageModel<hrt::postprocess::SegmentResultMap>::create(algo);
    const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                              {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
    ASSERT_TRUE(image);
    const auto result = model->process_sync(*image);
    ASSERT_TRUE(result) << name;
    ASSERT_TRUE(result->contains(0)) << name;
    EXPECT_EQ(result->at(0).size(), people) << name;
    ASSERT_TRUE(result->contains(5)) << name;
    EXPECT_FALSE(result->contains(36)) << name;
    for (const auto& [cls, objects] : *result) {
      for (const auto& object : objects)
        EXPECT_FALSE(object.mask.data.empty()) << name;
    }
    ran = true;
  }
  if (!ran)
    GTEST_SKIP() << "raw-head segment ONNX fixtures or bus.jpg are unavailable";
}
#endif

} // namespace
