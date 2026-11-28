// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/model/ocr_model_test.cpp
// Description: Tests OCR package round trips, sequential real-model calls and tall lines.
// -----------------------------------------------------------------------------

#include <hrt/model/ocr_model.hpp>

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>

#ifdef HRT_ORT_MODEL_TEST
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "../engine/template_config.hpp"

namespace {

TEST(OcrModelTest, RunsPackedModelsWithoutExternalDictionary) {
#ifdef HRT_ORT_MODEL_TEST
  hrt::hrtc::Package package;
  package.name = "ocr";
  package.version = "1.0.0";
  for (const auto name : {"ppocrv6_tiny_det", "ppocrv6_tiny_rec", "ppocrv6_tiny_rec_batch8"}) {
    const auto algo = hrt::test::template_algo(name);
    if (algo.is_null() || !std::filesystem::exists(hrt::test::template_model(algo)))
      GTEST_SKIP() << "OCR models unavailable";
    const auto& post = algo["config"]["postprocess"];
    if (post.contains("dictionary") && !std::filesystem::exists(post["dictionary"].get<hrt::String>()))
      GTEST_SKIP() << "OCR dictionary unavailable";
    auto loaded = hrt::hrtc::load_algo(algo);
    ASSERT_TRUE(loaded);
    package.algos.push_back(std::move(*loaded));
  }
  const auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/ocr_document.png").string());
  if (bgr.empty())
    GTEST_SKIP() << "OCR image unavailable";
  const auto path = std::filesystem::temp_directory_path() / "hybridrt_ocr_model_test.hrtc";
  ASSERT_TRUE(hrt::hrtc::save(package, path));
  auto packed = hrt::hrtc::load(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(packed);
  package = {};
  for (const auto name : {"ppocrv6_tiny_rec", "ppocrv6_tiny_rec_batch8"}) {
    const auto* recognition = packed->find(name);
    ASSERT_NE(recognition, nullptr);
    const kitzoo::json::Reader config{recognition->config};
    EXPECT_TRUE(config.raw()["postprocess"]["dictionary"].get<hrt::String>().starts_with("string:"));
    auto model = hrt::model::OcrModel::create(*packed, "ppocrv6_tiny_det", name);
    auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                        {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
    ASSERT_TRUE(image);
    const auto result = model->process_sync(*image);
    ASSERT_TRUE(result);
    ASSERT_FALSE(result->empty());
    EXPECT_TRUE(std::any_of(result->begin(), result->end(),
                            [](const auto& item) -> bool { return item.text.find("BOARDING") != hrt::String::npos; }));
    hrt::core::Image invalid(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr);
    EXPECT_FALSE(model->process_sync(invalid));
    const auto repeated = model->process_sync(*image);
    ASSERT_TRUE(repeated);
    ASSERT_EQ(repeated->size(), result->size());
    for (std::size_t i = 0; i < result->size(); ++i)
      EXPECT_EQ((*result)[i].text, (*repeated)[i].text);
    // Turned clockwise, every line is a tall box that is rotated back on the host.
    cv::Mat turned;
    cv::rotate(bgr, turned, cv::ROTATE_90_CLOCKWISE);
    auto tall = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, turned.data, hrt::core::ImageFormat::InterBgr,
                                       {turned.cols, turned.rows}, hrt::core::DataType::Uint8, turned.step);
    ASSERT_TRUE(tall);
    const auto tall_result = model->process_sync(*tall);
    ASSERT_TRUE(tall_result);
    EXPECT_TRUE(std::any_of(tall_result->begin(), tall_result->end(),
                            [](const auto& item) -> bool { return item.text.find("BOARDING") != hrt::String::npos; }));
  }
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}

// A recognizer with a static batch reads several lines per call and must give each the text of the batch-1 one.
TEST(OcrModelTest, RecognizesLinesInBatchesLikeOneByOne) {
#ifdef HRT_ORT_MODEL_TEST
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  const auto single = hrt::test::template_algo("ppocrv6_tiny_rec");
  const auto batched = hrt::test::template_algo("ppocrv6_tiny_rec_batch8");
  const auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/ocr_document.png").string());
  for (const auto* algo : {&det, &single, &batched}) {
    if (algo->is_null() || !std::filesystem::exists(hrt::test::template_model(*algo)))
      GTEST_SKIP() << "OCR models unavailable";
  }
  if (bgr.empty() || !std::filesystem::exists(single["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR image or dictionary unavailable";
  const auto det_algo = hrt::hrtc::load_algo(det);
  const auto single_algo = hrt::hrtc::load_algo(single);
  const auto batched_algo = hrt::hrtc::load_algo(batched);
  ASSERT_TRUE(det_algo && single_algo && batched_algo);
  auto one_by_one = hrt::model::OcrModel::create(*det_algo, *single_algo);
  auto in_batches = hrt::model::OcrModel::create(*det_algo, *batched_algo);
  EXPECT_EQ(one_by_one->detection_engine_type(), hrt::engine::EngineType::Ort);
  EXPECT_EQ(one_by_one->recognition_engine_type(), hrt::engine::EngineType::Ort);
  EXPECT_FALSE(one_by_one->detection_input_size().empty());
  EXPECT_FALSE(one_by_one->recognition_input_size().empty());
  EXPECT_EQ(one_by_one->recognition_batch(), 1U);
  EXPECT_EQ(in_batches->recognition_batch(), 8U);
  EXPECT_EQ(in_batches->detection_input_size(), one_by_one->detection_input_size());
  EXPECT_EQ(in_batches->recognition_input_size(), one_by_one->recognition_input_size());
  // Upright lines read their region of the frame; turned clockwise, every line is rotated back first.
  cv::Mat turned;
  cv::rotate(bgr, turned, cv::ROTATE_90_CLOCKWISE);
  hrt::Vector<hrt::core::Image> batch_images;
  hrt::Vector<hrt::postprocess::OcrResult> single_results;
  for (const auto* frame : hrt::Array<const cv::Mat*, 2>{&bgr, &turned}) {
    auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, frame->data, hrt::core::ImageFormat::InterBgr,
                                        {frame->cols, frame->rows}, hrt::core::DataType::Uint8, frame->step);
    ASSERT_TRUE(image);
    batch_images.push_back(*image);
    const auto expected = one_by_one->process_sync(*image);
    const auto actual = in_batches->process_sync(*image);
    ASSERT_TRUE(expected && actual);
    // More lines than one batch, so the recognizer runs several batches and a partial one.
    ASSERT_GT(expected->size(), 8U);
    ASSERT_EQ(actual->size(), expected->size());
    for (std::size_t i = 0; i < expected->size(); ++i) {
      EXPECT_EQ((*actual)[i].text, (*expected)[i].text) << i;
      EXPECT_NEAR((*actual)[i].recognition_score, (*expected)[i].recognition_score, 1e-4F) << i;
      EXPECT_GT((*actual)[i].points[1].x + (*actual)[i].points[2].x, 0.0F) << i;
    }
    single_results.push_back(*actual);
  }
  const auto batched_results = in_batches->process_sync(batch_images);
  ASSERT_TRUE(batched_results);
  ASSERT_EQ(batched_results->size(), batch_images.size());
  for (std::size_t k = 0; k < batch_images.size(); ++k) {
    ASSERT_EQ((*batched_results)[k].size(), single_results[k].size()) << k;
    for (std::size_t i = 0; i < single_results[k].size(); ++i)
      EXPECT_EQ((*batched_results)[k][i].text, single_results[k][i].text) << k << ":" << i;
  }
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}

TEST(OcrModelTest, RejectsStagesOnDifferentDevices) {
#ifdef HRT_ORT_MODEL_TEST
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  auto rec = hrt::test::template_algo("ppocrv6_tiny_rec");
  if (det.is_null() || rec.is_null() || !std::filesystem::exists(hrt::test::template_model(det)) ||
      !std::filesystem::exists(hrt::test::template_model(rec)) ||
      !std::filesystem::exists(rec["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR models unavailable";
  rec["config"]["engine"]["engine_type"] = "trt";
  const auto det_algo = hrt::hrtc::load_algo(det);
  const auto rec_algo = hrt::hrtc::load_algo(rec);
  ASSERT_TRUE(det_algo && rec_algo);
  // ONNX Runtime detection runs on the CPU; a TensorRT-only recognizer is another device.
  EXPECT_EQ(hrt::model::OcrModel::create(*det_algo, *rec_algo), nullptr);
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}

TEST(OcrModelTest, WarpsTiltedTextQuads) {
#ifdef HRT_ORT_MODEL_TEST
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  const auto rec = hrt::test::template_algo("ppocrv6_tiny_rec");
  const auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/ocr_document.png").string());
  for (const auto* algo : {&det, &rec}) {
    if (algo->is_null() || !std::filesystem::exists(hrt::test::template_model(*algo)))
      GTEST_SKIP() << "OCR models unavailable";
  }
  if (bgr.empty() || !std::filesystem::exists(rec["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR image or dictionary unavailable";
  const auto det_algo = hrt::hrtc::load_algo(det);
  const auto rec_algo = hrt::hrtc::load_algo(rec);
  ASSERT_TRUE(det_algo && rec_algo);

  auto model = hrt::model::OcrModel::create(*det_algo, *rec_algo);
  ASSERT_NE(model, nullptr);

  // Pad and rotate the document by 15 degrees so DB emits tilted quads without clipping the header.
  cv::Mat padded, tilted;
  cv::copyMakeBorder(bgr, padded, 160, 160, 160, 160, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  const auto rotation = cv::getRotationMatrix2D(
      cv::Point2f(static_cast<float>(padded.cols) * 0.5F, static_cast<float>(padded.rows) * 0.5F), 15.0, 1.0);
  cv::warpAffine(padded, tilted, rotation, padded.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar(255, 255, 255));
  auto tilted_image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, tilted.data, hrt::core::ImageFormat::InterBgr,
                                             {tilted.cols, tilted.rows}, hrt::core::DataType::Uint8, tilted.step);
  ASSERT_TRUE(tilted_image);
  const auto tilted_result = model->process_sync(*tilted_image);
  ASSERT_TRUE(tilted_result);
  ASSERT_FALSE(tilted_result->empty());
  EXPECT_TRUE(std::any_of(tilted_result->begin(), tilted_result->end(), [](const auto& item) -> bool {
    return std::abs(item.points[0].y - item.points[1].y) >= 1.0F ||
           std::abs(item.points[0].x - item.points[3].x) >= 1.0F;
  }));
  EXPECT_TRUE(std::any_of(tilted_result->begin(), tilted_result->end(),
                          [](const auto& item) -> bool { return item.text.find("BOARDING") != hrt::String::npos; }));
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}

} // namespace
