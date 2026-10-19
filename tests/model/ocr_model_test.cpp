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
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#include "../engine/template_config.hpp"

namespace {

TEST(OcrModelTest, RunsPackedModelsWithoutExternalDictionary) {
#ifdef HRT_ORT_MODEL_TEST
  hrt::hrtc::Package package;
  package.name = "ocr";
  package.version = "1.0.0";
  for (const auto name : {"ppocrv6_tiny_det", "ppocrv6_tiny_rec", "ppocrv6_small_rec"}) {
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
  for (const auto name : {"ppocrv6_tiny_rec", "ppocrv6_small_rec"}) {
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

TEST(OcrModelTest, RejectsStagesOnDifferentDevices) {
#ifdef HRT_ORT_MODEL_TEST
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  auto rec = hrt::test::template_algo("ppocrv6_tiny_rec");
  if (det.is_null() || rec.is_null() || !std::filesystem::exists(hrt::test::template_model(det)) ||
      !std::filesystem::exists(hrt::test::template_model(rec)) ||
      !std::filesystem::exists(rec["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR models unavailable";
  rec["config"]["engine"]["backends"][0]["engine_type"] = "trt";
  const auto det_algo = hrt::hrtc::load_algo(det);
  const auto rec_algo = hrt::hrtc::load_algo(rec);
  ASSERT_TRUE(det_algo && rec_algo);
  // ONNX Runtime detection runs on the CPU; a TensorRT-only recognizer is another device.
  EXPECT_DEATH(hrt::model::OcrModel::create(*det_algo, *rec_algo), "");
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}

} // namespace
