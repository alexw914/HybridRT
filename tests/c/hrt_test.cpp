// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/c/hrt_test.cpp
// Description: Tests the C API: packages, model creation errors, processing
//              and flattened results.
// -----------------------------------------------------------------------------

#include <capi/hrt.h>
#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>
#include <kitzoo/os/filesys.hpp>

#include <array>
#include <atomic>
#include <filesystem>
#include <gtest/gtest.h>
#include <string_view>
#include <thread>

#ifdef HRT_ORT_MODEL_TEST
#include <opencv2/imgcodecs.hpp>
#endif

#include "../engine/onnx_model.hpp"
#include "../engine/template_config.hpp"

namespace {

// One directory per test, as CTest runs them in parallel.
auto temp_dir() -> std::filesystem::path {
  const auto dir = std::filesystem::temp_directory_path() / "hybridrt_c_api" /
                   ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::filesystem::create_directories(dir);
  return dir;
}

// An identity ORT model over {1, 3, 2, 2} with `config` as its algo config.
auto identity_algo(std::string_view name, const kitzoo::json::Json& config, std::string_view engine_type = "ort")
    -> hrt::hrtc::AlgoParam {
  return {
      .name = hrt::String{name},
      .engines = {{.engine_type = hrt::String{engine_type},
                   .config = {},
                   .proto = {},
                   .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::test::unary_model("Identity", {1, 3, 2, 2}))}},
      .config = config.dump()};
}

auto classify_config() -> kitzoo::json::Json {
  return kitzoo::json::Json::parse(R"({
    "preprocess": {"dst_image_format": "planar_rgb", "std": [255, 255, 255]},
    "postprocess": {"type": "GeneralClassify", "classify_postprocess_type": "none"}
  })");
}

// Saves `algos` as an .hrtc package and loads it through the C API.
auto load_package(const hrt::Vector<hrt::hrtc::AlgoParam>& algos) -> HrtPackage* {
  const auto path = temp_dir() / "package.hrtc";
  EXPECT_TRUE(
      hrt::hrtc::save({.name = "capi", .version = "1.0.0", .timestamp = 0, .description = {}, .algos = algos}, path));
  HrtPackage* package = nullptr;
  EXPECT_EQ(hrt_package_load(path.string().c_str(), &package), HRT_STATUS_OK) << hrt_last_error();
  return package;
}

KZ_MAYBE_UNUSED auto uniform_bgr(hrt::Vector<std::uint8_t>& pixels, int width, int height) -> HrtImage {
  pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  for (std::size_t i = 0; i < pixels.size(); i += 3) {
    pixels[i] = 10;
    pixels[i + 1] = 20;
    pixels[i + 2] = 30;
  }
  return {pixels.data(), HRT_IMAGE_FORMAT_INTER_BGR, width, height, 0};
}

auto contains(std::string_view text, std::string_view part) -> bool {
  return text.find(part) != std::string_view::npos;
}

} // namespace

TEST(HrtTest, ReportsVersionAndEngines) {
  EXPECT_NE(std::string_view(hrt_version()), "");
  EXPECT_EQ(hrt_engine_available("unknown"), 0);
  EXPECT_EQ(hrt_engine_available(nullptr), 0);
#ifdef HRT_TEST_ORT
  EXPECT_EQ(hrt_engine_available("ort"), 1);
#endif
  const auto options = hrt_model_options_default();
  EXPECT_EQ(options.preprocess, 1U);
  EXPECT_EQ(options.engine, 1U);
  EXPECT_EQ(options.postprocess, 1U);
}

TEST(HrtTest, ReportsLoadErrors) {
  HrtPackage* package = nullptr;
  EXPECT_EQ(hrt_package_load(nullptr, &package), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(hrt_package_load((temp_dir() / "missing.hrtc").string().c_str(), &package), HRT_STATUS_NOT_FOUND);
  EXPECT_EQ(package, nullptr);
  EXPECT_NE(std::string_view(hrt_last_error()), "");
  const std::array<std::uint8_t, 3> bytes{1, 2, 3};
  EXPECT_EQ(hrt_package_load_memory(bytes.data(), bytes.size(), &package), HRT_STATUS_NOT_FOUND);
  EXPECT_EQ(hrt_package_algo_count(nullptr), 0U);
  EXPECT_EQ(hrt_package_algo_name(nullptr, 0), nullptr);
}

TEST(HrtTest, ReportsModelErrorsWithoutAborting) {
  auto unknown_key = classify_config();
  unknown_key["preprocess"]["unknown_key"] = 1;
  auto* package = load_package({identity_algo("classify", classify_config()), identity_algo("bad", unknown_key),
                                identity_algo("absent", classify_config(), "unknown")});
  ASSERT_NE(package, nullptr);
  ASSERT_EQ(hrt_package_algo_count(package), 3U);
  EXPECT_EQ(std::string_view(hrt_package_algo_name(package, 1)), "bad");
  EXPECT_EQ(hrt_package_algo_name(package, 3), nullptr);

  HrtModel* model = nullptr;
  EXPECT_EQ(hrt_model_create(package, "missing", nullptr, &model), HRT_STATUS_NOT_FOUND);
  EXPECT_TRUE(contains(hrt_last_error(), "missing"));
  EXPECT_EQ(hrt_model_create(package, "bad", nullptr, &model), HRT_STATUS_INVALID_CONFIG);
  EXPECT_TRUE(contains(hrt_last_error(), "unknown_key")) << hrt_last_error();
  const HrtModelOptions zero{0, 1, 1};
  EXPECT_EQ(hrt_model_create(package, "classify", &zero, &model), HRT_STATUS_INVALID_ARGUMENT);
  const HrtModelOptions large{1, 65, 1};
  EXPECT_EQ(hrt_model_create(package, "classify", &large, &model), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(hrt_model_create(package, "absent", nullptr, &model), HRT_STATUS_UNAVAILABLE);
  EXPECT_EQ(model, nullptr);
  EXPECT_EQ(hrt_model_create(nullptr, nullptr, nullptr, &model), HRT_STATUS_INVALID_ARGUMENT);
  hrt_package_destroy(package);
}

TEST(HrtTest, LogsFromThreadsWhileTheCallbackChanges) {
  // Two callbacks in turn keep the console quiet while threads log errors and warnings.
  std::atomic<int> received{0};
  const auto count = [](HrtLogLevel /*level*/, const char* /*message*/, std::size_t /*length*/, void* user_data) {
    static_cast<std::atomic<int>*>(user_data)->fetch_add(1);
  };
  const auto other = [](HrtLogLevel /*level*/, const char* /*message*/, std::size_t /*length*/, void* /*user_data*/) {};
  hrt_set_log_callback(count, &received);
  const auto missing = (temp_dir() / "missing.hrtc").string();
  hrt::Vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&missing] {
      for (int round = 0; round < 200; ++round) {
        HrtPackage* package = nullptr;
        EXPECT_EQ(hrt_package_load(missing.c_str(), &package), HRT_STATUS_NOT_FOUND);
        EXPECT_TRUE(contains(hrt_last_error(), "missing.hrtc")) << hrt_last_error();
        EXPECT_EQ(hrt_engine_available("unknown"), 0);
      }
    });
  }
  for (int round = 0; round < 200; ++round) {
    hrt_set_log_callback(other, nullptr);
    hrt_set_log_callback(count, &received);
  }
  for (auto& thread : threads)
    thread.join();
  HrtPackage* package = nullptr;
  EXPECT_EQ(hrt_package_load(missing.c_str(), &package), HRT_STATUS_NOT_FOUND);
  hrt_set_log_callback(nullptr, nullptr);
  EXPECT_GT(received.load(), 0);
}

#ifdef HRT_TEST_ORT
TEST(HrtTest, RunsModelAndReusesResult) {
  auto* package = load_package({identity_algo("classify", classify_config())});
  ASSERT_NE(package, nullptr);
  HrtModel* model = nullptr;
  const HrtModelOptions options{2, 1, 2};
  ASSERT_EQ(hrt_model_create(package, nullptr, &options, &model), HRT_STATUS_OK) << hrt_last_error();
  // The model keeps what it needs from the package.
  hrt_package_destroy(package);
  EXPECT_EQ(hrt_model_task(model), HRT_TASK_CLASSIFY);
  EXPECT_EQ(std::string_view(hrt_model_algo_name(model)), "classify");

  HrtResult* result = nullptr;
  ASSERT_EQ(hrt_result_create(&result), HRT_STATUS_OK);
  hrt::Vector<std::uint8_t> pixels;
  const auto image = uniform_bgr(pixels, 8, 8);
  for (int round = 0; round < 2; ++round) {
    ASSERT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_OK) << hrt_last_error();
    const auto* data = hrt_result_data(result);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->task, HRT_TASK_CLASSIFY);
    ASSERT_EQ(data->count, 1U);
    EXPECT_EQ(data->boxes, nullptr);
    ASSERT_EQ(data->score_offsets[1], 12U);
    EXPECT_FLOAT_EQ(data->scores[0], 30.0F / 255.0F);
    EXPECT_FLOAT_EQ(data->scores[4], 20.0F / 255.0F);
    EXPECT_FLOAT_EQ(data->scores[11], 10.0F / 255.0F);
  }

  // A padded row pitch and a region inside the image.
  constexpr std::size_t kPitch = 32;
  hrt::Vector<std::uint8_t> padded(kPitch * 4, 0);
  for (std::size_t row = 0; row < 4; ++row) {
    auto* pixel = padded.data() + (row * kPitch);
    for (std::size_t i = 0; i < 4; ++i, pixel += 3) {
      pixel[0] = 10;
      pixel[1] = 20;
      pixel[2] = 30;
    }
  }
  const HrtImage pitched{padded.data(), HRT_IMAGE_FORMAT_INTER_BGR, 4, 4, kPitch};
  const HrtRect roi{1, 1, 2, 2};
  ASSERT_EQ(hrt_model_process(model, &pitched, &roi, result), HRT_STATUS_OK) << hrt_last_error();
  EXPECT_FLOAT_EQ(hrt_result_data(result)->scores[0], 30.0F / 255.0F);

  hrt_result_destroy(result);
  hrt_model_destroy(model);
}

TEST(HrtTest, ReturnsEmbeddings) {
  auto config = classify_config();
  config["postprocess"] = {{"type", "Embedding"}};
  auto* package = load_package({identity_algo("embedding", config)});
  ASSERT_NE(package, nullptr);
  HrtModel* model = nullptr;
  ASSERT_EQ(hrt_model_create(package, nullptr, nullptr, &model), HRT_STATUS_OK) << hrt_last_error();
  hrt_package_destroy(package);
  EXPECT_EQ(hrt_model_task(model), HRT_TASK_EMBEDDING);
  HrtResult* result = nullptr;
  ASSERT_EQ(hrt_result_create(&result), HRT_STATUS_OK);
  hrt::Vector<std::uint8_t> pixels;
  const auto image = uniform_bgr(pixels, 8, 8);
  ASSERT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_OK) << hrt_last_error();
  const auto* data = hrt_result_data(result);
  EXPECT_EQ(data->task, HRT_TASK_EMBEDDING);
  ASSERT_EQ(data->count, 1U);
  ASSERT_EQ(data->score_offsets[1], 12U);
  EXPECT_FLOAT_EQ(data->scores[0], 30.0F / 255.0F);
  hrt_result_destroy(result);
  hrt_model_destroy(model);
}

TEST(HrtTest, ReportsProcessErrorsAndForwardsLogs) {
  auto* package = load_package({identity_algo("classify", classify_config())});
  ASSERT_NE(package, nullptr);
  HrtModel* model = nullptr;
  ASSERT_EQ(hrt_model_create(package, "classify", nullptr, &model), HRT_STATUS_OK) << hrt_last_error();
  hrt_package_destroy(package);
  HrtResult* result = nullptr;
  ASSERT_EQ(hrt_result_create(&result), HRT_STATUS_OK);

  hrt::Vector<std::uint8_t> pixels;
  auto image = uniform_bgr(pixels, 8, 8);
  EXPECT_EQ(hrt_model_process(model, nullptr, nullptr, result), HRT_STATUS_INVALID_ARGUMENT);
  image.format = HRT_IMAGE_FORMAT_UNKNOWN;
  EXPECT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_INVALID_ARGUMENT);
  image.format = HRT_IMAGE_FORMAT_INTER_BGR;
  image.row_pitch = 4;
  EXPECT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(contains(hrt_last_error(), "row pitch")) << hrt_last_error();
  image.row_pitch = 0;

  hrt::Vector<hrt::String> errors;
  hrt_set_log_callback(
      [](HrtLogLevel level, const char* message, std::size_t length, void* user_data) {
        if (level >= HRT_LOG_LEVEL_ERROR)
          static_cast<hrt::Vector<hrt::String>*>(user_data)->emplace_back(message, length);
      },
      &errors);
  // The region leaves the image, which only pad_outside allows.
  const HrtRect outside{4, 4, 8, 8};
  EXPECT_EQ(hrt_model_process(model, &image, &outside, result), HRT_STATUS_PROCESS_FAILED);
  EXPECT_TRUE(contains(hrt_last_error(), "leaves")) << hrt_last_error();
  hrt_set_log_callback(nullptr, nullptr);
  ASSERT_FALSE(errors.empty());
  EXPECT_TRUE(contains(errors.front(), "leaves"));
  EXPECT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_OK) << hrt_last_error();

  hrt_result_destroy(result);
  hrt_model_destroy(model);
}
#endif

#ifdef HRT_ORT_MODEL_TEST
TEST(HrtTest, RunsSegmentModelWhenAvailable) {
  const auto algo = hrt::test::template_algo("yolo26n_e2e_segment");
  auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (!std::filesystem::is_regular_file(hrt::test::template_model(algo)) || bgr.empty())
    GTEST_SKIP() << "YOLO26n end-to-end segment ONNX fixture or bus.jpg is unavailable";
  const auto manifest = temp_dir() / "segment.json";
  kitzoo::os::write_file(manifest,
                         kitzoo::json::Json{{"name", "capi"}, {"version", "1.0.0"}, {"algos", {algo}}}.dump());
  HrtPackage* package = nullptr;
  ASSERT_EQ(hrt_package_load(manifest.string().c_str(), &package), HRT_STATUS_OK) << hrt_last_error();
  HrtModel* model = nullptr;
  ASSERT_EQ(hrt_model_create(package, "yolo26n_e2e_segment", nullptr, &model), HRT_STATUS_OK) << hrt_last_error();
  hrt_package_destroy(package);
  EXPECT_EQ(hrt_model_task(model), HRT_TASK_SEGMENT);

  HrtResult* result = nullptr;
  ASSERT_EQ(hrt_result_create(&result), HRT_STATUS_OK);
  const HrtImage image{bgr.data, HRT_IMAGE_FORMAT_INTER_BGR, bgr.cols, bgr.rows, bgr.step};
  ASSERT_EQ(hrt_model_process(model, &image, nullptr, result), HRT_STATUS_OK) << hrt_last_error();
  const auto* data = hrt_result_data(result);
  // bus.jpg: three people and the bus, as the C++ segment test reports.
  ASSERT_EQ(data->count, 4U);
  for (std::size_t i = 0; i < data->count; ++i) {
    EXPECT_EQ(data->boxes[i].class_id, i < 3 ? 0 : 5);
    const auto& mask = data->masks[i];
    EXPECT_GT(mask.rect.width, 0);
    EXPECT_GT(mask.rect.height, 0);
    EXPECT_NE(mask.data, nullptr);
  }
  hrt_result_destroy(result);
  hrt_model_destroy(model);
}
#endif

TEST(HrtTest, ReportsOcrErrorsAndClearsFailedResults) {
  HrtOcrModel* model = nullptr;
  EXPECT_EQ(hrt_ocr_model_create(nullptr, "det", "rec", &model), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(model, nullptr);
  HrtOcrResult* result = nullptr;
  ASSERT_EQ(hrt_ocr_result_create(&result), HRT_STATUS_OK);
  EXPECT_EQ(hrt_ocr_result_count(result), 0U);
  EXPECT_EQ(hrt_ocr_result_text(result, 0), nullptr);
  EXPECT_EQ(hrt_ocr_model_process(nullptr, nullptr, result), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(hrt_ocr_result_count(result), 0U);
  hrt_ocr_result_destroy(result);
  hrt_ocr_model_destroy(nullptr);
}

TEST(HrtTest, RunsPackedOcrModelWhenAvailable) {
#ifdef HRT_ORT_MODEL_TEST
  hrt::Vector<hrt::hrtc::AlgoParam> algos;
  for (const auto name : {"ppocrv6_tiny_det", "ppocrv6_tiny_rec"}) {
    const auto entry = hrt::test::template_algo(name);
    if (entry.is_null() || !std::filesystem::exists(hrt::test::template_model(entry)))
      GTEST_SKIP() << "OCR models unavailable";
    if (entry["config"]["postprocess"].contains("dictionary") &&
        !std::filesystem::exists(entry["config"]["postprocess"]["dictionary"].get<hrt::String>()))
      GTEST_SKIP() << "OCR dictionary unavailable";
    auto algo = hrt::hrtc::load_algo(entry);
    ASSERT_TRUE(algo);
    algos.push_back(std::move(*algo));
  }
  const auto bgr = cv::imread((hrt::test::source_root() / "assets/test/images/ocr_document.png").string());
  if (bgr.empty())
    GTEST_SKIP() << "OCR image unavailable";
  auto* package = load_package(algos);
  ASSERT_NE(package, nullptr);
  HrtOcrModel* model = nullptr;
  ASSERT_EQ(hrt_ocr_model_create(package, "ppocrv6_tiny_det", "ppocrv6_tiny_rec", &model), HRT_STATUS_OK);
  hrt_package_destroy(package);
  HrtOcrResult* result = nullptr;
  ASSERT_EQ(hrt_ocr_result_create(&result), HRT_STATUS_OK);
  HrtImage image{bgr.data, HRT_IMAGE_FORMAT_INTER_BGR, bgr.cols, bgr.rows, bgr.step};
  ASSERT_EQ(hrt_ocr_model_process(model, &image, result), HRT_STATUS_OK) << hrt_last_error();
  ASSERT_GT(hrt_ocr_result_count(result), 0U);
  bool found = false;
  for (std::size_t i = 0; i < hrt_ocr_result_count(result); ++i) {
    const auto* item = hrt_ocr_result_text(result, i);
    ASSERT_NE(item, nullptr);
    found = found || std::string_view(item->text, item->text_size).find("BOARDING") != std::string_view::npos;
  }
  EXPECT_TRUE(found);
  image.width = 0;
  EXPECT_EQ(hrt_ocr_model_process(model, &image, result), HRT_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(hrt_ocr_result_count(result), 0U);
  hrt_ocr_result_destroy(result);
  hrt_ocr_model_destroy(model);
#else
  GTEST_SKIP() << "Needs ONNX Runtime and OpenCV imgcodecs";
#endif
}
