// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/model/image_model_test.cpp
// Description: Tests image model pooling, static batches and engine fixtures.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>
#ifdef HRT_RKNN_MODEL_TEST
#include <kitzoo/core/scopeguard.hpp>

#include <rknn_api.h>
#endif

#ifdef HRT_TEST_TENSORRT
#include <NvInfer.h>
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <type_traits>

#if defined(HRT_ORT_MODEL_TEST) || defined(HRT_RKNN_MODEL_TEST)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#ifdef HRT_TEST_ORT
#include "../engine/onnx_model.hpp"
#endif
#if defined(HRT_ORT_MODEL_TEST) || defined(HRT_RKNN_MODEL_TEST)
#include "../engine/template_config.hpp"
#endif

#ifdef HRT_TEST_ORT

namespace {

// `type` names the postprocessor; `batch` is the static batch of the identity model.
auto classify_algo(std::string_view preprocess, std::string_view type = "GeneralClassify", std::int64_t batch = 1)
    -> hrt::hrtc::AlgoParam {
  const auto config = kitzoo::json::Json{{"preprocess", kitzoo::json::Json::parse(preprocess)},
                                         {"postprocess", {{"type", type}, {"classify_postprocess_type", "none"}}}};
  return {.name = "classify",
          .engines = {{.engine_type = "ort",
                       .config = {},
                       .proto = {},
                       .data = std::make_shared<const hrt::hrtc::Bytes>(
                           hrt::test::unary_model("Identity", {batch, 3, 2, 2}))}},
          .config = config.dump()};
}

class ImageModelTest : public ::testing::Test {
protected:
  hrt::SharedPtr<hrt::core::Context> cpu_ = hrt::core::Context::create(hrt::core::DeviceType::Cpu);
};

// Uniform BGR image: every input value is its channel / 255.
auto uniform_bgr(hrt::Vector<std::uint8_t>& pixels, std::uint8_t b, std::uint8_t g, std::uint8_t r,
                 hrt::core::Size size = {8, 8}) -> hrt::core::Image {
  pixels.resize(static_cast<std::size_t>(size.width * size.height * 3));
  for (std::size_t i = 0; i < pixels.size(); i += 3) {
    pixels[i] = b;
    pixels[i + 1] = g;
    pixels[i + 2] = r;
  }
  return *hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr, size);
}

TEST_F(ImageModelTest, RunsPreprocessInferenceAndPostprocess) {
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_);
  ASSERT_NE(model, nullptr);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1U);
  ASSERT_EQ(result->front().size(), 12U);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][4], 20.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][11], 10.0F / 255.0F);
}

#if defined(__APPLE__) && defined(HRT_ORT_MODEL_TEST)
TEST_F(ImageModelTest, MetalYuvDetectionMatchesCpu) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto probe = ImagePreprocessor::create(ImagePreprocessorType::Metal);
  if (!probe)
    GTEST_SKIP() << "Metal GPU unavailable";
  auto algo = hrt::test::template_algo("yolov8n_detect");
  const auto image = cv::imread((hrt::test::source_root() / "assets/test/images/bus.jpg").string());
  if (algo.is_null() || !std::filesystem::exists(hrt::test::template_model(algo)) || image.empty())
    GTEST_SKIP() << "Detection model or image unavailable";
  cv::Mat bgr = image(cv::Rect{0, 0, image.cols & ~1, image.rows & ~1});
  cv::Mat planar;
  cv::cvtColor(bgr, planar, cv::COLOR_BGR2YUV_I420);
  const auto area = static_cast<std::size_t>(bgr.cols * bgr.rows);
  const auto* yuv = planar.ptr<std::uint8_t>();
  auto cpu_algo = algo;
  cpu_algo["config"]["preprocess"]["device"] = "cpu";
  algo["config"]["preprocess"]["device"] = "metal";
  using Model = hrt::model::ImageModel<hrt::postprocess::DetectResultMap>;
  auto cpu = Model::create(cpu_algo, cpu_);
  auto metal = Model::create(algo, cpu_);
  ASSERT_TRUE(cpu && metal);
  auto batch_algo = hrt::test::template_algo("yolov8n_detect_batch4");
  const bool has_batch = batch_algo.is_object() && std::filesystem::exists(hrt::test::template_model(batch_algo));
  hrt::UniquePtr<Model> batch_cpu, batch_metal;
  if (has_batch) {
    auto reference_algo = batch_algo;
    reference_algo["config"]["preprocess"]["device"] = "cpu";
    batch_algo["config"]["preprocess"]["device"] = "metal";
    batch_cpu = Model::create(reference_algo, cpu_);
    batch_metal = Model::create(batch_algo, cpu_);
  }
  for (const auto format : {ImageFormat::Nv12, ImageFormat::Nv21}) {
    SCOPED_TRACE(static_cast<int>(format));
    hrt::Vector<std::uint8_t> bytes(area * 3 / 2);
    std::copy_n(yuv, area, bytes.data());
    for (std::size_t i = 0; i < area / 4; ++i) {
      bytes[area + i * 2] = yuv[area + i + (format == ImageFormat::Nv21 ? area / 4 : 0)];
      bytes[area + i * 2 + 1] = yuv[area + i + (format == ImageFormat::Nv12 ? area / 4 : 0)];
    }
    auto source = Image::wrap(MemoryType::Cpu, bytes.data(), format, {bgr.cols, bgr.rows});
    ASSERT_TRUE(source);
    Image target(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {640, 640});
    ASSERT_TRUE(probe->supports(ImageOperation::Fused, target, *source));
    const auto expected = cpu->process_sync(*source);
    const auto actual = metal->process_sync(*source);
    ASSERT_TRUE(expected && actual);
    ASSERT_FALSE(expected->empty());
    ASSERT_EQ(expected->size(), actual->size());
    for (const auto& [type, boxes] : *expected) {
      const auto found = actual->find(type);
      ASSERT_NE(found, actual->end());
      ASSERT_EQ(boxes.size(), found->second.size());
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        EXPECT_GT(hrt::core::iou(boxes[i].rect, found->second[i].rect), 0.95F);
        EXPECT_NEAR(boxes[i].score, found->second[i].score, 0.02F);
      }
    }
    if (has_batch) {
      const hrt::Vector<Image> frames(5, *source);
      const auto reference = batch_cpu->process_sync(frames);
      const auto results = batch_metal->process_sync(frames);
      ASSERT_TRUE(reference && results);
      ASSERT_EQ(results->size(), frames.size());
      ASSERT_EQ(reference->size(), results->size());
      for (std::size_t n = 0; n < results->size(); ++n) {
        ASSERT_EQ((*reference)[n].size(), (*results)[n].size());
        for (const auto& [type, boxes] : (*reference)[n]) {
          const auto found = (*results)[n].find(type);
          ASSERT_NE(found, (*results)[n].end());
          ASSERT_EQ(boxes.size(), found->second.size());
          for (std::size_t i = 0; i < boxes.size(); ++i) {
            EXPECT_GT(hrt::core::iou(boxes[i].rect, found->second[i].rect), 0.95F);
            EXPECT_NEAR(boxes[i].score, found->second[i].score, 0.02F);
          }
        }
      }
    }
  }
}
#endif

TEST_F(ImageModelTest, CreatesFromAlgoJson) {
  const auto dir = std::filesystem::temp_directory_path() / "hybridrt_image_model_json";
  std::filesystem::create_directories(dir / "models");
  const auto model_bytes = hrt::test::unary_model("Identity", {1, 3, 2, 2});
  kitzoo::os::write_file(dir / "models" / "identity.onnx",
                         std::string_view(reinterpret_cast<const char*>(model_bytes.data()), model_bytes.size()));
  // One entry of a manifest's "algos"; an absolute model path is read as is.
  auto algo = kitzoo::json::Json::parse(R"({
    "name": "classify",
    "config": {
      "preprocess": {"std": [255, 255, 255]},
      "engine": {"backends": [{"engine_type": "ort", "config": {"intra_op_num_threads": 1}}]},
      "postprocess": {"type": "GeneralClassify"}
    }
  })");
  algo["config"]["engine"]["backends"][0]["data"] = (dir / "models" / "identity.onnx").string();
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(algo, cpu_);
  std::filesystem::remove_all(dir);
  ASSERT_NE(model, nullptr);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  ASSERT_EQ(result->front().size(), 12U);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][11], 10.0F / 255.0F);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  EXPECT_DEATH(Model::create(kitzoo::json::Json::array(), cpu_), "");
  auto missing = algo;
  missing["config"]["engine"]["backends"][0]["data"] = (dir / "models" / "missing.onnx").string();
  EXPECT_DEATH(Model::create(missing, cpu_), "");
}

TEST_F(ImageModelTest, ServesConcurrentCallsFromPools) {
  hrt::Vector<std::string_view> configs{R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"};
  if (hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Metal))
    configs.emplace_back(R"({"device": "metal", "std": [255, 255, 255]})");
  for (const auto config : configs) {
    SCOPED_TRACE(config);
    auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
        classify_algo(config), cpu_, {.preprocess = 3, .engine = 2, .postprocess = 3});
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->options().engine, 2U);

    hrt::Vector<std::future<bool>> runs;
    for (int t = 0; t < 4; ++t) {
      runs.push_back(std::async(std::launch::async, [&model, t]() -> bool {
        hrt::Vector<std::uint8_t> pixels;
        const auto value = static_cast<std::uint8_t>(40 * t);
        for (int i = 0; i < 20; ++i) {
          const auto result = model->process_sync(uniform_bgr(pixels, value, value, value, {3 + i + t, 5 + (i % 7)}));
          if (!result || !(std::abs((*result)[0][5] - static_cast<float>(value) / 255.0F) <= 1.0e-7F))
            return false;
        }
        return true;
      }));
    }
    for (auto& run : runs)
      EXPECT_TRUE(run.get());
  }
}

TEST_F(ImageModelTest, HandlesFrameSizeChangesAndRecoversAfterInvalidInput) {
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_);
  for (const auto size : {hrt::core::Size{8, 8}, {17, 5}, {3, 21}, {8, 8}}) {
    hrt::core::Image invalid(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr);
    EXPECT_FALSE(model->process_sync(invalid));
    hrt::Vector<std::uint8_t> pixels(static_cast<std::size_t>(size.width * size.height * 3), 64);
    const auto image =
        hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr, size);
    ASSERT_TRUE(image);
    const auto result = model->process_sync(*image);
    ASSERT_TRUE(result);
    ASSERT_EQ(result->size(), 1U);
    ASSERT_EQ(result->front().size(), 12U);
    for (const auto value : result->front())
      EXPECT_FLOAT_EQ(value, 64.0F / 255.0F);
  }
}

TEST_F(ImageModelTest, PicksTheFirstUsableEngine) {
  // TensorRT is not usable here, so the model falls back to the next engine.
  auto algo = classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})");
  auto trt = algo.engines.front();
  trt.engine_type = "trt";
  algo.engines.insert(algo.engines.begin(), trt);
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(algo);
  EXPECT_EQ(model->context()->device_type(), hrt::core::DeviceType::Cpu);

  hrt::Vector<std::uint8_t> pixels;
  const auto result = model->process_sync(uniform_bgr(pixels, 10, 20, 30));
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
}

TEST_F(ImageModelTest, ChecksConfigWithoutThrowing) {
  using hrt::model::check_algo_config;
  EXPECT_TRUE(check_algo_config(classify_algo(R"({"resize_type": "center_keep_ratio"})")).empty());
  EXPECT_EQ(check_algo_config(classify_algo(R"({"resize_typ": "normal"})")).size(), 1U);
  EXPECT_EQ(check_algo_config(classify_algo(R"({"resize_type": "stretch"})")).size(), 1U);
  EXPECT_EQ(check_algo_config(classify_algo(R"({"mean": [0, 0]})")).size(), 1U);
  EXPECT_EQ(check_algo_config(classify_algo("{}", "Mystery")).size(), 1U);
  // Stage counts are ImageModelOptions and engine settings each engine's
  // config, so neither belongs in the algo config.
  const auto with = [](const char* section, const kitzoo::json::Json& value) -> hrt::hrtc::AlgoParam {
    auto algo = classify_algo("{}");
    auto config = kitzoo::json::Json::parse(algo.config);
    config[section] = value;
    algo.config = config.dump();
    return algo;
  };
  EXPECT_EQ(check_algo_config(with("preprocess", {{"num", 2}})).size(), 1U);
  EXPECT_EQ(check_algo_config(with("postprocess", {{"type", "GeneralClassify"}, {"num", 2}})).size(), 1U);
  EXPECT_EQ(check_algo_config(with("engine", {{"num", 2}})).size(), 1U);
}

TEST_F(ImageModelTest, CreatesContextForEngineAndPreprocessor) {
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  // A CPU engine with CPU preprocessing gets a CPU context.
  const auto cpu_model = Model::create(classify_algo("{}"));
  ASSERT_NE(cpu_model, nullptr);
  EXPECT_EQ(cpu_model->context()->device_type(), hrt::core::DeviceType::Cpu);

  // A CPU engine takes the context of a device preprocessor.
  const auto cuda_algo = classify_algo(R"({"device": "cuda"})");
  if (hrt::core::Context::create(hrt::core::DeviceType::Cuda) == nullptr) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(Model::create(cuda_algo), "");
  } else {
    EXPECT_EQ(Model::create(cuda_algo)->context()->device_type(), hrt::core::DeviceType::Cuda);
  }
}

TEST_F(ImageModelTest, AbortsOnUnusableConfigurations) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  EXPECT_DEATH(Model::create(classify_algo("{}", "Mystery"), cpu_), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), cpu_, {.preprocess = 1, .engine = 0, .postprocess = 1}), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), cpu_, {.preprocess = 65, .engine = 1, .postprocess = 1}), "");
  EXPECT_DEATH(Model::create(classify_algo("{}"), hrt::SharedPtr<hrt::core::Context>{}), "");

  auto algo = classify_algo("{}");
  algo.engines[0].engine_type = "unknown";
  EXPECT_DEATH(Model::create(algo, cpu_), "");

  // Engine settings are checked when the engine loads.
  auto bad_config = classify_algo("{}");
  bad_config.engines[0].config = R"({"intra_op_threads": 2})";
  EXPECT_DEATH(Model::create(bad_config, cpu_), "");

  // An interleaved target cannot feed an NCHW input.
  EXPECT_DEATH(Model::create(classify_algo(R"({"dst_image_format": "inter_rgb"})"), cpu_), "");
}

// Second stage of a cascade: an identity classifier with a width x height
// input, so its scores are exactly the planar BGR tensor it was fed.
auto crop_algo(int width, int height, std::string_view resize_type, float pad_scale = 1.0F, int resize_short_size = -1)
    -> hrt::hrtc::AlgoParam {
  const auto preprocess = kitzoo::json::Json{{"resize_type", resize_type},
                                             {"pad_scale", pad_scale},
                                             {"resize_short_size", resize_short_size},
                                             {"dst_image_format", "planar_bgr"},
                                             {"default_color", {7, 7, 7}}};
  const auto config = kitzoo::json::Json{{"preprocess", preprocess}, {"postprocess", {{"type", "GeneralClassify"}}}};
  return {.name = "crop",
          .engines = {{.engine_type = "ort",
                       .config = {},
                       .proto = {},
                       .data = std::make_shared<const hrt::hrtc::Bytes>(
                           hrt::test::unary_model("Identity", {1, 3, height, width}))}},
          .config = config.dump()};
}

// 64x48 frame whose B channel is 4x and G channel 5y, so a fed value tells
// the source position it was sampled from; R is 100, padding 7.
auto position_frame(hrt::Vector<std::uint8_t>& pixels) -> hrt::core::Image {
  constexpr int kWidth = 64;
  constexpr int kHeight = 48;
  pixels.resize(static_cast<std::size_t>(kWidth * kHeight * 3));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      auto* px = pixels.data() + static_cast<std::size_t>((y * kWidth + x) * 3);
      px[0] = static_cast<std::uint8_t>(4 * x);
      px[1] = static_cast<std::uint8_t>(5 * y);
      px[2] = 100;
    }
  }
  return *hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr,
                                 {kWidth, kHeight});
}

// Checks every fed pixel: `whole` is the source rect the full input stands
// for, `read` the part of it actually sampled. Pixels over `read` carry the
// source position under them (within 1 px), the rest are padding. Returns
// the padding count.
auto expect_fed_crop(const hrt::Vector<float>& fed, int width, int height, const hrt::core::Rect& whole,
                     const hrt::core::Rect& read) -> int {
  const auto plane = static_cast<std::size_t>(width * height);
  const auto x_min = read.x - 0.5;
  const auto y_min = read.y - 0.5;
  const auto x_max = read.x + read.width - 0.5;
  const auto y_max = read.y + read.height - 0.5;
  int padding = 0;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const auto i = static_cast<std::size_t>(v * width + u);
      const auto sx = whole.x + ((u + 0.5) * whole.width / width) - 0.5;
      const auto sy = whole.y + ((v + 0.5) * whole.height / height) - 0.5;
      if (fed[(2 * plane) + i] == 7.0F) {
        ++padding;
        EXPECT_TRUE(sx < x_min + 2 || sx > x_max - 2 || sy < y_min + 2 || sy > y_max - 2)
            << "padding at " << u << "," << v;
        continue;
      }
      EXPECT_EQ(fed[(2 * plane) + i], 100.0F) << "at " << u << "," << v;
      EXPECT_NEAR(fed[i] / 4.0F, std::clamp(sx, x_min + 0.5, x_max - 0.5), 1.0) << "x at " << u << "," << v;
      EXPECT_NEAR(fed[plane + i] / 5.0F, std::clamp(sy, y_min + 0.5, y_max - 0.5), 1.0) << "y at " << u << "," << v;
    }
  }
  return padding;
}

TEST_F(ImageModelTest, FeedsCascadedCropsForEveryResizeType) {
  hrt::Vector<std::uint8_t> pixels;
  const auto frame = position_frame(pixels);

  struct Case {
    const char* name = "";
    int width = 16;
    int height = 16;
    const char* resize_type = "";
    float pad_scale = 1.0F;
    int resize_short_size = -1;
    hrt::core::Rect box;
    hrt::core::Rect whole;
    hrt::core::Rect read;
    bool padding = false;
  };

  // A person box {20, 10, 10, 20} and heads near the frame border.
  const hrt::Array<Case, 10> cases = {{
      {"normal", 16, 16, "normal", 1.0F, -1, {20, 10, 10, 20}, {20, 10, 10, 20}, {20, 10, 10, 20}, false},
      {"keep_ratio", 16, 16, "keep_ratio", 1.0F, -1, {20, 10, 10, 20}, {20, 10, 20, 20}, {20, 10, 10, 20}, true},
      {"letter_box", 16, 16, "letter_box", 1.0F, -1, {20, 10, 10, 20}, {15, 10, 20, 20}, {20, 10, 10, 20}, true},
      {"center_crop", 16, 16, "center_crop", 1.0F, 20, {20, 10, 10, 20}, {21, 16, 8, 8}, {21, 16, 8, 8}, false},
      {"pad_outside", 16, 16, "pad_outside", 1.0F, -1, {20, 10, 10, 20}, {15, 10, 20, 20}, {15, 10, 20, 20}, false},
      {"pad_outside, tall input",
       16,
       32,
       "pad_outside",
       1.2F,
       -1,
       {20, 10, 10, 20},
       {19, 8, 12, 24},
       {19, 8, 12, 24},
       false},
      {"pad_outside, shrunk",
       16,
       16,
       "pad_outside",
       0.5F,
       -1,
       {20, 10, 10, 20},
       {20, 15, 10, 10},
       {20, 15, 10, 10},
       false},
      {"pad_outside, head at the top",
       16,
       16,
       "pad_outside",
       1.5F,
       -1,
       {40, 0, 8, 8},
       {38, -2, 12, 12},
       {38, 0, 12, 10},
       true},
      {"pad_outside, head at the left",
       16,
       16,
       "pad_outside",
       2.0F,
       -1,
       {0, 20, 8, 8},
       {-4, 16, 16, 16},
       {0, 16, 12, 16},
       true},
      {"pad_outside, past the corner",
       16,
       16,
       "pad_outside",
       1.0F,
       -1,
       {58, 40, 12, 12},
       {58, 40, 12, 12},
       {58, 40, 6, 8},
       true},
  }};
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    const auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
        crop_algo(c.width, c.height, c.resize_type, c.pad_scale, c.resize_short_size), cpu_);
    const auto result = model->process_sync(frame, c.box);
    ASSERT_TRUE(result);
    ASSERT_EQ(result->size(), 1U);
    ASSERT_EQ(result->front().size(), static_cast<std::size_t>(3 * c.width * c.height));
    EXPECT_EQ(expect_fed_crop(result->front(), c.width, c.height, c.whole, c.read) > 0, c.padding);
  }
}

TEST_F(ImageModelTest, RejectsCropsLeavingTheFrameUnlessClipped) {
  hrt::Vector<std::uint8_t> pixels;
  const auto frame = position_frame(pixels);
  // A detection running past the bottom-right corner, as YOLO boxes may.
  const hrt::core::Rect2f detection{58.0F, 40.0F, 12.0F, 12.0F};
  const hrt::core::Rect raw{58, 40, 12, 12};
  const auto clipped = hrt::core::intersect(hrt::core::cover_pixels(detection), frame.rect());
  ASSERT_EQ(clipped, (hrt::core::Rect{58, 40, 6, 8}));
  for (const char* type : {"normal", "keep_ratio", "letter_box", "center_crop", "pad_outside"}) {
    SCOPED_TRACE(type);
    const auto model =
        hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(crop_algo(16, 16, type, 1.0F, 16), cpu_);
    // Only pad_outside clips by itself; the others fail instead of reading past the frame.
    EXPECT_EQ(model->process_sync(frame, raw).has_value(), std::string_view(type) == "pad_outside");
    EXPECT_TRUE(model->process_sync(frame, clipped));
    EXPECT_FALSE(model->process_sync(frame, {100, 100, 10, 10}));
  }

  // center_crop reads a square of min(w, h) * input / resize_short_size, which
  // can leave the frame even for a clipped box.
  const auto model =
      hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(crop_algo(16, 16, "center_crop", 1.0F, 8), cpu_);
  EXPECT_FALSE(model->process_sync(frame, clipped));
}

TEST_F(ImageModelTest, FeedsHeadsGrownWithPadScale) {
  hrt::Vector<std::uint8_t> pixels;
  const auto frame = position_frame(pixels);
  // A head detected at the top edge, grown by 1.5 and clipped to the frame.
  const auto head = hrt::core::intersect(
      hrt::core::cover_pixels(hrt::core::pad_scale(hrt::core::Rect2f{40.3F, 0.6F, 7.5F, 7.8F}, 1.5F)), frame.rect());
  ASSERT_EQ(head, (hrt::core::Rect{38, 0, 12, 11}));
  for (const char* type : {"normal", "pad_outside"}) {
    SCOPED_TRACE(type);
    const auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(crop_algo(16, 16, type), cpu_);
    const auto result = model->process_sync(frame, head);
    ASSERT_TRUE(result);
    const auto whole = std::string_view(type) == "normal" ? head : hrt::core::Rect{38, -1, 12, 12};
    const auto read = std::string_view(type) == "normal" ? head : hrt::core::Rect{38, 0, 12, 11};
    expect_fed_crop(result->front(), 16, 16, whole, read);
  }
}

TEST_F(ImageModelTest, RunsImagesAsOneStaticBatch) {
  using Images = hrt::Vector<hrt::core::Image>;
  const auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})", "GeneralClassify", 2), cpu_);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->batch(), 2U);

  hrt::Vector<std::uint8_t> first;
  hrt::Vector<std::uint8_t> second;
  const Images images{uniform_bgr(first, 10, 20, 30), uniform_bgr(second, 40, 50, 60)};
  const auto results = model->process_sync(images);
  ASSERT_TRUE(results);
  ASSERT_EQ(results->size(), 2U);
  // The identity model returns each sample's R, G and B planes.
  ASSERT_EQ((*results)[0].front().size(), 12U);
  EXPECT_FLOAT_EQ((*results)[0][0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*results)[0][0][11], 10.0F / 255.0F);
  ASSERT_EQ((*results)[1].front().size(), 12U);
  EXPECT_FLOAT_EQ((*results)[1][0][0], 60.0F / 255.0F);
  EXPECT_FLOAT_EQ((*results)[1][0][11], 40.0F / 255.0F);

  // Fewer images than the batch, the single-image overload and per-image rects.
  const auto one = model->process_sync(Images{images[1]});
  ASSERT_TRUE(one);
  ASSERT_EQ(one->size(), 1U);
  EXPECT_FLOAT_EQ((*one)[0][0][0], 60.0F / 255.0F);
  const auto single = model->process_sync(images[0]);
  ASSERT_TRUE(single);
  EXPECT_FLOAT_EQ((*single)[0][0], 30.0F / 255.0F);
  EXPECT_TRUE(model->process_sync(images, {{0, 0, 8, 8}, {0, 0, 4, 4}}));

  // More images than the batch run as consecutive batches, the last one partial.
  const auto split = model->process_sync(Images{images[0], images[1], images[1], images[0], images[1]});
  ASSERT_TRUE(split);
  ASSERT_EQ(split->size(), 5U);
  const hrt::Array<float, 5> reds{30.0F, 60.0F, 60.0F, 30.0F, 60.0F};
  for (std::size_t i = 0; i < reds.size(); ++i)
    EXPECT_FLOAT_EQ((*split)[i][0][0], reds[i] / 255.0F) << i;

  // Empty and mismatched requests are refused.
  EXPECT_FALSE(model->process_sync(Images{}));
  EXPECT_FALSE(model->process_sync(images, {{0, 0, 8, 8}}));
  EXPECT_FALSE(model->process_sync(Images{images[0], images[1], images[0]}, {{0, 0, 8, 8}, {0, 0, 8, 8}}));
}

TEST_F(ImageModelTest, KeepsConcurrentBatchCallsIndependent) {
  const auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format":"planar_rgb","std":[255,255,255]})", "GeneralClassify", 4), cpu_,
      {.preprocess = 4, .engine = 1, .postprocess = 4});
  ASSERT_NE(model, nullptr);
  hrt::Vector<std::future<bool>> calls;
  for (int worker = 0; worker < 8; ++worker) {
    calls.push_back(std::async(std::launch::async, [&, worker]() -> bool {
      hrt::Vector<hrt::Vector<std::uint8_t>> pixels(4);
      hrt::Vector<hrt::core::Image> images;
      for (std::size_t sample = 0; sample < 4; ++sample) {
        const auto value = static_cast<std::uint8_t>(worker * 20 + static_cast<int>(sample));
        images.push_back(uniform_bgr(pixels[sample], value, value, value));
      }
      for (int iteration = 0; iteration < 10; ++iteration) {
        const auto results = model->process_sync(images);
        if (!results || results->size() != images.size())
          return false;
        for (std::size_t sample = 0; sample < images.size(); ++sample) {
          const float expected = static_cast<float>(worker * 20 + static_cast<int>(sample)) / 255.0F;
          if ((*results)[sample].size() != 1 || (*results)[sample][0].size() != 12 ||
              !std::all_of((*results)[sample][0].begin(), (*results)[sample][0].end(),
                           [expected](float value) -> bool { return std::abs(value - expected) < 1e-6F; }))
            return false;
        }
        if (images.size() == 4)
          images.pop_back();
      }
      return true;
    }));
  }
  for (auto& call : calls)
    EXPECT_TRUE(call.get());
}

TEST_F(ImageModelTest, RunsTheBatchOverloadOnABatchOfOne) {
  using Images = hrt::Vector<hrt::core::Image>;
  const auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->batch(), 1U);

  hrt::Vector<std::uint8_t> pixels;
  const auto image = uniform_bgr(pixels, 10, 20, 30);
  const auto results = model->process_sync(Images{image});
  const auto single = model->process_sync(image);
  ASSERT_TRUE(results);
  ASSERT_TRUE(single);
  ASSERT_EQ(results->size(), 1U);
  EXPECT_EQ(results->front(), *single);

  // A batch of 1 runs every image on its own, as on engines without batching.
  hrt::Vector<std::uint8_t> other;
  const auto runs = model->process_sync(Images{image, uniform_bgr(other, 40, 50, 60), image});
  ASSERT_TRUE(runs);
  ASSERT_EQ(runs->size(), 3U);
  EXPECT_EQ((*runs)[0], *single);
  EXPECT_FLOAT_EQ((*runs)[1][0][0], 60.0F / 255.0F);
  EXPECT_EQ((*runs)[2], *single);

  const hrt::core::Image invalid(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr);
  EXPECT_FALSE(model->process_sync(invalid));
  EXPECT_FALSE(model->process_sync(Images{invalid}));
  const hrt::core::Rect outside{-1, 0, 8, 8};
  EXPECT_FALSE(model->process_sync(image, outside));
  EXPECT_FALSE(model->process_sync(Images{image}, {outside}));
}

} // namespace
#endif

#if defined(HRT_ORT_MODEL_TEST) || defined(HRT_RKNN_MODEL_TEST)
namespace {
#ifndef HRT_TEST_ORT
class ImageModelTest : public ::testing::Test {};
#endif

auto expect_same(const hrt::postprocess::ObjectBox& batched, const hrt::postprocess::ObjectBox& single) -> void {
  EXPECT_EQ(batched.type, single.type);
  EXPECT_NEAR(batched.score, single.score, 1e-4F);
  EXPECT_NEAR(batched.rect.x, single.rect.x, 1e-2F);
  EXPECT_NEAR(batched.rect.y, single.rect.y, 1e-2F);
  EXPECT_NEAR(batched.rect.width, single.rect.width, 1e-2F);
  EXPECT_NEAR(batched.rect.height, single.rect.height, 1e-2F);
}
auto expect_same(const hrt::postprocess::KeypointObject& batched, const hrt::postprocess::KeypointObject& single)
    -> void {
  expect_same(batched.box, single.box);
  ASSERT_EQ(batched.keypoints.size(), single.keypoints.size());
  for (std::size_t i = 0; i < batched.keypoints.size(); ++i) {
    EXPECT_NEAR(batched.keypoints[i].point.x, single.keypoints[i].point.x, 1e-2F);
    EXPECT_NEAR(batched.keypoints[i].point.y, single.keypoints[i].point.y, 1e-2F);
    EXPECT_NEAR(batched.keypoints[i].score, single.keypoints[i].score, 1e-4F);
  }
}
auto expect_same(const hrt::postprocess::SegmentObject& batched, const hrt::postprocess::SegmentObject& single,
                 double min_mask_iou) -> void {
  expect_same(batched.box, single.box);
  EXPECT_EQ(batched.mask.rect, single.mask.rect);
  ASSERT_EQ(batched.mask.data.size(), single.mask.data.size());
  std::size_t different = 0, intersection = 0, combined = 0;
  for (std::size_t i = 0; i < batched.mask.data.size(); ++i) {
    const auto lhs = batched.mask.data[i] != 0;
    const auto rhs = single.mask.data[i] != 0;
    different += lhs != rhs;
    intersection += lhs && rhs;
    combined += lhs || rhs;
  }
  if (different != 0)
    KZ_LOG_INFO("batch mask: {} / {} pixels differ, intersection={}, union={}, IoU={}", different,
                batched.mask.data.size(), intersection, combined,
                combined == 0 ? 1.0 : static_cast<double>(intersection) / static_cast<double>(combined));
  const auto iou = combined == 0 ? 1.0 : static_cast<double>(intersection) / static_cast<double>(combined);
  EXPECT_GE(iou, min_mask_iou) << different << " differing mask pixels";
}
auto expect_same(const hrt::postprocess::ClassifyResult& batched, const hrt::postprocess::ClassifyResult& single)
    -> void {
  ASSERT_EQ(batched.size(), single.size());
  for (std::size_t output = 0; output < batched.size(); ++output) {
    ASSERT_EQ(batched[output].size(), single[output].size());
    for (std::size_t i = 0; i < batched[output].size(); ++i)
      EXPECT_NEAR(batched[output][i], single[output][i], 1e-4F);
  }
}
template <typename Map>
auto expect_same(const Map& batched, const Map& single, double min_mask_iou = 1.0) -> void {
  ASSERT_EQ(batched.size(), single.size());
  for (const auto& [cls, objects] : single) {
    ASSERT_TRUE(batched.contains(cls)) << "class " << cls;
    ASSERT_EQ(batched.at(cls).size(), objects.size()) << "class " << cls;
    for (std::size_t i = 0; i < objects.size(); ++i) {
      if constexpr (std::is_same_v<typename Map::mapped_type, hrt::postprocess::SegmentResult>)
        expect_same(batched.at(cls)[i], objects[i], min_mask_iou);
      else
        expect_same(batched.at(cls)[i], objects[i]);
    }
  }
}

// Compares native batch-1 and batch-`batch` models with identical 3/2/3 pools on
// bus, zidane and up to 32 sorted COCO images, including partial and split batches.
template <typename Result>
auto expect_batch_matches_single(std::string_view name, std::string_view device = "cpu", std::size_t batch = 4)
    -> void {
  SCOPED_TRACE(hrt::String(name) + "_batch" + std::to_string(batch));
  cv::setNumThreads(-1);
  auto single_algo = hrt::test::template_algo(name);
  auto batch_algo = hrt::test::template_algo(hrt::String(name) + "_batch" + std::to_string(batch));
  if (device == "rga") {
    for (auto* algo : {&single_algo, &batch_algo}) {
      if (algo->is_null())
        continue;
      auto path = hrt::test::template_model(*algo);
      path = hrt::test::source_root() / "assets/test/rknn" / path.filename().replace_extension("rknn");
      (*algo)["config"]["engine"]["backends"] = {
          {{"engine_type", "rknn"}, {"data", path.string()}, {"config", {{"image_input", true}}}}};
      auto& preprocess = (*algo)["config"]["preprocess"];
      preprocess["device"] = "rga";
      preprocess["dst_image_format"] = "inter_rgb";
      preprocess["std"] = {1, 1, 1};
    }
  }
  const auto images_dir = hrt::test::source_root() / "assets/test/images";
  hrt::Vector<cv::Mat> frames{cv::imread((images_dir / "bus.jpg").string()),
                              cv::imread((images_dir / "zidane.jpg").string())};
  if (single_algo.is_null() || batch_algo.is_null() ||
      !std::filesystem::is_regular_file(hrt::test::template_model(single_algo)) ||
      !std::filesystem::is_regular_file(hrt::test::template_model(batch_algo)) || frames[0].empty() ||
      frames[1].empty())
    GTEST_SKIP() << name << " batch-" << batch << " fixture, its batch-1 model or the test images are unavailable";

  const auto dataset = hrt::test::source_root() / "assets/test/calib/coco128/images/train2017";
  if (std::filesystem::is_directory(dataset)) {
    hrt::Vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dataset)) {
      if (entry.path().extension() == ".jpg")
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    for (std::size_t i = 0; i < std::min(paths.size(), std::size_t{32}); ++i) {
      auto frame = cv::imread(paths[i].string());
      ASSERT_FALSE(frame.empty()) << paths[i];
      frames.push_back(std::move(frame));
    }
  }
  hrt::Vector<hrt::core::Image> images;
  for (auto& frame : frames) {
    images.push_back(*hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, frame.data, hrt::core::ImageFormat::InterBgr,
                                             {frame.cols, frame.rows}, hrt::core::DataType::Uint8, frame.step));
  }
  const hrt::model::ImageModelOptions options{3, 2, 3};
  hrt::SharedPtr<hrt::core::Context> context;
  if (device == "trt_cpu" || device == "trt_cuda") {
    context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
    if (context == nullptr)
      GTEST_SKIP() << "CUDA device is unavailable";
    context->set_current();
    for (auto* algo : {&single_algo, &batch_algo}) {
      auto engine_path = hrt::test::template_model(*algo);
      engine_path = hrt::test::source_root() / "assets/test/trt" / engine_path.filename().replace_extension("engine");
      if (!std::filesystem::is_regular_file(engine_path))
        GTEST_SKIP() << "TensorRT fixture is unavailable: " << engine_path;
      (*algo)["config"]["engine"]["backends"] = {{{"engine_type", "trt"}, {"data", engine_path.string()}}};
      (*algo)["config"]["preprocess"]["device"] = device == "trt_cpu" ? "cpu" : "cuda";
      const auto param = hrt::hrtc::load_algo(*algo);
      auto probe = hrt::engine::Engine::create(hrt::engine::engine_type_from("trt"));
      if (!param || !probe || !probe->initialize(param->engines.front()))
        GTEST_SKIP() << "TensorRT fixture does not load here: " << engine_path;
    }
  }
  const auto single = context ? hrt::model::ImageModel<Result>::create(single_algo, context, options)
                              : hrt::model::ImageModel<Result>::create(single_algo, options);
  const auto batched = context ? hrt::model::ImageModel<Result>::create(batch_algo, context, options)
                               : hrt::model::ImageModel<Result>::create(batch_algo, options);
  ASSERT_EQ(single->batch(), 1U);
  ASSERT_NE(batched, nullptr);
  ASSERT_EQ(batched->batch(), batch);
  const auto results = batched->process_sync(images);
  ASSERT_TRUE(results);
  ASSERT_EQ(results->size(), images.size());
  for (std::size_t i = 0; i < images.size(); ++i) {
    SCOPED_TRACE(i);
    const auto expected = single->process_sync(images[i]);
    ASSERT_TRUE(expected);
    // Independent TensorRT plans can differ by a pixel at the binary mask
    // threshold; within one plan (the partial/split checks below), masks must
    // still match exactly.
    if constexpr (std::is_same_v<Result, hrt::postprocess::SegmentResultMap>)
      expect_same((*results)[i], *expected, device == "cpu" ? 1.0 : 0.9999);
    else
      expect_same((*results)[i], *expected);
  }
  images.pop_back();
  const auto partial = batched->process_sync(images);
  ASSERT_TRUE(partial);
  ASSERT_EQ(partial->size(), images.size());
  for (std::size_t i = 0; i < images.size(); ++i)
    expect_same((*partial)[i], (*results)[i]);
  // Repeating the partial request also exercises splitting across engine batches.
  auto more = images;
  more.insert(more.end(), images.begin(), images.end());
  const auto split = batched->process_sync(more);
  ASSERT_TRUE(split);
  ASSERT_EQ(split->size(), more.size());
  for (std::size_t i = 0; i < more.size(); ++i)
    expect_same((*split)[i], (*results)[i % images.size()]);

  hrt::Vector<std::future<std::optional<hrt::Vector<Result>>>> calls;
  for (int i = 0; i < 4; ++i)
    calls.push_back(std::async(std::launch::async, [&] { return batched->process_sync(images); }));
  for (auto& call : calls) {
    const auto concurrent = call.get();
    ASSERT_TRUE(concurrent);
    ASSERT_EQ(concurrent->size(), images.size());
    for (std::size_t i = 0; i < images.size(); ++i)
      expect_same((*concurrent)[i], (*partial)[i]);
  }
}

#ifdef HRT_ORT_MODEL_TEST
TEST_F(ImageModelTest, MatchesBatchOneOnStaticBatchModels) {
  for (const auto batch : {std::size_t{4}, std::size_t{8}}) {
    expect_batch_matches_single<hrt::postprocess::ClassifyResult>("yolov8n_classify", "cpu", batch);
    expect_batch_matches_single<hrt::postprocess::DetectResultMap>("yolov8n_detect", "cpu", batch);
    expect_batch_matches_single<hrt::postprocess::KeypointResultMap>("yolov8n_pose", "cpu", batch);
    expect_batch_matches_single<hrt::postprocess::SegmentResultMap>("yolov8n_segment", "cpu", batch);
  }
}

#ifdef HRT_TEST_TENSORRT
TEST_F(ImageModelTest, MatchesBatchOneOnTensorRtStaticBatchModels) {
  for (const auto device : {"trt_cpu", "trt_cuda"}) {
    SCOPED_TRACE(device);
    for (const auto batch : {std::size_t{4}, std::size_t{8}}) {
      expect_batch_matches_single<hrt::postprocess::ClassifyResult>("yolov8n_classify", device, batch);
      expect_batch_matches_single<hrt::postprocess::DetectResultMap>("yolov8n_detect", device, batch);
      expect_batch_matches_single<hrt::postprocess::KeypointResultMap>("yolov8n_pose", device, batch);
      expect_batch_matches_single<hrt::postprocess::SegmentResultMap>("yolov8n_segment", device, batch);
    }
  }
}

#endif
#endif

#ifdef HRT_RKNN_MODEL_TEST
TEST_F(ImageModelTest, QueriesRknnBatchFixtures) {
  const auto directory = hrt::test::source_root() / "assets/test/rknn";
  if (!std::filesystem::is_directory(directory))
    GTEST_SKIP() << directory;
  hrt::Vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const auto path = entry.path();
    const auto stem = path.stem().string();
    if (path.extension() == ".rknn" &&
        (stem.ends_with("Int8") || (!stem.ends_with("Batch4") && !stem.ends_with("Batch8"))))
      paths.push_back(path);
  }
  if (paths.empty())
    GTEST_SKIP() << "RKNN fixtures unavailable";
  std::sort(paths.begin(), paths.end());
  for (const auto& path : paths) {
    std::error_code error;
    auto bytes = kitzoo::os::read_file(path, error);
    ASSERT_FALSE(error) << path;
    rknn_context context = 0;
    ASSERT_EQ(rknn_init(&context, bytes.data(), static_cast<std::uint32_t>(bytes.size()), 0, nullptr), RKNN_SUCC);
    KZ_SCOPE_EXIT {
      rknn_destroy(context);
    };
    rknn_sdk_version version{};
    ASSERT_EQ(rknn_query(context, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), RKNN_SUCC);
    KZ_LOG_INFO("RKNN {}: runtime={}, driver={}", path.filename().string(), version.api_version, version.drv_version);
    rknn_input_output_num count{};
    ASSERT_EQ(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), RKNN_SUCC);
    for (const auto query : {RKNN_QUERY_INPUT_ATTR, RKNN_QUERY_NATIVE_INPUT_ATTR, RKNN_QUERY_OUTPUT_ATTR}) {
      const auto size = query == RKNN_QUERY_OUTPUT_ATTR ? count.n_output : count.n_input;
      for (std::uint32_t i = 0; i < size; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        ASSERT_EQ(rknn_query(context, query, &attr, sizeof(attr)), RKNN_SUCC);
        ASSERT_GT(attr.n_dims, 0U) << path;
        if (attr.type == RKNN_TENSOR_INT8) {
          EXPECT_TRUE(std::isfinite(attr.scale)) << path;
          EXPECT_GT(attr.scale, 0.0F) << path;
        }
        KZ_LOG_INFO("RKNN query={} index={} name={} ndims={} dims=[{},{},{},{}] fmt={} type={} zp={} scale={} "
                    "w_stride={} h_stride={}",
                    static_cast<int>(query), i, attr.name, attr.n_dims, attr.dims[0], attr.dims[1], attr.dims[2],
                    attr.dims[3], static_cast<int>(attr.fmt), static_cast<int>(attr.type), attr.zp, attr.scale,
                    attr.w_stride, attr.h_stride);
      }
    }
  }
}

TEST_F(ImageModelTest, MatchesBatchOneOnRknnStaticBatchModels) {
  for (const auto batch : {std::size_t{4}, std::size_t{8}}) {
    expect_batch_matches_single<hrt::postprocess::ClassifyResult>("yolov8n_classify", "rga", batch);
    expect_batch_matches_single<hrt::postprocess::DetectResultMap>("yolov8n_detect", "rga", batch);
    expect_batch_matches_single<hrt::postprocess::KeypointResultMap>("yolov8n_pose", "rga", batch);
    expect_batch_matches_single<hrt::postprocess::SegmentResultMap>("yolov8n_segment", "rga", batch);
  }
}
#endif

} // namespace
#endif

#ifdef HRT_TEST_TENSORRT
namespace {

class TestLogger final : public nvinfer1::ILogger {
public:
  auto log(Severity severity, const char* message) noexcept -> void override {
    if (severity <= Severity::kERROR)
      KZ_LOG_ERROR("tensorrt test: {}", message);
  }
};

#if !defined(HRT_TEST_ORT) && !defined(HRT_RKNN_MODEL_TEST)
class ImageModelTest : public ::testing::Test {};
#endif

TEST_F(ImageModelTest, KeepsCudaPreprocessorScratchAliveUntilStreamCompletes) {
  auto first_context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (first_context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  first_context->set_current();

  TestLogger logger;
  hrt::UniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  ASSERT_NE(builder, nullptr);
#if HRT_TENSORRT_VERSION < 100000
  constexpr auto kNetworkFlags =
      1U << static_cast<unsigned int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
  constexpr auto kNetworkFlags = 0U;
#endif
  hrt::UniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(kNetworkFlags));
  hrt::UniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  ASSERT_NE(network, nullptr);
  ASSERT_NE(config, nullptr);
  auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 3, 1024, 1024});
  ASSERT_NE(input, nullptr);
  auto* relu = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
  ASSERT_NE(relu, nullptr);
  network->markOutput(*relu->getOutput(0));
  hrt::UniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
  ASSERT_NE(serialized, nullptr);
  const auto* begin = static_cast<const std::uint8_t*>(serialized->data());
  const auto bytes = std::make_shared<const hrt::hrtc::Bytes>(begin, begin + serialized->size());
  serialized.reset();
  config.reset();
  network.reset();
  builder.reset();
  const hrt::hrtc::AlgoParam algo{
      .name = "relu",
      .engines = {{.engine_type = "trt", .config = {}, .proto = {}, .data = bytes}},
      .config = R"({"preprocess":{"std":[255,255,255]},"postprocess":{"type":"GeneralClassify"}})"};
  using Model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>;
  auto cpu_preprocess_algo = algo;
  cpu_preprocess_algo.config =
      R"({"preprocess":{"device":"cpu","std":[255,255,255]},"postprocess":{"type":"GeneralClassify"}})";
  hrt::SharedPtr<hrt::core::Context> cpu_preprocess_context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (cpu_preprocess_context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  auto cpu_preprocess_model = Model::create(cpu_preprocess_algo, cpu_preprocess_context);
  ASSERT_NE(cpu_preprocess_model, nullptr);
  hrt::Vector<std::uint8_t> cpu_preprocess_pixels(1024 * 1024 * 3, 64);
  const auto cpu_preprocess_image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, cpu_preprocess_pixels.data(),
                                                           hrt::core::ImageFormat::InterBgr, {1024, 1024});
  ASSERT_TRUE(cpu_preprocess_image);
  const auto cpu_preprocess_result = cpu_preprocess_model->process_sync(*cpu_preprocess_image);
  ASSERT_TRUE(cpu_preprocess_result);
  EXPECT_TRUE(std::all_of(cpu_preprocess_result->front().begin(), cpu_preprocess_result->front().end(),
                          [](float value) -> bool { return std::abs(value - 64.0F / 255.0F) <= 0.00001F; }));

  for (int device_id : {0, 1}) {
    hrt::SharedPtr<hrt::core::Context> context = hrt::core::Context::create(hrt::core::DeviceType::Cuda, device_id);
    if (context == nullptr)
      continue;
    for (const auto options : {hrt::model::ImageModelOptions{1, 2, 2}, hrt::model::ImageModelOptions{3, 2, 3}}) {
      auto model = Model::create(algo, context, options);
      hrt::Vector<std::future<bool>> runs;
      for (int worker = 0; worker < 8; ++worker) {
        runs.push_back(std::async(std::launch::async, [&model, &context, worker]() -> bool {
          context->set_current();
          auto stream = context->create_stream();
          // Distinct frame values expose scratch reuse across streams.
          hrt::Vector<std::uint8_t> pixels(1024 * 1024 * 3, static_cast<std::uint8_t>(20 + worker * 20));
          const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(),
                                                    hrt::core::ImageFormat::InterBgr, {1024, 1024});
          if (!image)
            return false;
          hrt::core::Image device_image(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::InterBgr,
                                        hrt::core::DataType::Uint8, {1024, 1024});
          if (!hrt::core::Buffer::copy(*device_image.buffer(), *image->buffer(), hrt::core::CopyPolicy::DirectCopy,
                                       stream.get()))
            return false;
          stream->synchronize();
          const auto device_view = hrt::core::Image::wrap(
              hrt::core::MemoryType::Cuda, reinterpret_cast<void*>(device_image.buffer()->memory()->device()),
              hrt::core::ImageFormat::InterBgr, {1024, 1024});
          if (!device_view)
            return false;
          hrt::core::Image host_copy(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr,
                                     hrt::core::DataType::Uint8, {1024, 1024});
          if (!hrt::core::Buffer::copy(*host_copy.buffer(), *device_image.buffer(), hrt::core::CopyPolicy::DirectCopy,
                                       stream.get()))
            return false;
          stream->synchronize();
          const auto* copied_pixels = static_cast<const std::uint8_t*>(host_copy.buffer()->memory()->cpu());
          if (copied_pixels == nullptr || !std::equal(pixels.begin(), pixels.end(), copied_pixels))
            return false;
          const auto expected = static_cast<float>(20 + worker * 20) / 255.0F;
          const auto valid_result = [expected](const auto& result) -> bool {
            if (!result || result->size() != 1 || result->front().size() != 3 * 1024 * 1024)
              return false;
            return std::all_of(result->front().begin(), result->front().end(),
                               [expected](float value) -> bool { return std::abs(value - expected) <= 0.00001F; });
          };
          for (int frame = 0; frame < 20; ++frame) {
            const auto result = model->process_sync(frame % 2 == 0 ? *device_view : *image);
            if (!valid_result(result))
              return false;
          }
          for (int memory_type = 0; memory_type < 2; ++memory_type) {
            hrt::Vector<std::uint8_t> transient_pixels(1024 * 1024 * 3, static_cast<std::uint8_t>(20 + worker * 20));
            const auto transient_host = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, transient_pixels.data(),
                                                               hrt::core::ImageFormat::InterBgr, {1024, 1024});
            if (!transient_host)
              return false;
            if (memory_type == 0) {
              if (!valid_result(model->process_sync(*transient_host)))
                return false;
              continue;
            }
            hrt::core::Image transient_device(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::InterBgr,
                                              hrt::core::DataType::Uint8, {1024, 1024});
            if (!hrt::core::Buffer::copy(*transient_device.buffer(), *transient_host->buffer(),
                                         hrt::core::CopyPolicy::DirectCopy, stream.get()))
              return false;
            stream->synchronize();
            const auto transient_view = hrt::core::Image::wrap(
                hrt::core::MemoryType::Cuda, reinterpret_cast<void*>(transient_device.buffer()->memory()->device()),
                hrt::core::ImageFormat::InterBgr, {1024, 1024});
            if (!transient_view || !valid_result(model->process_sync(*transient_view)))
              return false;
          }
          return true;
        }));
      }
      for (auto& run : runs)
        EXPECT_TRUE(run.get()) << "device=" << device_id << ", preprocess=" << options.preprocess;
    }
  }
  first_context->set_current();
}

} // namespace
#endif
