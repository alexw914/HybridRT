// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/model/image_model_test.cpp
// Description: Tests the pooled image model pipeline on ONNX Runtime and TensorRT.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#ifdef HRT_TEST_TENSORRT
#include <kitzoo/log/logger.hpp>
#endif
#include <kitzoo/os/filesys.hpp>

#ifdef HRT_TEST_TENSORRT
#include <NvInfer.h>
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>

#ifdef HRT_ORT_MODEL_TEST
#include <opencv2/imgcodecs.hpp>
#endif

#ifdef HRT_TEST_ORT
#include "../engine/onnx_model.hpp"
#endif
#ifdef HRT_ORT_MODEL_TEST
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

  // Empty, oversized and mismatched requests are refused.
  EXPECT_FALSE(model->process_sync(Images{}));
  EXPECT_FALSE(model->process_sync(Images{images[0], images[1], images[0]}));
  EXPECT_FALSE(model->process_sync(images, {{0, 0, 8, 8}}));
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
  EXPECT_FALSE(model->process_sync(Images{image, image}));
}

#ifdef HRT_ORT_MODEL_TEST
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

auto expect_same(const hrt::postprocess::SegmentObject& batched, const hrt::postprocess::SegmentObject& single)
    -> void {
  expect_same(batched.box, single.box);
  EXPECT_EQ(batched.mask.rect, single.mask.rect);
  EXPECT_EQ(batched.mask.data, single.mask.data);
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
auto expect_same(const Map& batched, const Map& single) -> void {
  ASSERT_EQ(batched.size(), single.size());
  for (const auto& [cls, objects] : single) {
    ASSERT_TRUE(batched.contains(cls)) << "class " << cls;
    ASSERT_EQ(batched.at(cls).size(), objects.size()) << "class " << cls;
    for (std::size_t i = 0; i < objects.size(); ++i)
      expect_same(batched.at(cls)[i], objects[i]);
  }
}

// Runs bus.jpg and zidane.jpg through the static batch-4 copy of `name` in a
// full and a partial batch and expects the results of the batch-1 model,
// image by image.
template <typename Result>
auto expect_batch_matches_single(std::string_view name, std::string_view device = "cpu") -> void {
  const auto single_algo = hrt::test::template_algo(name);
  auto batch_algo = hrt::test::template_algo(hrt::String(name) + "_batch4");
  const auto images_dir = hrt::test::source_root() / "assets/test/images";
  hrt::Vector<cv::Mat> frames{cv::imread((images_dir / "bus.jpg").string()),
                              cv::imread((images_dir / "zidane.jpg").string())};
  if (single_algo.is_null() || batch_algo.is_null() ||
      !std::filesystem::is_regular_file(hrt::test::template_model(single_algo)) ||
      !std::filesystem::is_regular_file(hrt::test::template_model(batch_algo)) || frames[0].empty() ||
      frames[1].empty())
    GTEST_SKIP() << name << " batch-4 fixture, its batch-1 model or the test images are unavailable";

  hrt::Vector<hrt::core::Image> images;
  for (auto& frame : frames) {
    images.push_back(*hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, frame.data, hrt::core::ImageFormat::InterBgr,
                                             {frame.cols, frame.rows}, hrt::core::DataType::Uint8, frame.step));
  }
  images.push_back(images[1]);
  images.push_back(images[0]);
  auto single = hrt::model::ImageModel<Result>::create(single_algo);
  hrt::SharedPtr<hrt::core::Context> context;
  if (device != "cpu") {
    context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
    if (context == nullptr)
      GTEST_SKIP() << "CUDA device is unavailable";
    auto engine_path = hrt::test::template_model(batch_algo);
    engine_path.replace_extension("engine");
    if (!std::filesystem::is_regular_file(engine_path))
      GTEST_SKIP() << "TensorRT batch-4 fixture is unavailable: " << engine_path;
    batch_algo["config"]["engine"]["backends"] = {{{"engine_type", "trt"}, {"data", engine_path.string()}}};
    batch_algo["config"]["preprocess"]["device"] = device == "trt_cpu" ? "cpu" : "cuda";
    single = hrt::model::ImageModel<Result>::create(batch_algo, context);
  }
  const auto batched = context ? hrt::model::ImageModel<Result>::create(batch_algo, context)
                               : hrt::model::ImageModel<Result>::create(batch_algo);
  ASSERT_NE(batched, nullptr);
  ASSERT_EQ(batched->batch(), 4U);
  const auto results = batched->process_sync(images);
  ASSERT_TRUE(results);
  ASSERT_EQ(results->size(), images.size());
  for (std::size_t i = 0; i < images.size(); ++i) {
    SCOPED_TRACE(i);
    const auto expected = single->process_sync(images[i]);
    ASSERT_TRUE(expected);
    EXPECT_FALSE(expected->empty());
    expect_same((*results)[i], *expected);
  }
  images.pop_back();
  const auto partial = batched->process_sync(images);
  ASSERT_TRUE(partial);
  ASSERT_EQ(partial->size(), images.size());
  for (std::size_t i = 0; i < images.size(); ++i)
    expect_same((*partial)[i], (*results)[i]);
}

TEST_F(ImageModelTest, MatchesBatchOneOnStaticBatchModels) {
  expect_batch_matches_single<hrt::postprocess::ClassifyResult>("yolov8n_classify");
  expect_batch_matches_single<hrt::postprocess::DetectResultMap>("yolov8n_detect");
  expect_batch_matches_single<hrt::postprocess::KeypointResultMap>("yolov8n_pose");
  expect_batch_matches_single<hrt::postprocess::SegmentResultMap>("yolov8n_segment");
}

#ifdef HRT_TEST_TENSORRT
TEST_F(ImageModelTest, MatchesBatchOneOnTensorRtStaticBatchModels) {
  for (const auto device : {"trt_cpu", "trt_cuda"}) {
    SCOPED_TRACE(device);
    expect_batch_matches_single<hrt::postprocess::ClassifyResult>("yolov8n_classify", device);
    expect_batch_matches_single<hrt::postprocess::DetectResultMap>("yolov8n_detect", device);
    expect_batch_matches_single<hrt::postprocess::KeypointResultMap>("yolov8n_pose", device);
    expect_batch_matches_single<hrt::postprocess::SegmentResultMap>("yolov8n_segment", device);
  }
}

#endif
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

#ifndef HRT_TEST_ORT
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
