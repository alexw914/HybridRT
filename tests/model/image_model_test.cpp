// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/model/image_model_test.cpp
// Description: Tests the pooled image model pipeline end to end on ONNX
//              Runtime.
// -----------------------------------------------------------------------------

#include <hrt/model/image_model.hpp>

#include <kitzoo/os/filesys.hpp>

#include <filesystem>
#include <future>
#include <gtest/gtest.h>

#include "../engine/onnx_model.hpp"

namespace {

// `type` names the postprocessor.
auto classify_algo(std::string_view preprocess, std::string_view type = "GeneralClassify") -> hrt::hrtc::AlgoParam {
  const auto config = kitzoo::json::Json{{"preprocess", kitzoo::json::Json::parse(preprocess)},
                                         {"postprocess", {{"type", type}, {"classify_postprocess_type", "none"}}}};
  return {
      .name = "classify",
      .engines = {{.engine_type = "ort",
                   .config = {},
                   .proto = {},
                   .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::test::unary_model("Identity", {1, 3, 2, 2}))}},
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
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(
      classify_algo(R"({"dst_image_format": "planar_rgb", "std": [255, 255, 255]})"), cpu_,
      {.preprocess = 3, .engine = 2, .postprocess = 3});
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->options().engine, 2U);

  hrt::Vector<std::future<bool>> runs;
  for (int t = 0; t < 4; ++t) {
    runs.push_back(std::async(std::launch::async, [&model, t] {
      hrt::Vector<std::uint8_t> pixels;
      const auto value = static_cast<std::uint8_t>(40 * t);
      for (int i = 0; i < 20; ++i) {
        const auto result = model->process_sync(uniform_bgr(pixels, value, value, value, {3 + i + t, 5 + (i % 7)}));
        if (!result || (*result)[0][5] != static_cast<float>(value) / 255.0F)
          return false;
      }
      return true;
    }));
  }
  for (auto& run : runs)
    EXPECT_TRUE(run.get());
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
  using hrt::model::check_config;
  EXPECT_TRUE(check_config(classify_algo(R"({"resize_type": "center_keep_ratio"})")).empty());
  EXPECT_EQ(check_config(classify_algo(R"({"resize_typ": "normal"})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo(R"({"resize_type": "stretch"})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo(R"({"mean": [0, 0]})")).size(), 1U);
  EXPECT_EQ(check_config(classify_algo("{}", "Mystery")).size(), 1U);
  // Stage counts are ImageModelOptions and engine settings each engine's
  // config, so neither belongs in the algo config.
  const auto with = [](const char* section, const kitzoo::json::Json& value) -> hrt::hrtc::AlgoParam {
    auto algo = classify_algo("{}");
    auto config = kitzoo::json::Json::parse(algo.config);
    config[section] = value;
    algo.config = config.dump();
    return algo;
  };
  EXPECT_EQ(check_config(with("preprocess", {{"num", 2}})).size(), 1U);
  EXPECT_EQ(check_config(with("postprocess", {{"type", "GeneralClassify"}, {"num", 2}})).size(), 1U);
  EXPECT_EQ(check_config(with("engine", {{"num", 2}})).size(), 1U);
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

} // namespace
