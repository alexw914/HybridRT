// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/tools/hrt_validate_test.cpp
// Description: Tests the hrt-validate command line on a generated ONNX
//              Runtime classification model and an image directory.
// -----------------------------------------------------------------------------

#include <hrt/hrtc/hrtc.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "../engine/onnx_model.hpp"
#include "run_tool.hpp"

namespace {

using kitzoo::json::Json;

class HrtValidateTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("hrt_validate_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_ / "images");
    const auto model = hrt::test::unary_model("Identity", {1, 3, 2, 2});
    kitzoo::os::write_file(dir_ / "identity.onnx",
                           std::string_view(reinterpret_cast<const char*>(model.data()), model.size()));
    auto manifest = Json::parse(R"({
      "name": "validate",
      "version": "1.0.0",
      "algos": [{"name": "classify", "config": {"preprocess": {"std": [255, 255, 255]},
                                                "engine": {"backends": [{"engine_type": "ort"}]},
                                                "postprocess": {"type": "GeneralClassify",
                                                                "classify_postprocess_type": "softmax"}}},
                {"name": "embedding", "config": {"preprocess": {"std": [255, 255, 255]},
                                                 "engine": {"backends": [{"engine_type": "ort"}]},
                                                 "postprocess": {"type": "Embedding"}}}]
    })");
    for (auto& algo : manifest["algos"])
      algo["config"]["engine"]["backends"][0]["data"] = (dir_ / "identity.onnx").string();
    manifest_ = (dir_ / "manifest.json").string();
    kitzoo::os::write_file(manifest_, manifest.dump());
    ASSERT_TRUE(cv::imwrite((dir_ / "images" / "a.png").string(), cv::Mat(8, 6, CV_8UC3, cv::Scalar(10, 20, 30))));
    output_ = (dir_ / "results.json").string();
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  auto run(std::initializer_list<hrt::String> args) const -> hrt::test::ToolResult {
    return hrt::test::run_tool(HRT_VALIDATE, args, dir_ / "log.txt");
  }

  auto read_output() const -> Json { return kitzoo::json::Reader{std::filesystem::path(output_)}.raw(); }

  std::filesystem::path dir_;
  hrt::String manifest_;
  hrt::String output_;
};

TEST_F(HrtValidateTest, ChecksArguments) {
  EXPECT_EQ(run({"-h"}).exit_code, 0);
  EXPECT_EQ(run({}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest_, "-i", (dir_ / "images").string()}).exit_code, 1);
  const auto no_algo = run({"-m", manifest_, "-n", "missing", "-i", (dir_ / "images").string()});
  EXPECT_EQ(no_algo.exit_code, 1);
  EXPECT_NE(no_algo.output.find("no algo missing"), hrt::String::npos);
  EXPECT_EQ(run({"-m", manifest_, "-n", "classify", "-e", "ncnn", "-i", (dir_ / "images").string()}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest_, "-n", "classify", "-i", (dir_ / "missing").string()}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest_, "-n", "classify", "-i", (dir_ / "images").string(), "--engine-num", "0"}).exit_code,
            1);
}

TEST_F(HrtValidateTest, WritesClassifyResults) {
  kitzoo::os::write_file(dir_ / "images" / "notes.txt", "not an image");
  const auto result = run({"-m", manifest_, "-n", "classify", "-e", "ort", "-i", (dir_ / "images").string(), "-o",
                           output_, "-r", (dir_ / "drawn").string()});
  ASSERT_EQ(result.exit_code, 0) << result.output;
  EXPECT_FALSE(cv::imread((dir_ / "drawn" / "a.png").string()).empty());
  // Counts are logged.
  EXPECT_NE(result.output.find("total images: 1"), hrt::String::npos) << result.output;
  EXPECT_NE(result.output.find("1/1 images inferred"), hrt::String::npos) << result.output;
  const auto json = read_output();
  EXPECT_EQ(json["model"], "classify");
  EXPECT_EQ(json["task"], "classify");
  EXPECT_EQ(json["type"], "GeneralClassify");
  EXPECT_EQ(json["engine"], "ort");
  ASSERT_EQ(json["images"].size(), 1U);
  const auto& image = json["images"][0];
  EXPECT_EQ(image["file"], "a.png");
  EXPECT_EQ(image["width"], 6);
  EXPECT_EQ(image["height"], 8);
  EXPECT_TRUE(image["error"].is_null());
  ASSERT_EQ(image["results"].size(), 1U);
  const auto& item = image["results"][0];
  EXPECT_EQ(item["width"], 6.0);
  EXPECT_EQ(item["height"], 8.0);
  EXPECT_TRUE(item["mask"].is_null());
  EXPECT_TRUE(item["keypoints"].empty());
  ASSERT_EQ(item["cls"].size(), 1U);
  const auto& cls = item["cls"][0];
  EXPECT_EQ(cls["output"], 0);
  EXPECT_EQ(cls["scores"].size(), 12U);
  ASSERT_EQ(cls["topk"].size(), 5U);
  EXPECT_EQ(item["class_id"], cls["topk"][0][0]);
  EXPECT_EQ(item["score"], cls["topk"][0][1]);
}

TEST_F(HrtValidateTest, WritesEmbeddingResults) {
  const auto result = run({"-m", manifest_, "-n", "embedding", "-i", (dir_ / "images").string(), "-o", output_, "-r",
                           (dir_ / "drawn").string()});
  ASSERT_EQ(result.exit_code, 0) << result.output;
  EXPECT_FALSE(std::filesystem::exists(dir_ / "drawn" / "a.png"));
  const auto json = read_output();
  EXPECT_EQ(json["task"], "embedding");
  EXPECT_EQ(json["type"], "Embedding");
  const auto& item = json["images"][0]["results"].at(0);
  EXPECT_EQ(item["class_id"], -1);
  EXPECT_TRUE(item["cls"].empty());
  // The identity model returns the normalized planar RGB input.
  const auto& vector = item["embedding"];
  ASSERT_EQ(vector.size(), 12U);
  EXPECT_NEAR(vector[0].get<double>(), 30.0 / 255.0, 1e-6);
  EXPECT_NEAR(vector[11].get<double>(), 10.0 / 255.0, 1e-6);
}

TEST_F(HrtValidateTest, RecordsUnreadableImages) {
  kitzoo::os::write_file(dir_ / "images" / "b.png", "\x89PNG\r\n\x1a\nbroken");
  const auto package = hrt::hrtc::load_json(manifest_);
  ASSERT_TRUE(package);
  const auto path = dir_ / "validate.hrtc";
  ASSERT_TRUE(hrt::hrtc::save(*package, path));
  // Without -o the results go beside the input directory.
  const auto result = run({"-m", path.string(), "-n", "classify", "-i", (dir_ / "images").string() + "/"});
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_NE(result.output.find("1/2 images inferred"), hrt::String::npos) << result.output;
  output_ = (dir_ / "images_classify.json").string();
  const auto json = read_output();
  EXPECT_TRUE(json["engine"].is_null());
  ASSERT_EQ(json["images"].size(), 2U);
  EXPECT_TRUE(json["images"][0]["error"].is_null());
  EXPECT_EQ(json["images"][1]["file"], "b.png");
  EXPECT_EQ(json["images"][1]["error"], "cannot read the image");
  EXPECT_TRUE(json["images"][1]["results"].empty());
}

TEST_F(HrtValidateTest, WritesSemanticMasksAsRunLengths) {
  auto manifest = kitzoo::json::Reader{std::filesystem::path(manifest_)}.raw();
  manifest["algos"][0]["config"]["postprocess"] = {{"type", "SemanticSegment"}};
  kitzoo::os::write_file(manifest_, manifest.dump());
  cv::Mat halves(8, 6, CV_8UC3, cv::Scalar(200, 0, 0));
  halves.colRange(3, 6).setTo(cv::Scalar(0, 0, 200));
  ASSERT_TRUE(cv::imwrite((dir_ / "images" / "a.png").string(), halves));

  const auto drawn = dir_ / "drawn";
  const auto result =
      run({"-m", manifest_, "-n", "classify", "-i", (dir_ / "images").string(), "-o", output_, "-r", drawn.string()});
  ASSERT_EQ(result.exit_code, 0) << result.output;
  EXPECT_FALSE(cv::imread((drawn / "a.png").string()).empty());
  std::filesystem::remove_all(drawn);
  EXPECT_EQ(run({"-m", manifest_, "-n", "classify", "-i", (dir_ / "images").string(), "-o", output_}).exit_code, 0);
  EXPECT_FALSE(std::filesystem::exists(drawn));
  EXPECT_FALSE(std::filesystem::exists(dir_ / "result"));
  const auto json = read_output();
  EXPECT_EQ(json["task"], "segment");
  const auto& results = json["images"][0]["results"];
  ASSERT_EQ(results.size(), 2U);
  for (const auto& item : results) {
    EXPECT_TRUE(item["cls"].empty());
    EXPECT_EQ(item["mask"]["rect"], Json::array({0, 0, 6, 8}));
    const auto& rle = item["mask"]["rle"];
    std::size_t total = 0;
    for (const auto& count : rle)
      total += count.get<std::size_t>();
    EXPECT_EQ(total, 48U);
    EXPECT_GT(rle.size(), 2U);
  }
}

} // namespace
