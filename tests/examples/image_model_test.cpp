// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/examples/image_model_test.cpp
// Description: Tests the image_model example on a generated classification
//              model, from a manifest and from a package.
// -----------------------------------------------------------------------------

#include <hrt/hrtc/hrtc.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "../engine/onnx_model.hpp"
#include "../tools/run_tool.hpp"

namespace {

using kitzoo::json::Json;

class ImageModelExampleTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("image_model_example_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    const auto model = hrt::test::unary_model("Identity", {1, 3, 2, 2});
    kitzoo::os::write_file(dir_ / "identity.onnx",
                           std::string_view(reinterpret_cast<const char*>(model.data()), model.size()));
    auto manifest = Json::parse(R"({
      "name": "example",
      "version": "1.0.0",
      "algos": [{"name": "classify", "config": {"preprocess": {"std": [255, 255, 255]},
                                                "engine": {"backends": [{"engine_type": "ort"}]},
                                                "postprocess": {"type": "GeneralClassify"}}}]
    })");
    manifest["algos"][0]["config"]["engine"]["backends"][0]["data"] = (dir_ / "identity.onnx").string();
    manifest_ = (dir_ / "manifest.json").string();
    kitzoo::os::write_file(manifest_, manifest.dump());
    image_ = (dir_ / "input.png").string();
    ASSERT_TRUE(cv::imwrite(image_, cv::Mat(8, 8, CV_8UC3, cv::Scalar(10, 20, 30))));
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  KZ_NODISCARD auto run(std::initializer_list<hrt::String> args) const -> hrt::test::ToolResult {
    return hrt::test::run_tool(HRT_EXAMPLE_IMAGE_MODEL, args, dir_ / "log.txt");
  }

  std::filesystem::path dir_;
  hrt::String manifest_;
  hrt::String image_;
};

TEST_F(ImageModelExampleTest, RunsFromManifestAndPackage) {
  const auto from_manifest = (dir_ / "manifest.png").string();
  const auto result = run({"-m", manifest_, "-i", image_, "-o", from_manifest});
  EXPECT_EQ(result.exit_code, 0) << result.output;
  EXPECT_FALSE(cv::imread(from_manifest).empty());

  const auto package = hrt::hrtc::load_json(manifest_);
  ASSERT_TRUE(package);
  const auto path = dir_ / "example.hrtc";
  ASSERT_TRUE(hrt::hrtc::save(*package, path));
  const auto from_package = (dir_ / "package.png").string();
  EXPECT_EQ(run({"-m", path.string(), "-a", "classify", "-i", image_, "-o", from_package}).exit_code, 0);
  EXPECT_FALSE(cv::imread(from_package).empty());
}

TEST_F(ImageModelExampleTest, ReportsBadInputs) {
  EXPECT_EQ(run({}).exit_code, 1);
  EXPECT_EQ(run({"-h"}).exit_code, 0);
  const auto no_algo = run({"-m", manifest_, "-a", "missing", "-i", image_});
  EXPECT_EQ(no_algo.exit_code, 1);
  EXPECT_NE(no_algo.output.find("no algo missing"), hrt::String::npos);
  const auto no_image = run({"-m", manifest_, "-i", (dir_ / "missing.png").string()});
  EXPECT_EQ(no_image.exit_code, 1);
  EXPECT_NE(no_image.output.find("cannot read"), hrt::String::npos);
}

} // namespace
