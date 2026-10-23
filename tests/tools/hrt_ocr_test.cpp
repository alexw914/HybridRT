// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/tools/hrt_ocr_test.cpp
// Description: Tests the hrt-ocr command line arguments and, with the template
//              PP-OCR models, its JSON results on an image directory.
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "../engine/onnx_model.hpp"
#include "../engine/template_config.hpp"
#include "run_tool.hpp"

namespace {

using kitzoo::json::Json;

class HrtOcrTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("hrt_ocr_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_ / "images");
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  KZ_NODISCARD auto run(std::initializer_list<hrt::String> args) const -> hrt::test::ToolResult {
    return hrt::test::run_tool(HRT_OCR, args, dir_ / "log.txt");
  }

  KZ_NODISCARD auto write_manifest(const Json& algos) const -> hrt::String {
    const auto path = dir_ / "manifest.json";
    kitzoo::os::write_file(path, Json{{"name", "ocr"}, {"version", "1.0.0"}, {"algos", algos}}.dump());
    return path.string();
  }

  std::filesystem::path dir_;
};

TEST_F(HrtOcrTest, ChecksArguments) {
  const auto model = hrt::test::unary_model("Identity", {1, 3, 2, 2});
  kitzoo::os::write_file(dir_ / "identity.onnx",
                         std::string_view(reinterpret_cast<const char*>(model.data()), model.size()));
  auto classify = Json::parse(R"({"name": "classify", "config": {
    "engine": {"backends": [{"engine_type": "ort"}]},
    "postprocess": {"type": "GeneralClassify"}}})");
  classify["config"]["engine"]["backends"][0]["data"] = (dir_ / "identity.onnx").string();
  const auto manifest = write_manifest(Json::array({classify}));
  const auto images = (dir_ / "images").string();

  EXPECT_EQ(run({"-h"}).exit_code, 0);
  EXPECT_EQ(run({}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest}).exit_code, 1);
  const auto no_algo = run({"-m", manifest, "-i", images});
  EXPECT_EQ(no_algo.exit_code, 1);
  EXPECT_NE(no_algo.output.find("no algo ppocrv6_tiny_det"), hrt::String::npos) << no_algo.output;
  const auto no_engine = run({"-m", manifest, "-d", "classify", "-r", "classify", "-e", "ncnn", "-i", images});
  EXPECT_EQ(no_engine.exit_code, 1);
  EXPECT_NE(no_engine.output.find("has no ncnn engine"), hrt::String::npos) << no_engine.output;
  // Both stages must be OCR algos of the matching role.
  EXPECT_EQ(run({"-m", manifest, "-d", "classify", "-r", "classify", "-i", images}).exit_code, 1);
}

TEST_F(HrtOcrTest, WritesOcrResults) {
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  const auto rec = hrt::test::template_algo("ppocrv6_tiny_rec");
  const auto image = hrt::test::source_root() / "assets/test/images/ocr_document.jpg";
  if (det.is_null() || rec.is_null() || !std::filesystem::exists(image) ||
      !std::filesystem::exists(hrt::test::template_model(det)) ||
      !std::filesystem::exists(hrt::test::template_model(rec)) ||
      !std::filesystem::exists(rec["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR fixtures unavailable";
  std::filesystem::copy_file(image, dir_ / "images" / "a.jpg");
  kitzoo::os::write_file(dir_ / "images" / "b.png", "\x89PNG\r\n\x1a\nbroken");
  const auto manifest = write_manifest(Json::array({det, rec}));
  const auto output = (dir_ / "ocr.json").string();

  const auto result = run({"-m", manifest, "-e", "ort", "-i", (dir_ / "images").string(), "-o", output});
  EXPECT_EQ(result.exit_code, 1) << result.output;
  EXPECT_NE(result.output.find("1/2 images inferred"), hrt::String::npos) << result.output;
  const auto json = kitzoo::json::Reader{std::filesystem::path(output)}.raw();
  EXPECT_EQ(json["detection"], "ppocrv6_tiny_det");
  EXPECT_EQ(json["recognition"], "ppocrv6_tiny_rec");
  EXPECT_EQ(json["task"], "ocr");
  EXPECT_EQ(json["engine"], "ort");
  ASSERT_EQ(json["images"].size(), 2U);
  const auto& a = json["images"][0];
  EXPECT_EQ(a["file"], "a.jpg");
  EXPECT_TRUE(a["error"].is_null());
  EXPECT_GT(a["width"].get<int>(), 0);
  ASSERT_FALSE(a["results"].empty());
  bool boarding = false;
  for (const auto& item : a["results"]) {
    EXPECT_TRUE(item["text"].is_string());
    EXPECT_GT(item["width"].get<double>(), 0.0);
    boarding = boarding || item["text"].get<hrt::String>().find("BOARDING") != hrt::String::npos;
  }
  EXPECT_TRUE(boarding);
  const auto& b = json["images"][1];
  EXPECT_EQ(b["error"], "cannot read the image");
  EXPECT_TRUE(b["results"].empty());
}

} // namespace
