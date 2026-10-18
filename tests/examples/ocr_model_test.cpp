// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/examples/ocr_model_test.cpp
// Description: Tests OCR example arguments and JSON output from template models.
// -----------------------------------------------------------------------------

#include <kitzoo/json/reader.hpp>

#include <gtest/gtest.h>

#include "../engine/template_config.hpp"
#include "../tools/run_tool.hpp"

namespace {

TEST(OcrModelExampleTest, PrintsUsageAndRejectsMissingArguments) {
  const auto log = std::filesystem::temp_directory_path() / "hybridrt_ocr_example_usage.log";
  EXPECT_EQ(hrt::test::run_tool(HRT_EXAMPLE_OCR_MODEL, {"--help"}, log).exit_code, 0);
  EXPECT_EQ(hrt::test::run_tool(HRT_EXAMPLE_OCR_MODEL, {}, log).exit_code, 1);
}

TEST(OcrModelExampleTest, WritesOcrJson) {
  const auto det = hrt::test::template_algo("ppocrv6_tiny_det");
  const auto rec = hrt::test::template_algo("ppocrv6_tiny_rec");
  const auto image = hrt::test::source_root() / "assets/test/images/ocr_document.png";
  if (det.is_null() || rec.is_null() || !std::filesystem::exists(image) ||
      !std::filesystem::exists(hrt::test::template_model(det)) ||
      !std::filesystem::exists(hrt::test::template_model(rec)) ||
      !std::filesystem::exists(rec["config"]["postprocess"]["dictionary"].get<hrt::String>()))
    GTEST_SKIP() << "OCR fixtures unavailable";
  const auto dir = std::filesystem::temp_directory_path() / "hybridrt_ocr_example";
  std::filesystem::create_directories(dir);
  const kitzoo::json::Json manifest{{"name", "ocr"}, {"version", "1.0.0"}, {"algos", {det, rec}}};
  kitzoo::os::write_file(dir / "ocr.json", manifest.dump());
  const auto result = hrt::test::run_tool(
      HRT_EXAMPLE_OCR_MODEL,
      {"-m", (dir / "ocr.json").string(), "-i", image.string(), "-o", (dir / "result.json").string()}, dir / "run.log");
  ASSERT_EQ(result.exit_code, 0) << result.output;
  const kitzoo::json::Reader output(dir / "result.json");
  ASSERT_TRUE(output.is_parse_success());
  ASSERT_TRUE(output.raw().is_array());
  ASSERT_FALSE(output.raw().empty());
  EXPECT_EQ(output.raw()[0]["rect"].size(), 4U);
  EXPECT_TRUE(output.raw()[0]["text"].is_string());
}

} // namespace
