// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/tools/hrt_compress_test.cpp
// Description: Tests the hrt-compress command line: packing, inspecting,
//              extracting and checking packages.
// -----------------------------------------------------------------------------

#include <kitzoo/json/reader.hpp>

#include <gtest/gtest.h>

#include "run_tool.hpp"

namespace {

using kitzoo::json::Json;

class HrtCompressTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("hrt_compress_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    kitzoo::os::write_file(dir_ / "net.bin", hrt::String("\x00\x01\x02\x03", 4));
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  KZ_NODISCARD auto run(std::initializer_list<hrt::String> args) const -> hrt::test::ToolResult {
    return hrt::test::run_tool(HRT_COMPRESS, args, dir_ / "log.txt");
  }

  // A manifest with one algo `name` whose ncnn engine reads `net.bin`.
  KZ_NODISCARD auto write_manifest(const hrt::String& file, const hrt::String& name, const Json& preprocess) const
      -> hrt::String {
    auto manifest = Json::parse(R"({
      "name": "demo",
      "version": "1.0.0",
      "algos": [{
        "config": {
          "engine": {"backends": [{"engine_type": "ncnn", "config": {"num_threads": 2}, "proto": "string:param-text"}]},
          "postprocess": {"type": "GeneralDetect", "class_num": 1, "confidence_threshold": [0.5], "iou_threshold": [0.5]}
        }
      }]
    })");
    auto& algo = manifest["algos"][0];
    algo["name"] = name;
    algo["config"]["preprocess"] = preprocess;
    algo["config"]["engine"]["backends"][0]["data"] = (dir_ / "net.bin").string();
    const auto path = dir_ / file;
    kitzoo::os::write_file(path, manifest.dump());
    return path.string();
  }

  std::filesystem::path dir_;
};

TEST_F(HrtCompressTest, PrintsUsage) {
  EXPECT_EQ(run({"-h"}).exit_code, 0);
  EXPECT_EQ(run({}).exit_code, 1);
}

TEST_F(HrtCompressTest, PacksInspectsAndExtracts) {
  const auto manifest = write_manifest("manifest.json", "detect", Json::parse(R"({"std": [255, 255, 255]})"));
  const auto package = (dir_ / "demo.hrtc").string();
  ASSERT_EQ(run({"-j", manifest, "-o", package}).exit_code, 0);

  const auto inspected = run({"-i", package});
  EXPECT_EQ(inspected.exit_code, 0);
  EXPECT_NE(inspected.output.find("Algo detect"), hrt::String::npos);
  EXPECT_NE(inspected.output.find("ncnn"), hrt::String::npos);

  const auto out = dir_ / "out";
  ASSERT_EQ(run({"-i", package, "-x", out.string()}).exit_code, 0);
  // The packed algo config keeps the model description and drops the engines.
  EXPECT_EQ(Json::parse(kitzoo::os::read_file(out / "detect.config")),
            Json::parse(R"({"preprocess": {"std": [255, 255, 255]},
                            "postprocess": {"type": "GeneralDetect", "class_num": 1,
                                            "confidence_threshold": [0.5], "iou_threshold": [0.5]}})"));
  EXPECT_EQ(Json::parse(kitzoo::os::read_file(out / "detect.engine.0.config")), Json::parse(R"({"num_threads": 2})"));
  EXPECT_EQ(kitzoo::os::read_file(out / "detect.engine.0.proto"), "param-text");
  EXPECT_EQ(kitzoo::os::read_file(out / "detect.engine.0.data"), hrt::String("\x00\x01\x02\x03", 4));
}

TEST_F(HrtCompressTest, ChecksConfigs) {
  const auto valid = write_manifest("valid.json", "detect", Json::object());
  EXPECT_EQ(run({"-c", "-j", valid}).exit_code, 0);

  const auto invalid = write_manifest("invalid.json", "detect", Json::parse(R"({"resize_type": "stretch"})"));
  const auto checked = run({"-c", "-j", invalid});
  EXPECT_EQ(checked.exit_code, 1);
  EXPECT_NE(checked.output.find("resize_type"), hrt::String::npos);
  // Packing reports the same problem but still writes the package.
  const auto package = dir_ / "invalid.hrtc";
  EXPECT_EQ(run({"-j", invalid, "-o", package.string()}).exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(package));
  EXPECT_EQ(run({"-c", "-i", package.string()}).exit_code, 1);
}

TEST_F(HrtCompressTest, ExtractsOnlySafeNames) {
  for (const hrt::String name :
       {"../escaped", "nested/escaped", "nested\\escaped", "/absolute", "C:escaped", ".", ".."}) {
    const auto package = (dir_ / "unsafe.hrtc").string();
    ASSERT_EQ(run({"-j", write_manifest("unsafe.json", name, Json::object()), "-o", package}).exit_code, 0) << name;
    const auto out = dir_ / "unsafe_out";
    EXPECT_NE(run({"-i", package, "-x", out.string()}).exit_code, 0) << name;
    EXPECT_FALSE(std::filesystem::exists(out)) << name;
  }

  // Extraction does not write through a symbolic link planted in the output.
  const auto package = (dir_ / "safe.hrtc").string();
  ASSERT_EQ(run({"-j", write_manifest("safe.json", "detect-v1.0", Json::object()), "-o", package}).exit_code, 0);
  const auto out = dir_ / "out";
  ASSERT_EQ(run({"-i", package, "-x", out.string()}).exit_code, 0);
  kitzoo::os::write_file(dir_ / "outside.config", "unchanged");
  std::filesystem::remove(out / "detect-v1.0.config");
  std::error_code ec;
  std::filesystem::create_symlink(dir_ / "outside.config", out / "detect-v1.0.config", ec);
  if (ec)
    GTEST_SKIP() << "cannot create symbolic links here: " << ec.message();
  EXPECT_NE(run({"-i", package, "-x", out.string()}).exit_code, 0);
  EXPECT_EQ(kitzoo::os::read_file(dir_ / "outside.config"), "unchanged");
}

} // namespace
