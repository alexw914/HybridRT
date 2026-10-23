// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/tools/hrt_model_bench_test.cpp
// Description: Tests the hrt-model-bench command line on a generated ONNX
//              Runtime model, from a manifest and from a package.
// -----------------------------------------------------------------------------

#include <hrt/hrtc/hrtc.hpp>

#include <gtest/gtest.h>

#include "../engine/onnx_model.hpp"
#include "run_tool.hpp"

namespace {

using kitzoo::json::Json;

class HrtModelBenchTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("hrt_model_bench_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    const auto model = hrt::test::unary_model("Relu", {1, 4});
    kitzoo::os::write_file(dir_ / "relu.onnx",
                           std::string_view(reinterpret_cast<const char*>(model.data()), model.size()));
    manifest_ = write_manifest("manifest.json", dir_ / "relu.onnx");
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  KZ_NODISCARD auto run(std::initializer_list<hrt::String> args) const -> hrt::test::ToolResult {
    return hrt::test::run_tool(HRT_MODEL_BENCH, args, dir_ / "log.txt");
  }

  // A manifest with one algo "relu" whose ORT engine reads `model`.
  KZ_NODISCARD auto write_manifest(const hrt::String& file, const std::filesystem::path& model) const -> hrt::String {
    auto manifest = Json::parse(R"({
      "name": "bench",
      "version": "1.0.0",
      "algos": [{"name": "relu", "config": {"engine": {"backends": [{"engine_type": "ort"}]},
                                            "postprocess": {"type": "GeneralClassify"}}}]
    })");
    manifest["algos"][0]["config"]["engine"]["backends"][0]["data"] = model.string();
    const auto path = dir_ / file;
    kitzoo::os::write_file(path, manifest.dump());
    return path.string();
  }

  std::filesystem::path dir_;
  hrt::String manifest_;
};

TEST_F(HrtModelBenchTest, ChecksArguments) {
  EXPECT_EQ(run({"-h"}).exit_code, 0);
  EXPECT_EQ(run({}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest_}).exit_code, 1);
  EXPECT_EQ(run({"-m", manifest_, "-n", "relu", "-i", "0"}).exit_code, 1);
}

TEST_F(HrtModelBenchTest, TimesManifestAndPackage) {
  const auto from_manifest = run({"-m", manifest_, "-n", "relu", "-i", "3", "-w", "1"});
  EXPECT_EQ(from_manifest.exit_code, 0);
  EXPECT_NE(from_manifest.output.find("relu/ort"), hrt::String::npos);
  EXPECT_NE(from_manifest.output.find("FPS"), hrt::String::npos);

  const auto package = hrt::hrtc::load_json(manifest_);
  ASSERT_TRUE(package);
  const auto path = dir_ / "bench.hrtc";
  ASSERT_TRUE(hrt::hrtc::save(*package, path));
  EXPECT_EQ(run({"-m", path.string(), "-n", "relu", "-e", "ort", "-i", "3", "-w", "1"}).exit_code, 0);
}

TEST_F(HrtModelBenchTest, FailsWhenNothingRuns) {
  const auto no_algo = run({"-m", manifest_, "-n", "missing"});
  EXPECT_EQ(no_algo.exit_code, 1);
  EXPECT_NE(no_algo.output.find("no algo missing"), hrt::String::npos);
  const auto no_engine = run({"-m", manifest_, "-n", "relu", "-e", "ncnn"});
  EXPECT_EQ(no_engine.exit_code, 1);
  EXPECT_NE(no_engine.output.find("no engine"), hrt::String::npos);

  kitzoo::os::write_file(dir_ / "broken.onnx", "not a model");
  const auto broken = run({"-m", write_manifest("broken.json", dir_ / "broken.onnx"), "-n", "relu", "-i", "1"});
  EXPECT_EQ(broken.exit_code, 1);
  EXPECT_NE(broken.output.find("did not load"), hrt::String::npos);
}

} // namespace
