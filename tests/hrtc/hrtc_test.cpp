// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/hrtc/hrtc_test.cpp
// Description: Tests hrtc manifest building, saving and loading.
// -----------------------------------------------------------------------------

#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/os/filesys.hpp>

#include <gtest/gtest.h>

namespace {

class HrtcTest : public ::testing::Test {
protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           (hrt::String("hrtc_test_") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(dir_ / "models");
    kitzoo::os::write_file(dir_ / "models" / "net.param", "param-bytes");
    kitzoo::os::write_file(dir_ / "models" / "net.bin", hrt::String("\x00\x01\x02\x03", 4));
    kitzoo::os::write_file(dir_ / "manifest.json", R"({
      "name": "demo",
      "version": "1.2.3",
      "description": "test package",
      "algos": [{
        "name": "detect_ncnn",
        "config": {
          "engine": {"engine_type": "ncnn", "proto": "models/net.param", "data": "models/net.bin"},
          "postprocess": {"type": "YoloV8Detect"}
        }
      }, {
        "name": "detect_ort",
        "config": {
          "engine": {"engine_type": "ort", "data": "models/net.bin"},
          "postprocess": {"type": "YoloV8Detect"}
        }
      }]
    })");
    // Relative manifest paths resolve against the working directory.
    previous_ = std::filesystem::current_path();
    std::filesystem::current_path(dir_);
  }

  void TearDown() override {
    std::filesystem::current_path(previous_);
    std::filesystem::remove_all(dir_);
  }

  std::filesystem::path dir_;
  std::filesystem::path previous_;
};

auto text(const hrt::SharedPtr<const hrt::hrtc::Bytes>& bytes) -> hrt::String {
  return bytes == nullptr ? hrt::String{} : hrt::String(bytes->begin(), bytes->end());
}

TEST_F(HrtcTest, ManifestRoundTrip) {
  auto package = hrt::hrtc::load_json(dir_ / "manifest.json");
  ASSERT_TRUE(package);
  ASSERT_TRUE(hrt::hrtc::save(*package, dir_ / "demo.hrtc"));

  auto loaded = hrt::hrtc::load(dir_ / "demo.hrtc");
  ASSERT_TRUE(loaded);
  EXPECT_EQ(loaded->name, "demo");
  EXPECT_EQ(loaded->version, "1.2.3");
  EXPECT_EQ(loaded->description, "test package");
  EXPECT_EQ(loaded->timestamp, package->timestamp);

  const auto* ncnn = loaded->find("detect_ncnn");
  ASSERT_NE(ncnn, nullptr);
  // The "engine" section is packed into `engine` and dropped from the stored config.
  EXPECT_EQ(ncnn->config, R"({"postprocess":{"type":"YoloV8Detect"}})");
  EXPECT_EQ(ncnn->engine.engine_type, "ncnn");
  EXPECT_EQ(text(ncnn->engine.proto), "param-bytes");
  ASSERT_NE(ncnn->engine.data, nullptr);
  EXPECT_EQ(*ncnn->engine.data, (hrt::hrtc::Bytes{0, 1, 2, 3}));

  const auto* ort = loaded->find("detect_ort");
  ASSERT_NE(ort, nullptr);
  EXPECT_EQ(ort->engine.engine_type, "ort");
  EXPECT_EQ(ort->engine.proto, nullptr);
  ASSERT_NE(ort->engine.data, nullptr);
  EXPECT_EQ(*ort->engine.data, (hrt::hrtc::Bytes{0, 1, 2, 3}));
  EXPECT_EQ(loaded->find("missing"), nullptr);

  // Model files outlive the package that loaded them.
  const auto weights = ort->engine.data;
  loaded.reset();
  EXPECT_EQ(*weights, (hrt::hrtc::Bytes{0, 1, 2, 3}));
}

TEST_F(HrtcTest, RejectsCorruptedFiles) {
  auto package = hrt::hrtc::load_json(dir_ / "manifest.json");
  ASSERT_TRUE(package);
  ASSERT_TRUE(hrt::hrtc::save(*package, dir_ / "demo.hrtc"));
  auto content = kitzoo::os::read_file(dir_ / "demo.hrtc");
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(content.data());

  EXPECT_FALSE(hrt::hrtc::load(std::span(bytes, content.size() - 1)));
  EXPECT_FALSE(hrt::hrtc::load(std::span(bytes, 3)));
  content.push_back('x');
  EXPECT_FALSE(hrt::hrtc::load(std::span(reinterpret_cast<const std::uint8_t*>(content.data()), content.size())));
  content[6] = 'Z';
  EXPECT_FALSE(hrt::hrtc::load(std::span(reinterpret_cast<const std::uint8_t*>(content.data()), content.size())));
}

TEST_F(HrtcTest, LoadsUnalignedBytes) {
  auto package = hrt::hrtc::load_json(dir_ / "manifest.json");
  ASSERT_TRUE(package);
  ASSERT_TRUE(hrt::hrtc::save(*package, dir_ / "demo.hrtc"));
  const auto content = kitzoo::os::read_file(dir_ / "demo.hrtc");
  hrt::hrtc::Bytes shifted(content.size() + 1);
  std::copy(content.begin(), content.end(), shifted.begin() + 1);
  const auto loaded = hrt::hrtc::load(std::span(shifted).subspan(1));
  ASSERT_TRUE(loaded);
  EXPECT_EQ(loaded->find("detect_ncnn")->engine.engine_type, "ncnn");
}

TEST_F(HrtcTest, RejectsOtherFormatVersions) {
  auto package = hrt::hrtc::load_json(dir_ / "manifest.json");
  ASSERT_TRUE(package);
  ASSERT_TRUE(hrt::hrtc::save(*package, dir_ / "demo.hrtc"));
  auto content = kitzoo::os::read_file(dir_ / "demo.hrtc");
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(content.data());
  ASSERT_TRUE(hrt::hrtc::load(std::span(bytes, content.size())));
  // The identifier follows the size prefix and the root offset.
  ASSERT_EQ(content.compare(8, 4, "HRT3"), 0);
  content.replace(8, 4, "HRTC");
  EXPECT_FALSE(hrt::hrtc::load(std::span(bytes, content.size())));
}

TEST_F(HrtcTest, ResolvesPathsAgainstWorkingDirectory) {
  // models/ exists next to the working directory, not next to the manifest.
  std::filesystem::create_directories(dir_ / "sub");
  std::filesystem::copy_file(dir_ / "manifest.json", dir_ / "sub" / "manifest.json");
  EXPECT_TRUE(hrt::hrtc::load_json(dir_ / "sub" / "manifest.json"));
  std::filesystem::current_path(dir_ / "sub");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "sub" / "manifest.json"));
}

TEST_F(HrtcTest, RejectsInvalidManifests) {
  kitzoo::os::write_file(dir_ / "bad.json", R"({"name": "x", "version": "1", "algos": [
    {"name": "a", "config": {"engine": {"engine_type": "ort", "data": "models/missing.onnx"}}}]})");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "bad.json"));

  kitzoo::os::write_file(dir_ / "missing_engine.json", R"({"name": "x", "version": "1", "algos": [
    {"name": "a", "config": {}}]})");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "missing_engine.json"));

  kitzoo::os::write_file(dir_ / "dup.json", R"({"name": "x", "version": "1", "algos": [
    {"name": "a", "config": {"engine": {"engine_type": "ort"}}},
    {"name": "a", "config": {"engine": {"engine_type": "ort"}}}]})");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "dup.json"));

  kitzoo::os::write_file(dir_ / "unknown.json", R"({"name": "x", "version": "1", "extra": true, "algos": [
    {"name": "a", "config": {"engine": {"engine_type": "ort"}}}]})");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "unknown.json"));

  kitzoo::os::write_file(dir_ / "empty_engine.json", R"({"name": "x", "version": "1", "algos": [
    {"name": "a", "config": {"engine": {"engine_type": ""}}}]})");
  EXPECT_FALSE(hrt::hrtc::load_json(dir_ / "empty_engine.json"));
}

} // namespace

TEST_F(HrtcTest, EmbedsOcrDictionaryBeforePacking) {
  kitzoo::os::write_file(dir_ / "characters.txt", "中\na\n");
  const kitzoo::json::Json entry{
      {"name", "recognition"},
      {"config",
       {{"engine", {{"engine_type", "ort"}, {"data", "string:weights"}}},
        {"postprocess", {{"type", "Ocr"}, {"stage", "recognition"}, {"dictionary", "characters.txt"}}}}}};
  auto algo = hrt::hrtc::load_algo(entry);
  ASSERT_TRUE(algo);
  std::filesystem::remove(dir_ / "characters.txt");
  const kitzoo::json::Reader config{algo->config};
  EXPECT_EQ(config.raw()["postprocess"]["dictionary"], "string:中\na\n");
  hrt::hrtc::Package package;
  package.name = "ocr";
  package.version = "1.0.0";
  package.algos.push_back(std::move(*algo));
  ASSERT_TRUE(hrt::hrtc::save(package, dir_ / "ocr.hrtc"));
  const auto loaded = hrt::hrtc::load(dir_ / "ocr.hrtc");
  ASSERT_TRUE(loaded);
  EXPECT_EQ(loaded->algos[0].config, package.algos[0].config);
  auto bad = entry;
  bad["config"]["postprocess"]["dictionary"] = 42;
  EXPECT_FALSE(hrt::hrtc::load_algo(bad));
}
