// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tools/hrt_compress.cpp
// Description: Packs JSON manifests into .hrtc files, prints their contents and
//              extracts their blobs.
// -----------------------------------------------------------------------------

#include <hrt/core/version.hpp>
#include <hrt/hrtc/hrtc.hpp>
#include <hrt/model/image_model.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>

#include <string_view>

namespace {

auto size_of(const hrt::SharedPtr<const hrt::hrtc::Bytes>& bytes) -> std::size_t {
  return bytes == nullptr ? 0 : bytes->size();
}

auto text_of(const hrt::SharedPtr<const hrt::hrtc::Bytes>& bytes) -> std::string_view {
  return bytes == nullptr ? std::string_view{}
                          : std::string_view(reinterpret_cast<const char*>(bytes->data()), bytes->size());
}

auto print(const hrt::hrtc::Package& package) -> void {
  KZ_LOG_INFO("Name:        {}", package.name);
  KZ_LOG_INFO("Version:     {}", package.version);
  KZ_LOG_INFO("Timestamp:   {}", package.timestamp);
  KZ_LOG_INFO("Description: {}", package.description);
  for (const auto& algo : package.algos) {
    KZ_LOG_INFO("Algo {}  config={}B", algo.name, algo.config.size());
    for (std::size_t i = 0; i < algo.engines.size(); ++i) {
      const auto& engine = algo.engines[i];
      KZ_LOG_INFO("  [{}] {}  config={}B proto={}B data={}B", i, engine.engine_type, engine.config.size(),
                  size_of(engine.proto), size_of(engine.data));
    }
  }
}

auto write_blob(const std::filesystem::path& path, std::string_view blob) -> bool {
  if (blob.empty())
    return true;
  std::error_code ec;
  if (std::filesystem::is_symlink(path, ec)) {
    KZ_LOG_ERROR("cannot extract to symbolic link: {}", path.string());
    return false;
  }
  ec.clear();
  kitzoo::os::write_file(path, blob, ec);
  if (ec)
    KZ_LOG_ERROR("cannot write {}: {}", path.string(), ec.message());
  return !ec;
}

auto extract(const hrt::hrtc::Package& package, const std::filesystem::path& dir) -> bool {
  for (const auto& algo : package.algos) {
    if (algo.name.empty() || algo.name == "." || algo.name == ".." ||
        algo.name.find_first_of("/\\:") != hrt::String::npos || algo.name.find('\0') != hrt::String::npos) {
      KZ_LOG_ERROR("cannot extract algo with unsafe name: {}", algo.name);
      return false;
    }
  }
  std::error_code ec;
  kitzoo::os::make_directory(dir, ec, true);
  if (ec) {
    KZ_LOG_ERROR("cannot create {}: {}", dir.string(), ec.message());
    return false;
  }
  for (const auto& algo : package.algos) {
    if (!write_blob(dir / (algo.name + ".config"), algo.config))
      return false;
    for (std::size_t i = 0; i < algo.engines.size(); ++i) {
      const auto& engine = algo.engines[i];
      const auto prefix = algo.name + ".engine." + std::to_string(i);
      if (!write_blob(dir / (prefix + ".config"), engine.config) ||
          !write_blob(dir / (prefix + ".proto"), text_of(engine.proto)) ||
          !write_blob(dir / (prefix + ".data"), text_of(engine.data)))
        return false;
    }
  }
  return true;
}

auto check(const hrt::hrtc::Package& package, bool as_error) -> bool {
  bool clean = true;
  for (const auto& algo : package.algos) {
    for (const auto& problem : hrt::model::check_config(algo)) {
      if (as_error)
        KZ_LOG_ERROR("Algo {}: {}", algo.name, problem);
      else
        KZ_LOG_WARN("Algo {}: {}", algo.name, problem);
      clean = false;
    }
  }
  return clean;
}

} // namespace

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hrt-compress", "Build and inspect hrtc model packages");
  auto add = options.add_options();
  add("j,json", "JSON manifest to pack (see assets/template/hrtc_template.json)", kitzoo::core::value<hrt::String>());
  add("o,output", "Output .hrtc file for --json", kitzoo::core::value<hrt::String>());
  add("i,input", "Existing .hrtc file to inspect", kitzoo::core::value<hrt::String>());
  add("x,extract", "Directory to extract --input blobs into", kitzoo::core::value<hrt::String>());
  add("c,check", "Check the configs of --json or --input against the built-in parameters");
  add("v,version", "Print version");
  add("h,help", "Print help");
  auto args = options.parse(argc, argv);

  if (args.count("help") != 0U) {
    KZ_LOG_INFO("\n{}", options.help());
    return 0;
  }
  if (args.count("version") != 0U) {
    KZ_LOG_INFO("hrt-compress {}", hrt::core::kVersion);
    return 0;
  }

  if (args.count("check") != 0U && (args.count("json") != 0U || args.count("input") != 0U)) {
    auto package = args.count("json") != 0U ? hrt::hrtc::load_json(args["json"].as<hrt::String>())
                                            : hrt::hrtc::load(args["input"].as<hrt::String>());
    if (!package || !check(*package, true))
      return 1;
    KZ_LOG_INFO("All {} algo configs are valid", package->algos.size());
    return 0;
  }

  if (args.count("json") != 0U && args.count("output") != 0U) {
    auto package = hrt::hrtc::load_json(args["json"].as<hrt::String>());
    if (!package)
      return 1;
    // Custom postprocessor types are only known to their factories, so packing
    // warns instead of failing.
    check(*package, false);
    if (!hrt::hrtc::save(*package, args["output"].as<hrt::String>()))
      return 1;
    print(*package);
    return 0;
  }

  if (args.count("input") != 0U) {
    auto package = hrt::hrtc::load(args["input"].as<hrt::String>());
    if (!package)
      return 1;
    print(*package);
    if (args.count("extract") != 0U && !extract(*package, args["extract"].as<hrt::String>()))
      return 1;
    return 0;
  }

  KZ_LOG_ERROR("Specify --json with --output, --input, or --check with either.\n{}", options.help());
  return 1;
}
