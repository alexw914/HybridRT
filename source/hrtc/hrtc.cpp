// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/hrtc/hrtc.cpp
// Description: Implements hrtc package loading, manifest building and saving.
// -----------------------------------------------------------------------------

#include <hrt/hrtc/hrtc.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>

#include <chrono>
#include <cstring>

#include "include/hrtc/hrtc_generated.h"

namespace hrt::hrtc {

namespace {

using kitzoo::json::Json;

constexpr std::string_view kInlinePrefix = "string:";

// Copies the model files that follow the flatbuffer, in declaration order.
class ByteReader {
public:
  explicit ByteReader(std::span<const std::uint8_t> bytes) : rest_(bytes) {}

  auto take(std::uint64_t size, SharedPtr<const Bytes>& out) -> bool {
    if (size > rest_.size())
      return false;
    const auto head = rest_.first(static_cast<std::size_t>(size));
    out = head.empty() ? nullptr : std::make_shared<const Bytes>(head.begin(), head.end());
    rest_ = rest_.subspan(head.size());
    return true;
  }

  KZ_NODISCARD auto empty() const -> bool { return rest_.empty(); }

private:
  std::span<const std::uint8_t> rest_;
};

auto str(const flatbuffers::String* value) -> String {
  return value == nullptr ? String{} : value->str();
}

auto read_string(const Json& object, const char* key, String& out) -> bool {
  auto it = object.find(key);
  if (it == object.end() || !it->is_string()) {
    KZ_LOG_ERROR("hrtc manifest: \"{}\" must be a string", key);
    return false;
  }
  out = it->get<String>();
  return true;
}

// A file path (absolute, or relative to the working directory), "string:<text>"
// or a JSON object; absent keys leave `out` empty.
auto read_text(const Json& object, const char* key, String& out) -> bool {
  auto it = object.find(key);
  if (it == object.end())
    return true;
  if (it->is_object()) {
    out = it->dump();
    return true;
  }
  if (!it->is_string()) {
    KZ_LOG_ERROR("hrtc manifest: \"{}\" must be a path, \"string:\" text or an object", key);
    return false;
  }
  const auto& value = it->get_ref<const String&>();
  if (value.starts_with(kInlinePrefix)) {
    out = value.substr(kInlinePrefix.size());
    return true;
  }
  const std::filesystem::path path(value);
  std::error_code ec;
  auto content = kitzoo::os::read_file(path, ec);
  if (ec) {
    KZ_LOG_ERROR("hrtc manifest: cannot read {}: {}", path.string(), ec.message());
    return false;
  }
  out = std::move(content);
  return true;
}

auto read_bytes(const Json& object, const char* key, SharedPtr<const Bytes>& out) -> bool {
  String text;
  if (!read_text(object, key, text))
    return false;
  out = text.empty() ? nullptr : std::make_shared<const Bytes>(text.begin(), text.end());
  return true;
}

auto read_engine(const Json& json, EngineParam& engine) -> bool {
  if (!json.is_object() || !read_string(json, "engine_type", engine.engine_type))
    return false;
  if (const auto config = json.find("config"); config != json.end()) {
    if (!config->is_object()) {
      KZ_LOG_ERROR("hrtc manifest: \"config\" of engine {} must be an object", engine.engine_type);
      return false;
    }
    engine.config = config->dump();
  }
  return read_bytes(json, "proto", engine.proto) && read_bytes(json, "data", engine.data);
}

// The engine list lives in config "engine.backends"; it is packed as engines
// and dropped, with the then empty "engine" section, from the stored config.
auto read_algo(const Json& json, AlgoParam& algo) -> bool {
  String config_text;
  if (!json.is_object() || !read_string(json, "name", algo.name) || !read_text(json, "config", config_text))
    return false;
  const kitzoo::json::Reader reader{config_text};
  auto config = reader.raw();
  auto* engine =
      reader.is_parse_success() && config.is_object() && config.contains("engine") ? &config["engine"] : nullptr;
  const auto* backends =
      engine != nullptr && engine->is_object() && engine->contains("backends") ? &(*engine)["backends"] : nullptr;
  if (backends == nullptr || !backends->is_array() || backends->empty()) {
    KZ_LOG_ERROR("hrtc manifest: config of algo {} needs a non-empty \"engine.backends\" array", algo.name);
    return false;
  }
  for (const auto& item : *backends) {
    if (!read_engine(item, algo.engines.emplace_back()))
      return false;
  }
  engine->erase("backends");
  if (engine->empty())
    config.erase("engine");
  algo.config = config.dump();
  return true;
}

auto check_unique_names(const Package& package) -> bool {
  Set<std::string_view> names;
  for (const auto& algo : package.algos) {
    if (algo.name.empty() || !names.insert(algo.name).second) {
      KZ_LOG_ERROR("hrtc: algo name \"{}\" is empty or duplicated", algo.name);
      return false;
    }
  }
  return true;
}

auto parse(std::span<const std::uint8_t> bytes) -> std::optional<Package> {
  constexpr auto kPrefixSize = sizeof(flatbuffers::uoffset_t);
  if (bytes.size() < kPrefixSize) {
    KZ_LOG_ERROR("hrtc: file too small");
    return std::nullopt;
  }
  // `bytes` may be unaligned, so the size prefix is copied out rather than read in place.
  flatbuffers::uoffset_t prefix = 0;
  std::memcpy(&prefix, bytes.data(), kPrefixSize);
  const std::size_t fb_size = flatbuffers::EndianScalar(prefix) + kPrefixSize;
  if (fb_size > bytes.size()) {
    KZ_LOG_ERROR("hrtc: truncated metadata");
    return std::nullopt;
  }
  // Copy the metadata so flatbuffers reads aligned memory regardless of the source.
  const Bytes meta(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(fb_size));
  if (!flatbuffers::BufferHasIdentifier(meta.data(), fb::PackageIdentifier(), true)) {
    KZ_LOG_ERROR("hrtc: not a package of this format version; repack it with hrt-compress");
    return std::nullopt;
  }
  flatbuffers::Verifier verifier(meta.data(), meta.size());
  if (!fb::VerifySizePrefixedPackageBuffer(verifier)) {
    KZ_LOG_ERROR("hrtc: invalid or corrupted metadata");
    return std::nullopt;
  }
  const auto* root = fb::GetSizePrefixedPackage(meta.data());

  Package package;
  package.name = root->name()->str();
  package.version = root->version()->str();
  package.timestamp = root->timestamp();
  package.description = str(root->description());

  ByteReader blobs(bytes.subspan(fb_size));
  if (root->algos() != nullptr) {
    for (const auto* fb_algo : *root->algos()) {
      auto& algo = package.algos.emplace_back();
      algo.name = fb_algo->name()->str();
      algo.config = str(fb_algo->config());
      if (fb_algo->engines() == nullptr)
        continue;
      for (const auto* fb_engine : *fb_algo->engines()) {
        auto& engine = algo.engines.emplace_back();
        engine.engine_type = fb_engine->type()->str();
        engine.config = str(fb_engine->config());
        if (!blobs.take(fb_engine->proto(), engine.proto) || !blobs.take(fb_engine->data(), engine.data)) {
          KZ_LOG_ERROR("hrtc: truncated engine data in algo {}", algo.name);
          return std::nullopt;
        }
      }
    }
  }
  if (!blobs.empty()) {
    KZ_LOG_ERROR("hrtc: unexpected trailing data");
    return std::nullopt;
  }
  if (!check_unique_names(package))
    return std::nullopt;
  return package;
}

} // namespace

auto Package::find(std::string_view algo_name) const -> const AlgoParam* {
  for (const auto& algo : algos) {
    if (algo.name == algo_name)
      return &algo;
  }
  return nullptr;
}

auto load(const std::filesystem::path& path) -> std::optional<Package> {
  std::error_code ec;
  auto content = kitzoo::os::read_file(path, ec);
  if (ec) {
    KZ_LOG_ERROR("hrtc: cannot read {}: {}", path.string(), ec.message());
    return std::nullopt;
  }
  return parse(std::span(reinterpret_cast<const std::uint8_t*>(content.data()), content.size()));
}

auto load(std::span<const std::uint8_t> bytes) -> std::optional<Package> {
  return parse(bytes);
}

auto load_algo(const Json& algo) -> std::optional<AlgoParam> {
  AlgoParam param;
  if (!read_algo(algo, param))
    return std::nullopt;
  return param;
}

auto load_json(const std::filesystem::path& path) -> std::optional<Package> {
  const kitzoo::json::Reader reader(path);
  const auto& json = reader.raw();
  if (!reader.is_parse_success() || !json.is_object()) {
    KZ_LOG_ERROR("hrtc manifest: cannot load {}: {}", path.string(), reader.error_info());
    return std::nullopt;
  }

  Package package;
  if (!read_string(json, "name", package.name) || !read_string(json, "version", package.version))
    return std::nullopt;
  if (json.contains("description") && !read_string(json, "description", package.description))
    return std::nullopt;
  package.timestamp = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

  auto algos = json.find("algos");
  if (algos == json.end() || !algos->is_array() || algos->empty()) {
    KZ_LOG_ERROR("hrtc manifest: needs a non-empty \"algos\" array");
    return std::nullopt;
  }
  for (const auto& item : *algos) {
    if (!read_algo(item, package.algos.emplace_back()))
      return std::nullopt;
  }
  if (!check_unique_names(package))
    return std::nullopt;
  return package;
}

auto save(const Package& package, const std::filesystem::path& path) -> bool {
  flatbuffers::FlatBufferBuilder builder;
  Vector<const Bytes*> blobs;
  auto add_blob = [&blobs](const SharedPtr<const Bytes>& bytes) -> std::uint64_t {
    if (bytes == nullptr || bytes->empty())
      return 0;
    blobs.push_back(bytes.get());
    return bytes->size();
  };

  Vector<flatbuffers::Offset<fb::Algo>> fb_algos;
  for (const auto& algo : package.algos) {
    Vector<flatbuffers::Offset<fb::Engine>> fb_engines;
    for (const auto& engine : algo.engines) {
      const auto proto = add_blob(engine.proto);
      const auto data = add_blob(engine.data);
      fb_engines.push_back(fb::CreateEngineDirect(
          builder, engine.engine_type.c_str(), engine.config.empty() ? nullptr : engine.config.c_str(), proto, data));
    }
    fb_algos.push_back(fb::CreateAlgoDirect(builder, algo.name.c_str(),
                                            algo.config.empty() ? nullptr : algo.config.c_str(), &fb_engines));
  }
  auto root = fb::CreatePackageDirect(builder, package.name.c_str(), package.version.c_str(), package.timestamp,
                                      package.description.c_str(), &fb_algos);
  fb::FinishSizePrefixedPackageBuffer(builder, root);

  std::error_code ec;
  kitzoo::os::write_file(
      path, std::string_view(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize()), ec);
  for (const auto* blob : blobs) {
    if (ec)
      break;
    kitzoo::os::append_file(path, std::string_view(reinterpret_cast<const char*>(blob->data()), blob->size()), ec);
  }
  KZ_CHECK_RETURN(!ec, false, "hrtc: cannot write {}: {}", path.string(), ec.message());
  return true;
}

} // namespace hrt::hrtc
