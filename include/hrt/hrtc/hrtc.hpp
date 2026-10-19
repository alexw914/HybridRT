// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/hrtc/hrtc.hpp
// Description: Declares the hrtc model package and its loading, building and
//              saving.
// -----------------------------------------------------------------------------

#ifndef HRT_HRTC_HRTC_HPP
#define HRT_HRTC_HRTC_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

namespace hrt::hrtc {

using Bytes = Vector<std::uint8_t>;

// One backend build of an algorithm. Model files are shared, null when absent,
// and read by the engine selected by engine_type (e.g. ncnn param in proto,
// weights in data).
struct EngineParam {
  String engine_type;
  String config; // JSON settings of this engine; empty means none.
  SharedPtr<const Bytes> proto;
  SharedPtr<const Bytes> data;
};

struct AlgoParam {
  String name;
  Vector<EngineParam> engines;
  String config; // JSON text with "preprocess", "engine" and "postprocess" sections.
};

struct Package {
  String name;
  String version;
  std::uint64_t timestamp = 0;
  String description;
  Vector<AlgoParam> algos;

  KZ_NODISCARD auto find(std::string_view algo_name) const -> const AlgoParam*;
};

auto load(const std::filesystem::path& path) -> std::optional<Package>;

auto load(std::span<const std::uint8_t> bytes) -> std::optional<Package>;

// Builds a package from a JSON manifest (assets/template/hrtc_template.json).
// Model files and algo configs are a file path (absolute, or relative to the
// working directory), "string:<text>" or a JSON object; "engine.backends" of
// each algo config becomes `engines`, each with its own "config" object.
auto load_json(const std::filesystem::path& path) -> std::optional<Package>;

// One entry of a manifest's "algos", read as load_json reads it.
auto load_algo(const kitzoo::json::Json& algo) -> std::optional<AlgoParam>;

auto save(const Package& package, const std::filesystem::path& path) -> bool;

} // namespace hrt::hrtc

#endif // HRT_HRTC_HRTC_HPP
