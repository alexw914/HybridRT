// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/template_config.hpp
// Description: Reads device configs and model algos from the reference
//              manifests in assets/template.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_TEMPLATE_CONFIG_HPP
#define HRT_ENGINE_TEMPLATE_CONFIG_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/json/reader.hpp>

#include <filesystem>
#include <string_view>

namespace hrt::test {

// The repository root, which the template's fixture paths are relative to.
inline auto source_root() -> std::filesystem::path {
  return std::filesystem::path(HRT_TEMPLATE_DIR).parent_path().parent_path();
}

// The backend config of engine_type in hrtc_device_config.json, as JSON text.
inline auto template_config(std::string_view engine_type) -> String {
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_TEMPLATE_DIR) / "hrtc_device_config.json");
  for (const auto& backend : reader.raw()["algos"][0]["config"]["engine"]["backends"]) {
    if (backend["engine_type"].is_string() && backend["engine_type"].get_ref<const String&>() == engine_type)
      return backend.value("config", kitzoo::json::Json::object()).dump();
  }
  return {};
}

// Algo `name` of the template as {"name", "config"}, with backend model paths
// made absolute against the repository root; null when there is none.
inline auto template_algo(std::string_view name) -> kitzoo::json::Json {
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_TEMPLATE_DIR) / "hrtc_template.json");
  for (auto algo : reader.raw()["algos"]) {
    if (!algo["name"].is_string() || algo["name"].get_ref<const String&>() != name)
      continue;
    for (auto& backend : algo["config"]["engine"]["backends"]) {
      if (backend.contains("data") && backend["data"].is_string())
        backend["data"] = (source_root() / backend["data"].get_ref<const String&>()).string();
    }
    return algo;
  }
  return {};
}

// The model file of the first backend of a template_algo() result.
inline auto template_model(const kitzoo::json::Json& algo) -> std::filesystem::path {
  return algo["config"]["engine"]["backends"][0]["data"].get<String>();
}

} // namespace hrt::test

#endif // HRT_ENGINE_TEMPLATE_CONFIG_HPP
