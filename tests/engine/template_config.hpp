// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/template_config.hpp
// Description: Reads a backend's config from the reference manifest
//              assets/template/hrtc_template.json.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_TEMPLATE_CONFIG_HPP
#define HRT_ENGINE_TEMPLATE_CONFIG_HPP

#include <hrt/core/containers.hpp>

#include <kitzoo/json/reader.hpp>

#include <filesystem>
#include <string_view>

namespace hrt::test {

// The "config" of the first algo's backend of `engine_type`, as JSON text.
inline auto template_config(std::string_view engine_type) -> String {
  const kitzoo::json::Reader reader(std::filesystem::path(HRT_TEMPLATE_DIR) / "hrtc_template.json");
  for (const auto& backend : reader.raw()["algos"][0]["config"]["engine"]["backends"]) {
    if (backend["engine_type"] == engine_type)
      return backend.value("config", kitzoo::json::Json::object()).dump();
  }
  return {};
}

} // namespace hrt::test

#endif // HRT_ENGINE_TEMPLATE_CONFIG_HPP
