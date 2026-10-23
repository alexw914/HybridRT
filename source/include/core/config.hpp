// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/core/config.hpp
// Description: Declares non-throwing checks of JSON config sections against
//              their parameter structs.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_CONFIG_HPP
#define HRT_CORE_CONFIG_HPP

#include <hrt/core/buffer.hpp>
#include <hrt/core/containers.hpp>

#include <kitzoo/json/reader.hpp>

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace hrt::core {

// One "section.key: reason" line per problem; returned by public checks.
using Problems = Vector<String>;

auto join(const Problems& problems) -> String;

// Checks keys and value types against `defaults` (a serialized default param)
// so a later get<T>() cannot throw; `extra` keys are skipped. A non-empty
// default array is fixed-size, an empty one takes numbers or arrays of
// numbers, and an empty object takes anything.
auto check_json(const kitzoo::json::Json& config, const kitzoo::json::Json& defaults, std::string_view section,
                std::initializer_list<std::string_view> extra, Problems& problems) -> void;

// Reports strings that `parsed` maps to `first`, the fallback of unknown enums.
auto check_enums(const kitzoo::json::Json& config, const kitzoo::json::Json& parsed, const kitzoo::json::Json& first,
                 std::string_view section, Problems& problems) -> void;

// check_json against Param's defaults, then rejects enum strings Param does
// not know: nlohmann silently maps those to the enum's first value.
template <typename Param>
auto check_param(const kitzoo::json::Json& config, std::string_view section,
                 std::initializer_list<std::string_view> extra, Problems& problems) -> void {
  const auto before = problems.size();
  const kitzoo::json::Json defaults = Param{};
  check_json(config, defaults, section, extra, problems);
  if (problems.size() != before)
    return;
  // `extra` keys were not checked, so they stay out of the parse.
  kitzoo::json::Json known = config;
  for (const auto key : extra)
    known.erase(String(key));
  kitzoo::json::Json probe = known;
  for (auto& [key, value] : probe.items()) {
    if (value.is_string())
      value = "\x01";
  }
  check_enums(known, kitzoo::json::Json(known.get<Param>()), kitzoo::json::Json(probe.get<Param>()), section, problems);
}

auto check_keys(const kitzoo::json::Json& object, std::initializer_list<std::string_view> keys,
                std::string_view section, Problems& problems) -> void;

// Reads an optional key; a mistyped value is a problem and leaves `value`.
auto read(const kitzoo::json::Json& object, std::string_view key, int& value, std::string_view section,
          Problems& problems) -> void;
auto read(const kitzoo::json::Json& object, std::string_view key, bool& value, std::string_view section,
          Problems& problems) -> void;
auto read(const kitzoo::json::Json& object, std::string_view key, String& value, std::string_view section,
          Problems& problems) -> void;
auto read(const kitzoo::json::Json& object, std::string_view key, Vector<String>& value, std::string_view section,
          Problems& problems) -> void;
// An object whose values are all strings.
auto read(const kitzoo::json::Json& object, std::string_view key, Map<String, String>& value, std::string_view section,
          Problems& problems) -> void;
// An object of shapes: arrays of positive integers.
auto read(const kitzoo::json::Json& object, std::string_view key, Map<String, Vector<std::int64_t>>& value,
          std::string_view section, Problems& problems) -> void;

// Fixes the dynamic dims of the model tensors `protos` (named `names`) from `fixed`; a tensor without an entry only
// gets a dynamic leading batch dim, set to 1. An entry must name a tensor and keep its rank and static dims.
auto fix_shapes(const Map<String, Vector<std::int64_t>>& fixed, const Vector<String>& names,
                Vector<BufferProto>& protos, std::string_view section, Problems& problems) -> void;

// A string naming a value of an enum serialized with NLOHMANN_JSON_SERIALIZE_ENUM.
template <typename Enum>
auto read_enum(const kitzoo::json::Json& object, std::string_view key, Enum& value, std::string_view section,
               Problems& problems) -> void {
  String name;
  const auto before = problems.size();
  read(object, key, name, section, problems);
  if (problems.size() != before || name.empty())
    return;
  const auto parsed = kitzoo::json::Json(name).get<Enum>();
  if (kitzoo::json::Json(parsed) != name) {
    problems.push_back(String(section) + "." + String(key) + ": unknown value \"" + name + "\"");
    return;
  }
  value = parsed;
}

} // namespace hrt::core

#endif // HRT_CORE_CONFIG_HPP
