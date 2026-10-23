// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/core/config.cpp
// Description: Implements non-throwing checks of JSON config sections.
// -----------------------------------------------------------------------------

#include "include/core/config.hpp"

#include <kitzoo/utilities/str.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace hrt::core {

namespace {

using kitzoo::json::Json;

auto is_numbers(const Json& value) -> bool {
  return value.is_array() && std::all_of(value.begin(), value.end(), [](const Json& item) -> bool {
           return item.is_number() ||
                  (item.is_array() &&
                   std::all_of(item.begin(), item.end(), [](const Json& x) -> bool { return x.is_number(); }));
         });
}

auto check_value(const Json& value, const Json& reference, const String& path, Problems& problems) -> void {
  const auto fail = [&](std::string_view expected) -> void {
    problems.push_back(path + ": expected " + String(expected));
  };
  switch (reference.type()) {
  case Json::value_t::null:
  case Json::value_t::discarded:
  case Json::value_t::binary:
    return;
  case Json::value_t::boolean:
    if (!value.is_boolean())
      fail("a boolean");
    return;
  case Json::value_t::number_unsigned:
    if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<std::int64_t>() >= 0))
      fail("a non-negative integer");
    return;
  case Json::value_t::number_integer:
    if (!value.is_number_integer())
      fail("an integer");
    return;
  case Json::value_t::number_float:
    if (!value.is_number())
      fail("a number");
    return;
  case Json::value_t::string:
    if (!value.is_string())
      fail("a string");
    return;
  case Json::value_t::array:
    if (reference.empty()) {
      if (!is_numbers(value))
        fail("an array of numbers");
      return;
    }
    if (!value.is_array() || value.size() != reference.size()) {
      fail("an array of " + std::to_string(reference.size()));
      return;
    }
    for (std::size_t i = 0; i < value.size(); ++i)
      check_value(value[i], reference[i], path + "[" + std::to_string(i) + "]", problems);
    return;
  case Json::value_t::object:
    if (!value.is_object())
      fail("an object");
    else if (!reference.empty())
      check_json(value, reference, path, {}, problems);
    return;
  }
}

} // namespace

auto join(const Problems& problems) -> String {
  const Vector<std::string_view> parts(problems.begin(), problems.end());
  return kitzoo::util::join(parts, "; ");
}

auto check_json(const Json& config, const Json& defaults, std::string_view section,
                std::initializer_list<std::string_view> extra, Problems& problems) -> void {
  if (!config.is_object()) {
    problems.push_back(String(section) + ": expected an object");
    return;
  }
  for (const auto& [key, value] : config.items()) {
    const auto path = String(section) + "." + key;
    if (std::find(extra.begin(), extra.end(), key) != extra.end())
      continue;
    const auto known = defaults.find(key);
    if (known == defaults.end())
      problems.push_back(path + ": unknown key");
    else
      check_value(value, *known, path, problems);
  }
}

auto check_enums(const Json& config, const Json& parsed, const Json& first, std::string_view section,
                 Problems& problems) -> void {
  for (const auto& [key, value] : config.items()) {
    const auto it = parsed.find(key);
    const auto unknown = first.find(key);
    if (!value.is_string() || it == parsed.end() || unknown == first.end() || !it->is_string())
      continue;
    // Only enums change the string; aliases map to another value than the
    // first, which unknown strings fall back to.
    if (*it == *unknown && value != *unknown)
      problems.push_back(String(section) + "." + key + ": unknown value \"" + value.get<String>() + "\"");
  }
}

auto check_keys(const Json& object, std::initializer_list<std::string_view> keys, std::string_view section,
                Problems& problems) -> void {
  if (!object.is_object()) {
    problems.push_back(String(section) + ": expected an object");
    return;
  }
  for (const auto& [key, value] : object.items()) {
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
      problems.push_back(String(section) + "." + key + ": unknown key");
  }
}

namespace {

template <typename Valid>
auto find(const Json& object, std::string_view key, std::string_view expected, std::string_view section,
          Problems& problems, Valid valid) -> const Json* {
  if (!object.is_object())
    return nullptr;
  const auto it = object.find(String(key));
  if (it == object.end())
    return nullptr;
  if (!valid(*it)) {
    problems.push_back(String(section) + "." + String(key) + ": expected " + String(expected));
    return nullptr;
  }
  return &*it;
}

} // namespace

auto read(const Json& object, std::string_view key, int& value, std::string_view section, Problems& problems) -> void {
  if (const auto* found = find(object, key, "an integer", section, problems, [](const Json& j) -> bool {
        return j.is_number_integer() && j.get<std::int64_t>() >= std::numeric_limits<int>::min() &&
               j.get<std::int64_t>() <= std::numeric_limits<int>::max();
      }))
    value = static_cast<int>(found->get<std::int64_t>());
}

auto read(const Json& object, std::string_view key, bool& value, std::string_view section, Problems& problems) -> void {
  if (const auto* found =
          find(object, key, "a boolean", section, problems, [](const Json& j) -> bool { return j.is_boolean(); }))
    value = found->get<bool>();
}

auto read(const Json& object, std::string_view key, String& value, std::string_view section, Problems& problems)
    -> void {
  if (const auto* found =
          find(object, key, "a string", section, problems, [](const Json& j) -> bool { return j.is_string(); }))
    value = found->get<String>();
}

auto read(const Json& object, std::string_view key, Vector<String>& value, std::string_view section, Problems& problems)
    -> void {
  if (const auto* found = find(object, key, "an array of strings", section, problems, [](const Json& j) -> bool {
        return j.is_array() &&
               std::all_of(j.begin(), j.end(), [](const Json& item) -> bool { return item.is_string(); });
      }))
    value = found->get<Vector<String>>();
}

auto read(const Json& object, std::string_view key, Map<String, String>& value, std::string_view section,
          Problems& problems) -> void {
  if (const auto* found = find(object, key, "an object of strings", section, problems, [](const Json& j) -> bool {
        return j.is_object() &&
               std::all_of(j.begin(), j.end(), [](const Json& item) -> bool { return item.is_string(); });
      }))
    value = found->get<Map<String, String>>();
}

auto read(const Json& object, std::string_view key, Map<String, Vector<std::int64_t>>& value, std::string_view section,
          Problems& problems) -> void {
  const auto shape = [](const Json& j) -> bool {
    return j.is_array() && !j.empty() &&
           std::all_of(j.begin(), j.end(), [](const Json& dim) -> bool { return dim.is_number_integer() && dim > 0; });
  };
  if (const auto* found = find(object, key, "an object of shapes", section, problems, [&](const Json& j) -> bool {
        return j.is_object() && std::all_of(j.begin(), j.end(), shape);
      }))
    value = found->get<Map<String, Vector<std::int64_t>>>();
}

auto fix_shapes(const Map<String, Vector<std::int64_t>>& fixed, const Vector<String>& names,
                Vector<BufferProto>& protos, std::string_view section, Problems& problems) -> void {
  for (const auto& [name, shape] : fixed) {
    if (std::find(names.begin(), names.end(), name) == names.end())
      problems.push_back(String(section) + "." + name + ": no such model input");
  }
  for (std::size_t i = 0; i < protos.size(); ++i) {
    auto& shape = protos[i].shape;
    const auto it = fixed.find(names[i]);
    if (it == fixed.end()) {
      if (!shape.empty() && shape[0] < 0)
        shape[0] = 1;
      continue;
    }
    const auto& want = it->second;
    const auto fits = want.size() == shape.size() &&
                      std::equal(shape.begin(), shape.end(), want.begin(),
                                 [](auto model, auto dim) -> bool { return model < 0 || model == dim; });
    if (fits)
      shape = want;
    else
      problems.push_back(String(section) + "." + names[i] + ": does not fit the model shape " + Json(shape).dump());
  }
}

} // namespace hrt::core
