// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/onnx_model.hpp
// Description: Declares a tiny ONNX writer for single-operator test models.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_ONNX_MODEL_HPP
#define HRT_ENGINE_ONNX_MODEL_HPP

#include <hrt/hrtc/hrtc.hpp>

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace hrt::test {

// Minimal protobuf writer for the handful of ONNX fields the models need.
inline auto varint(hrt::String& out, std::uint64_t value) -> void {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

inline auto field(int number, std::uint64_t value) -> hrt::String {
  hrt::String out;
  varint(out, static_cast<std::uint64_t>(number) << 3);
  varint(out, value);
  return out;
}

inline auto field(int number, std::string_view payload) -> hrt::String {
  hrt::String out;
  varint(out, (static_cast<std::uint64_t>(number) << 3) | 2);
  varint(out, payload.size());
  out.append(payload);
  return out;
}

// Float tensor value info; a negative dim is dynamic, named after its position.
inline auto value_info(std::string_view name, std::initializer_list<std::int64_t> dims) -> hrt::String {
  hrt::String shape;
  int position = 0;
  for (auto dim : dims) {
    hrt::String symbol = "d";
    symbol += std::to_string(position++);
    shape += field(1, dim < 0 ? field(2, symbol) : field(1, static_cast<std::uint64_t>(dim)));
  }
  const auto tensor = field(1, 1) + field(2, shape);
  return field(1, name) + field(2, field(1, tensor));
}

// y = op(x) with x and y float tensors of `dims`, opset 13.
inline auto unary_model(std::string_view op, std::initializer_list<std::int64_t> dims) -> hrtc::Bytes {
  const auto node = field(1, "x") + field(2, "y") + field(4, op);
  const auto graph =
      field(1, node) + field(2, "g") + field(11, value_info("x", dims)) + field(12, value_info("y", dims));
  const auto model = field(1, 7) + field(7, graph) + field(8, field(2, 13));
  return {model.begin(), model.end()};
}

} // namespace hrt::test

#endif // HRT_ENGINE_ONNX_MODEL_HPP
