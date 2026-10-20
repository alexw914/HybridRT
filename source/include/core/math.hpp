// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/core/math.hpp
// Description: Declares scalar activation and quantization math.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_MATH_HPP
#define HRT_CORE_MATH_HPP

#include <cmath>
#include <cstdint>
#include <limits>

namespace hrt::core {

inline auto sigmoid(float x) -> float {
  return 1.0F / (1.0F + std::exp(-x));
}

// Inverse of sigmoid; the largest float for p >= 1.
inline auto logit(float p) -> float {
  return p < 1.0F ? std::log(p / (1.0F - p)) : std::numeric_limits<float>::max();
}

// Affine int8 quantization: real = scale * (q - zero_point).
inline auto dequantize(std::int8_t value, float scale, std::int32_t zero_point) -> float {
  return scale * (static_cast<float>(value) - static_cast<float>(zero_point));
}

} // namespace hrt::core

#endif // HRT_CORE_MATH_HPP
