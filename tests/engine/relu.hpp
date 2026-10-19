// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/relu.hpp
// Description: Checks engine ReLU inference and preservation of host inputs.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_RELU_HPP
#define HRT_ENGINE_RELU_HPP

#include <hrt/engine/engine.hpp>

#include <algorithm>
#include <gtest/gtest.h>
#include <span>

namespace hrt::test {

inline auto expect_relu(engine::Engine& engine, core::Buffer& input, const core::Buffer& output,
                        std::span<const float> values) -> void {
  ASSERT_EQ(input.volume(), static_cast<std::int64_t>(values.size()));
  ASSERT_EQ(output.volume(), static_cast<std::int64_t>(values.size()));
  auto* data = static_cast<float*>(input.memory()->mutable_cpu());
  ASSERT_NE(data, nullptr);
  std::copy(values.begin(), values.end(), data);
  ASSERT_TRUE(engine.inference());
  const auto* result = static_cast<const float*>(output.memory()->cpu());
  ASSERT_NE(result, nullptr);
  for (std::size_t i = 0; i < values.size(); ++i)
    EXPECT_FLOAT_EQ(result[i], std::max(values[i], 0.0F)) << "at " << i;
  EXPECT_TRUE(std::equal(values.begin(), values.end(), static_cast<const float*>(input.memory()->cpu())));
}

} // namespace hrt::test

#endif // HRT_ENGINE_RELU_HPP
