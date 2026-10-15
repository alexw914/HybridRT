// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/define_test.cpp
// Description: Tests the geometry helpers cascades use to crop detections.
// -----------------------------------------------------------------------------

#include <hrt/core/define.hpp>

#include <gtest/gtest.h>

namespace {

using hrt::core::Rect;
using hrt::core::Rect2f;

TEST(DefineTest, PadScaleCoversWholePixels) {
  // x in [1.2, 4.2) and y in [2.5, 6.5) touch pixels 1..4 and 2..6.
  EXPECT_EQ(hrt::core::pad_scale({1.2F, 2.5F, 3.0F, 4.0F}, 1.0F, 1.0F, {0, 0, 64, 48}), (Rect{1, 2, 4, 5}));
}

TEST(DefineTest, PadScaleGrowsAboutTheCenter) {
  EXPECT_EQ(hrt::core::pad_scale({20.0F, 10.0F, 10.0F, 20.0F}, 1.5F, 1.5F, {0, 0, 64, 48}), (Rect{17, 5, 16, 30}));
  EXPECT_EQ(hrt::core::pad_scale({20.0F, 10.0F, 10.0F, 20.0F}, 2.0F, 1.0F, {0, 0, 64, 48}), (Rect{15, 10, 20, 20}));
  EXPECT_EQ(hrt::core::pad_scale({20.0F, 10.0F, 10.0F, 20.0F}, 0.5F, 0.5F, {0, 0, 64, 48}), (Rect{22, 15, 6, 10}));
}

TEST(DefineTest, PadScaleClipsToBounds) {
  // A head at the top edge grown by 1.5 spans y in [-1.35, 10.35).
  EXPECT_EQ(hrt::core::pad_scale({40.3F, 0.6F, 7.5F, 7.8F}, 1.5F, 1.5F, {0, 0, 64, 48}), (Rect{38, 0, 12, 11}));
  EXPECT_EQ(hrt::core::pad_scale({58.0F, 40.0F, 12.0F, 12.0F}, 1.0F, 1.0F, {0, 0, 64, 48}), (Rect{58, 40, 6, 8}));
  EXPECT_EQ(hrt::core::pad_scale({-8.0F, -8.0F, 4.0F, 4.0F}, 1.0F, 1.0F, {0, 0, 64, 48}), Rect{});
  EXPECT_TRUE(hrt::core::pad_scale({100.0F, 100.0F, 10.0F, 10.0F}, 1.5F, 1.5F, {0, 0, 64, 48}).empty());
}

} // namespace
