// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/define_test.cpp
// Description: Tests the rect templates and the geometry helpers built on them.
// -----------------------------------------------------------------------------

#include <hrt/core/define.hpp>

#include <gtest/gtest.h>
#include <limits>

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

TEST(DefineTest, BuildsRectsFromCornersCentersAndOtherTypes) {
  EXPECT_EQ(Rect2f({30.0F, 5.0F}, {10.0F, 25.0F}), (Rect2f{10.0F, 5.0F, 20.0F, 20.0F}));
  EXPECT_EQ(hrt::core::center_box(20.0F, 15.0F, 10.0F, 6.0F), (Rect2f{15.0F, 12.0F, 10.0F, 6.0F}));
  EXPECT_EQ(Rect2f(Rect{1, 2, 3, 4}), (Rect2f{1.0F, 2.0F, 3.0F, 4.0F}));
  EXPECT_EQ(Rect(Rect2f{1.7F, 2.2F, 3.9F, 4.0F}), (Rect{1, 2, 3, 4}));
  EXPECT_EQ((Rect{2, 3, 4, 5}).br().x, 6);
  EXPECT_EQ((Rect{2, 3, 4, 5}).size(), (hrt::core::Size{4, 5}));
}

TEST(DefineTest, IntersectsAndMeasuresOverlap) {
  EXPECT_EQ(hrt::core::intersect(Rect{0, 0, 10, 10}, Rect{5, 5, 10, 10}), (Rect{5, 5, 5, 5}));
  EXPECT_TRUE(hrt::core::intersect(Rect{0, 0, 4, 4}, Rect{4, 0, 4, 4}).empty());
  // Overlap 25 of union 175.
  EXPECT_FLOAT_EQ(hrt::core::iou(Rect{0, 0, 10, 10}, Rect{5, 5, 10, 10}), 25.0F / 175.0F);
  EXPECT_FLOAT_EQ(hrt::core::iou(Rect2f{0, 0, 2, 2}, Rect2f{0, 0, 2, 2}), 1.0F);
}

TEST(DefineTest, ValidBoxRejectsEmptyAndNonfiniteRects) {
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(hrt::core::valid_box(Rect2f{0, 0, 1, 1}));
  EXPECT_FALSE(hrt::core::valid_box(Rect2f{0, 0, 0, 1}));
  EXPECT_FALSE(hrt::core::valid_box(Rect2f{nan, 0, 1, 1}));
  EXPECT_FALSE(hrt::core::valid_box(hrt::core::center_box(0.0F, 0.0F, std::numeric_limits<float>::infinity(), 1.0F)));
}

} // namespace
