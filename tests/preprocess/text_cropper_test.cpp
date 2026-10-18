// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/preprocess/text_cropper_test.cpp
// Description: Tests text box crops and tall rotation.
// -----------------------------------------------------------------------------

#include <hrt/preprocess/text_cropper.hpp>

#include <gtest/gtest.h>

namespace {

TEST(TextCropperTest, TruncatesBoxesAndRotatesTallText) {
  hrt::core::Image image(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr, hrt::core::DataType::Uint8,
                         {32, 32});
  auto* pixels = static_cast<std::uint8_t*>(image.memory()->mutable_cpu());
  const auto pitch = image.stride()[1];
  for (std::size_t y = 0; y < 32; ++y)
    for (std::size_t x = 0; x < 32; ++x) {
      pixels[y * pitch + x * 3] = static_cast<std::uint8_t>(x);
      pixels[y * pitch + x * 3 + 1] = static_cast<std::uint8_t>(y);
    }
  hrt::preprocess::TextCropper cropper;
  const hrt::core::Rect2f rect{2.7F, 3.2F, 10.9F, 22.5F};
  auto unrotated = cropper.process(image, rect, false);
  ASSERT_TRUE(unrotated);
  EXPECT_EQ(unrotated->size(), (hrt::core::Size{10, 22}));
  const auto* first = static_cast<const std::uint8_t*>(unrotated->memory()->cpu());
  EXPECT_EQ(first[0], 2);
  EXPECT_EQ(first[1], 3);
  // Counterclockwise rotation moves the top-right pixel to the origin.
  auto rotated = cropper.process(image, rect);
  ASSERT_TRUE(rotated);
  EXPECT_EQ(rotated->size(), (hrt::core::Size{22, 10}));
  first = static_cast<const std::uint8_t*>(rotated->memory()->cpu());
  EXPECT_EQ(first[0], 11);
  EXPECT_EQ(first[1], 3);
  EXPECT_FALSE(cropper.process(image, {-1, 2, 10, 10}));
  EXPECT_FALSE(cropper.process(image, {30, 2, 10, 10}));
  EXPECT_FALSE(cropper.process(image, {2, 2, 1.5F, 10}));
}

} // namespace
