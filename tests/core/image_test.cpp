// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/image_test.cpp
// Description: Tests that CUDA image storage is freed with its last owner.
// -----------------------------------------------------------------------------

#include <hrt/core/context.hpp>
#include <hrt/core/image.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace {

TEST(ImageTest, ReleasesCudaImageStorageWithSharedOwnership) {
  auto context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  context->set_current();

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::size_t free_before = 0;
  std::size_t total = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);
  constexpr std::size_t kImageBytes = 16U * 1024U * 1024U;
  {
    auto image = std::make_unique<hrt::core::Image>(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::Gray,
                                                    hrt::core::DataType::Uint8, hrt::core::Size{4096, 4096});
    ASSERT_TRUE(image->valid());
    auto alias = std::make_unique<hrt::core::Image>(*image);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::size_t free_with_image = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_with_image, &total), cudaSuccess);
    EXPECT_LE(free_before - free_with_image, kImageBytes + 1024U * 1024U);

    image.reset();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::size_t free_with_alias = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_with_alias, &total), cudaSuccess);
    EXPECT_NEAR(static_cast<double>(free_with_image), static_cast<double>(free_with_alias), 1024.0 * 1024.0);
    alias.reset();
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::size_t free_after = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);
  EXPECT_NEAR(static_cast<double>(free_before), static_cast<double>(free_after), 1024.0 * 1024.0);

  for (int i = 0; i < 16; ++i) {
    {
      hrt::core::Image image(hrt::core::MemoryType::Cuda, hrt::core::ImageFormat::Gray, hrt::core::DataType::Uint8,
                             {4096, 4096});
      ASSERT_TRUE(image.valid());
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::size_t free_now = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_now, &total), cudaSuccess);
    EXPECT_NEAR(static_cast<double>(free_before), static_cast<double>(free_now), 1024.0 * 1024.0) << "iteration " << i;
  }

  void* external_device = nullptr;
  ASSERT_EQ(cudaMalloc(&external_device, 1024U * 1024U), cudaSuccess);
  {
    auto view = hrt::core::Image::wrap(hrt::core::MemoryType::Cuda, external_device, hrt::core::ImageFormat::Gray,
                                       {1024, 1024});
    ASSERT_TRUE(view);
  }
  EXPECT_EQ(cudaMemset(external_device, 0, 1024U * 1024U), cudaSuccess);
  EXPECT_EQ(cudaFree(external_device), cudaSuccess);
}

} // namespace
