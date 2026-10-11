// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/memory_test.cpp
// Description: Tests memory allocation, views, wrapping and strided copies.
// -----------------------------------------------------------------------------

#include <hrt/core/buffer.hpp>
#include <hrt/core/memory.hpp>

#include <gtest/gtest.h>
#include <numeric>

namespace {

TEST(MemoryTest, CreatesPackedHostMemory) {
  auto memory = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, hrt::core::Shape{2, 3, 4});
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->bytes(), 24U);
  EXPECT_EQ(memory->stride(), (hrt::core::Stride{24, 12, 4}));
  EXPECT_TRUE(memory->is_contiguous());
  EXPECT_TRUE(memory->have_cpu());
  EXPECT_FALSE(memory->have_device());
}

TEST(MemoryTest, RefSharesStorage) {
  auto memory = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, 8);
  auto view = memory->ref();
  static_cast<std::uint8_t*>(memory->mutable_cpu())[3] = 42;
  EXPECT_EQ(static_cast<const std::uint8_t*>(view->cpu())[3], 42);
  memory.reset();
  EXPECT_EQ(static_cast<const std::uint8_t*>(view->cpu())[3], 42);
}

TEST(MemoryTest, WrappedMemoryCannotGrow) {
  hrt::Array<std::uint8_t, 16> storage{};
  auto memory = hrt::core::Memory::wrap(hrt::core::MemoryType::Cpu, storage.data(), 0, storage.size());
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->cpu(), storage.data());
  EXPECT_TRUE(memory->resize(hrt::core::Shape{2, 8}));
  EXPECT_FALSE(memory->resize(hrt::core::Shape{17}));
  EXPECT_FALSE(hrt::core::Memory::wrap(hrt::core::MemoryType::Cpu, storage.data(), 0, 8, hrt::core::Shape{16}));
}

TEST(MemoryTest, RejectsTooSmallStride) {
  EXPECT_EQ(hrt::core::Memory::create(hrt::core::MemoryType::Cpu, hrt::core::Shape{2, 4}, hrt::core::Stride{8, 3}),
            nullptr);
}

TEST(MemoryTest, CopiesBetweenPaddedAndPackedRows) {
  // 3 rows of 4 bytes, padded to a pitch of 6.
  auto padded = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, hrt::core::Shape{3, 4}, hrt::core::Stride{18, 6});
  ASSERT_NE(padded, nullptr);
  EXPECT_FALSE(padded->is_contiguous());
  auto* rows = static_cast<std::uint8_t*>(padded->mutable_cpu());
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c)
      rows[r * 6 + c] = static_cast<std::uint8_t>(r * 10 + c);
  }

  auto packed = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, 0);
  ASSERT_TRUE(hrt::core::Memory::copy(*packed, *padded, hrt::core::CopyPolicy::AutoDstStride));
  EXPECT_EQ(packed->stride(), (hrt::core::Stride{12, 4}));
  const auto* out = static_cast<const std::uint8_t*>(packed->cpu());
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c)
      EXPECT_EQ(out[r * 4 + c], r * 10 + c);
  }

  auto back = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, 0);
  ASSERT_TRUE(hrt::core::Memory::copy(*back, *packed, hrt::core::CopyPolicy::KeepSrcStride));
  EXPECT_EQ(back->stride(), packed->stride());
}

TEST(MemoryTest, BufferUsesElementShape) {
  hrt::core::Buffer buffer(hrt::core::MemoryType::Cpu, hrt::core::DataType::Float32, hrt::core::Shape{1, 3, 2, 2});
  ASSERT_TRUE(buffer.valid());
  EXPECT_EQ(buffer.byte_size(), 48U);
  EXPECT_EQ(buffer.memory()->shape(), (hrt::core::Shape{1, 3, 2, 8}));
  EXPECT_EQ(buffer.offset({0, 1, 1, 1}), 16U + 8U + 4U);

  hrt::core::Buffer copy(hrt::core::MemoryType::Cpu);
  auto* data = static_cast<float*>(buffer.memory()->mutable_cpu());
  std::iota(data, data + 12, 0.0F);
  ASSERT_TRUE(hrt::core::Buffer::copy(copy, buffer, hrt::core::CopyPolicy::AutoDstStride));
  EXPECT_EQ(copy.data_type(), hrt::core::DataType::Float32);
  EXPECT_EQ(copy.shape(), buffer.shape());
  EXPECT_FLOAT_EQ(static_cast<const float*>(copy.memory()->cpu())[11], 11.0F);
}

} // namespace
