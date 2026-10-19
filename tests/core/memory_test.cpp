// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/memory_test.cpp
// Description: Tests memory allocation, views, wrapping and strided copies.
// -----------------------------------------------------------------------------

#include <hrt/core/buffer.hpp>
#include <hrt/core/context.hpp>
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

TEST(MemoryTest, DmaBufferIsMappedAndSharedByFd) {
  auto memory = hrt::core::Memory::create(hrt::core::MemoryType::DmaBuffer, hrt::core::Shape{1, 10, 3});
  if (memory == nullptr)
    GTEST_SKIP() << "no usable dma-heap";
  EXPECT_EQ(memory->memory_type(), hrt::core::MemoryType::DmaBuffer);
  // Rows are padded to 16 elements for media devices.
  EXPECT_EQ(memory->stride(), (hrt::core::Stride{48, 48, 3}));
  EXPECT_TRUE(memory->have_cpu());
  EXPECT_TRUE(memory->have_device());
  EXPECT_GT(memory->device(), 0U);
  static_cast<std::uint8_t*>(memory->mutable_cpu())[5] = 7;
  auto view = memory->ref();
  // Device access ends the CPU window; reading again starts a new one.
  EXPECT_EQ(view->device(), memory->device());
  static_cast<void>(view->mutable_device());
  EXPECT_EQ(static_cast<const std::uint8_t*>(memory->cpu())[5], 7);
  ASSERT_TRUE(memory->resize(hrt::core::Shape{1, 64, 3}));
  EXPECT_EQ(memory->bytes(), 192U);
}

TEST(MemoryTest, WrapsDmaBufferFdWithoutOwningIt) {
  auto owner = hrt::core::Memory::create(hrt::core::MemoryType::DmaBuffer, hrt::core::Shape{4, 16});
  if (owner == nullptr)
    GTEST_SKIP() << "no usable dma-heap";
  static_cast<std::uint8_t*>(owner->mutable_cpu())[17] = 9;
  const auto fd = owner->mutable_device();
  {
    // A null host pointer maps the fd here, as for a decoder frame.
    auto view = hrt::core::Memory::wrap(hrt::core::MemoryType::DmaBuffer, nullptr, fd, 64, hrt::core::Shape{4, 16});
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->device(), fd);
    EXPECT_NE(view->cpu(), owner->cpu());
    EXPECT_EQ(static_cast<const std::uint8_t*>(view->cpu())[17], 9);
    static_cast<std::uint8_t*>(view->mutable_cpu())[3] = 5;
    static_cast<void>(view->mutable_device());
    EXPECT_FALSE(view->resize(hrt::core::Shape{8, 16}));
  }
  // Dropping the view unmapped it but left the fd open.
  EXPECT_EQ(static_cast<const std::uint8_t*>(owner->cpu())[3], 5);
  EXPECT_EQ(hrt::core::Memory::wrap(hrt::core::MemoryType::DmaBuffer, nullptr, 0, 64), nullptr);
}

TEST(MemoryTest, CopiesThroughAscendDeviceMemory) {
  auto context = hrt::core::Context::create(hrt::core::DeviceType::Ascend);
  if (context == nullptr)
    GTEST_SKIP() << "Ascend device unavailable";
  context->make_current();
  // Acl comes from aclrtMalloc and Dvpp from the media allocator VPC needs.
  for (const auto type : {hrt::core::MemoryType::Acl, hrt::core::MemoryType::Dvpp}) {
    SCOPED_TRACE(hrt::core::to_string(type));
    auto device = hrt::core::Memory::create(type, hrt::core::Shape{4, 16});
    ASSERT_NE(device, nullptr);
    EXPECT_FALSE(device->have_cpu());
    EXPECT_NE(device->device(), 0U);
    auto host = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, hrt::core::Shape{4, 16});
    auto back = hrt::core::Memory::create(hrt::core::MemoryType::Cpu, hrt::core::Shape{4, 16});
    ASSERT_NE(host, nullptr);
    ASSERT_NE(back, nullptr);
    auto* bytes = static_cast<std::uint8_t*>(host->mutable_cpu());
    for (std::size_t i = 0; i < 64; ++i)
      bytes[i] = static_cast<std::uint8_t>(i * 3 + 1);
    const auto stream = context->create_stream();
    ASSERT_TRUE(hrt::core::Memory::copy(*device, *host, hrt::core::CopyPolicy::DirectCopy, stream.get()));
    ASSERT_TRUE(hrt::core::Memory::copy(*back, *device, hrt::core::CopyPolicy::DirectCopy, stream.get()));
    if (stream != nullptr)
      stream->synchronize();
    const auto* result = static_cast<const std::uint8_t*>(back->cpu());
    ASSERT_NE(result, nullptr);
    for (std::size_t i = 0; i < 64; ++i)
      ASSERT_EQ(result[i], bytes[i]) << "byte " << i;
    // Growing reallocates on the device and keeps the memory usable.
    ASSERT_TRUE(device->resize(hrt::core::Shape{8, 16}));
    EXPECT_TRUE(hrt::core::Memory::copy(*device, *host, hrt::core::CopyPolicy::AutoDstStride, stream.get()));
  }
}

} // namespace
