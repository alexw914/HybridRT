// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/tensorrt_engine_test.cpp
// Description: Tests the CUDA-pinned host memory offered by the TensorRT engine.
// -----------------------------------------------------------------------------

#include <hrt/core/context.hpp>
#include <hrt/core/memory.hpp>
#include <hrt/engine/engine.hpp>

#include <algorithm>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace {

auto is_pinned(const void* host) -> bool {
  cudaPointerAttributes attributes{};
  return cudaPointerGetAttributes(&attributes, host) == cudaSuccess && attributes.type == cudaMemoryTypeHost;
}

TEST(TensorrtEngineTest, CreatesPinnedHostMemory) {
  auto context = hrt::core::Context::create(hrt::core::DeviceType::Cuda);
  if (context == nullptr)
    GTEST_SKIP() << "CUDA device is unavailable";
  context->make_current();
  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::TensorRt);
  ASSERT_NE(engine, nullptr);

  auto memory = engine->create_host_memory();
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->memory_type(), hrt::core::MemoryType::Cpu);
  ASSERT_TRUE(memory->resize({4, 1024}));
  auto* host = static_cast<std::uint8_t*>(memory->mutable_cpu());
  ASSERT_NE(host, nullptr);
  EXPECT_TRUE(is_pinned(host));
  EXPECT_EQ(memory->ref()->cpu(), host);

  // Shrinking the shape keeps the allocation until shrink_to_fit.
  ASSERT_TRUE(memory->resize({2, 1024}));
  EXPECT_EQ(memory->cpu(), host);
  ASSERT_TRUE(memory->shrink_to_fit());
  EXPECT_TRUE(is_pinned(memory->cpu()));

  std::fill_n(static_cast<std::uint8_t*>(memory->mutable_cpu()), memory->bytes(), std::uint8_t{7});
  auto device = hrt::core::Memory::create(hrt::core::MemoryType::Cuda, memory->shape());
  auto back = engine->create_host_memory();
  ASSERT_NE(device, nullptr);
  ASSERT_TRUE(back->resize(memory->shape()));
  ASSERT_TRUE(hrt::core::Memory::copy(*device, *memory, hrt::core::CopyPolicy::DirectCopy));
  ASSERT_TRUE(hrt::core::Memory::copy(*back, *device, hrt::core::CopyPolicy::DirectCopy));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto* copied = static_cast<const std::uint8_t*>(back->cpu());
  EXPECT_TRUE(std::all_of(copied, copied + back->bytes(), [](std::uint8_t value) -> bool { return value == 7; }));
}

} // namespace
