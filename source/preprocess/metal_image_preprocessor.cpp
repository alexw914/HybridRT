// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/metal_image_preprocessor.cpp
// Description: Implements fused Metal preprocessing with reusable host staging.
// -----------------------------------------------------------------------------

#include "include/preprocess/metal_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "include/backend/metal/image_kernel.hpp"

namespace hrt::preprocess {

namespace {

struct Params {
  std::uint32_t sw, sh, dw, dh, rw, rh, left, top;
  std::uint32_t swap, nearest, normalization;
  Array<float, 3> fill, mean, scale;
};

static_assert(sizeof(Params) == 80);

auto ensure_buffer(id<MTLDevice> device, id<MTLBuffer> __strong& buffer, std::size_t bytes) -> bool {
  if (buffer == nil || [buffer length] < bytes)
    buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  KZ_CHECK_RETURN(buffer != nil, false, "metal: cannot allocate {} staging bytes", bytes);
  return true;
}

} // namespace

struct MetalImagePreprocessor::State {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> input = nil;
  id<MTLBuffer> output = nil;
};

MetalImagePreprocessor::MetalImagePreprocessor() : state_(std::make_unique<State>()) {
  @autoreleasepool {
    state_->device = MTLCreateSystemDefaultDevice();
    if (state_->device == nil)
      return;
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:backend::kMetalImageKernel];
    id<MTLLibrary> library = [state_->device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      KZ_LOG_WARN("metal: shader compilation failed: {}", [[error localizedDescription] UTF8String]);
      return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"preprocess"];
    state_->pipeline = [state_->device newComputePipelineStateWithFunction:function error:&error];
    if (state_->pipeline == nil) {
      KZ_LOG_WARN("metal: pipeline creation failed: {}", [[error localizedDescription] UTF8String]);
      return;
    }
    state_->queue = [state_->device newCommandQueue];
  }
}

MetalImagePreprocessor::~MetalImagePreprocessor() = default;

auto MetalImagePreprocessor::create() -> UniquePtr<ImagePreprocessor> {
  auto processor = std::make_unique<MetalImagePreprocessor>();
  if (processor->state_->pipeline == nil || processor->state_->queue == nil)
    return nullptr;
  return processor;
}

auto MetalImagePreprocessor::type() const -> ImagePreprocessorType {
  return ImagePreprocessorType::Metal;
}

auto MetalImagePreprocessor::memory_type() const -> core::MemoryType {
  return core::MemoryType::Cpu;
}

auto MetalImagePreprocessor::accepts(core::MemoryType type) const -> bool {
  return type == core::MemoryType::Cpu;
}

auto MetalImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  return op == ImageOperation::Fused && accepts(src.memory()->memory_type()) && accepts(dst.memory()->memory_type()) &&
         src.data_type() == core::DataType::Uint8 && dst.data_type() == core::DataType::Float32 &&
         (src.format() == core::ImageFormat::InterRgb || src.format() == core::ImageFormat::InterBgr) &&
         (dst.format() == core::ImageFormat::PlanarRgb || dst.format() == core::ImageFormat::PlanarBgr);
}

auto MetalImagePreprocessor::supports_fused(const core::Image& dst, const core::Image& src,
                                            const ImagePreprocessorParam& param) const -> bool {
  return supports(ImageOperation::Fused, dst, src) && param.interpolation_mode != InterpolationMode::Cubic;
}

auto MetalImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                   const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  @autoreleasepool {
    const auto& from = transform.src_roi;
    const auto& to = transform.dst_roi;
    KZ_CHECK_RETURN(!from.empty() && !to.empty(), false, "metal: empty resize region");
    KZ_CHECK_RETURN(from == core::intersect(from, src.rect()) && to == core::intersect(to, dst.rect()), false,
                    "metal: resize region is outside its image");
    const auto src_row = static_cast<std::size_t>(from.width) * 3;
    const auto dst_row = static_cast<std::size_t>(dst.size().width) * sizeof(float);
    const auto dst_plane = dst_row * static_cast<std::size_t>(dst.size().height);
    const auto capacity = std::min(static_cast<std::size_t>([state_->device maxBufferLength]),
                                   static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
    KZ_CHECK_RETURN(src_row <= capacity / static_cast<std::size_t>(from.height) && dst_plane <= capacity / 3, false,
                    "metal: image exceeds device buffer or kernel indexing limits");
    if (!ensure_buffer(state_->device, state_->input, src_row * static_cast<std::size_t>(from.height)) ||
        !ensure_buffer(state_->device, state_->output, dst_plane * 3))
      return false;
    const auto* input = static_cast<const std::byte*>(src.memory()->cpu(stream));
    auto* output = static_cast<std::byte*>(dst.memory()->mutable_cpu(stream));
    KZ_CHECK_RETURN(input != nullptr && output != nullptr, false, "metal: host buffers are unavailable");
    auto* staged = static_cast<std::byte*>([state_->input contents]);
    for (int y = 0; y < from.height; ++y)
      std::memcpy(staged + static_cast<std::size_t>(y) * src_row,
                  input + static_cast<std::size_t>(from.y + y) * src.stride()[1] + static_cast<std::size_t>(from.x) * 3,
                  src_row);
    Params params{};
    params.sw = static_cast<std::uint32_t>(from.width);
    params.sh = static_cast<std::uint32_t>(from.height);
    params.dw = static_cast<std::uint32_t>(dst.size().width);
    params.dh = static_cast<std::uint32_t>(dst.size().height);
    params.rw = static_cast<std::uint32_t>(to.width);
    params.rh = static_cast<std::uint32_t>(to.height);
    params.left = static_cast<std::uint32_t>(to.x);
    params.top = static_cast<std::uint32_t>(to.y);
    params.swap = (src.format() == core::ImageFormat::InterRgb) != (dst.format() == core::ImageFormat::PlanarRgb);
    params.nearest = param.interpolation_mode == InterpolationMode::Nearest;
    params.normalization = param.normalize_type == NormalizeType::Full  ? 1U
                           : param.normalize_type == NormalizeType::Roi ? 2U
                                                                        : 0U;
    params.mean = param.mean;
    for (std::size_t c = 0; c < 3; ++c) {
      params.fill[c] = std::clamp(std::nearbyint(param.default_color[c]), 0.0F, 255.0F);
      params.scale[c] = 1.0F / param.std[c];
    }
    id<MTLCommandBuffer> command = [state_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    KZ_CHECK_RETURN(command != nil && encoder != nil, false, "metal: command allocation failed");
    [encoder setComputePipelineState:state_->pipeline];
    [encoder setBuffer:state_->input offset:0 atIndex:0];
    [encoder setBuffer:state_->output offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(params.dw, params.dh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [command commit];
    // Host consumers and pool reuse must wait for the GPU, even on unified memory.
    [command waitUntilCompleted];
    KZ_CHECK_RETURN([command status] == MTLCommandBufferStatusCompleted, false, "metal: execution failed: {}",
                    [[[command error] localizedDescription] UTF8String]);
    const auto* result = static_cast<const std::byte*>([state_->output contents]);
    for (std::size_t c = 0; c < 3; ++c) {
      for (int y = 0; y < dst.size().height; ++y)
        std::memcpy(output + c * dst.stride()[1] + static_cast<std::size_t>(y) * dst.stride()[2],
                    result + c * dst_plane + static_cast<std::size_t>(y) * dst_row, dst_row);
    }
    return true;
  }
}

auto MetalImagePreprocessor::fill(core::Image&, const core::Rect&, const Array<float, 3>&, core::Stream*) -> bool {
  return false;
}

auto MetalImagePreprocessor::resize(core::Image&, const core::Rect&, const core::Image&, const core::Rect&,
                                    InterpolationMode, core::Stream*) -> bool {
  return false;
}

auto MetalImagePreprocessor::convert(core::Image&, const core::Image&, core::Stream*) -> bool {
  return false;
}

auto MetalImagePreprocessor::normalize(core::Image&, const core::Rect&, const Array<float, 3>&, const Array<float, 3>&,
                                       core::Stream*) -> bool {
  return false;
}

} // namespace hrt::preprocess
