// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/metal_image_preprocessor.cpp
// Description: Implements fused Metal preprocessing on wrapped host memory.
// -----------------------------------------------------------------------------

#include "include/preprocess/metal_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "include/backend/metal/kernels/image_kernel.hpp"

namespace hrt::preprocess {

namespace {

struct Params {
  std::uint32_t sw, sh, dw, dh, rw, rh, left, top;
  std::uint32_t src_row, dst_row, dst_plane;
  std::uint32_t swap, nearest, normalization;
  Array<float, 3> fill, mean, scale;
};

// Views host memory without copying. Metal needs page-aligned bounds, so the buffer spans the enclosing pages and
// `offset` locates `data`; the kernel only touches the `bytes` that follow it.
auto wrap_host(id<MTLDevice> device, const void* data, std::size_t bytes, NSUInteger& offset) -> id<MTLBuffer> {
  const auto page = static_cast<std::uintptr_t>(NSPageSize());
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto base = address & ~(page - 1);
  offset = address - base;
  const auto length = (offset + bytes + page - 1) & ~(page - 1);
  if (length > [device maxBufferLength])
    return nil;
  return [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                   length:length
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
}

} // namespace

struct MetalImagePreprocessor::State {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  bool unified = false;
};

MetalImagePreprocessor::MetalImagePreprocessor() : state_(std::make_unique<State>()) {
  @autoreleasepool {
    state_->device = MTLCreateSystemDefaultDevice();
    if (state_->device == nil)
      return;
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:backend::metal_image_kernel()];
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
    state_->unified = [state_->device hasUnifiedMemory];
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

// The kernel works on wrapped host memory in place. Discrete GPUs would read it over the bus for every tap, so they
// leave the work to the CPU flow; the float output must be addressable in whole elements.
auto MetalImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  return op == ImageOperation::Fused && state_->unified && accepts(src.memory()->memory_type()) &&
         accepts(dst.memory()->memory_type()) && src.data_type() == core::DataType::Uint8 &&
         dst.data_type() == core::DataType::Float32 &&
         (src.format() == core::ImageFormat::InterRgb || src.format() == core::ImageFormat::InterBgr) &&
         (dst.format() == core::ImageFormat::PlanarRgb || dst.format() == core::ImageFormat::PlanarBgr) &&
         dst.stride()[1] % sizeof(float) == 0 && dst.stride()[2] % sizeof(float) == 0;
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
    const auto* input = static_cast<const std::byte*>(src.memory()->cpu(stream));
    auto* output = dst.memory()->mutable_cpu(stream);
    KZ_CHECK_RETURN(input != nullptr && output != nullptr, false, "metal: host buffers are unavailable");
    const auto src_stride = src.stride()[1];
    const auto* src_origin =
        input + static_cast<std::size_t>(from.y) * src_stride + static_cast<std::size_t>(from.x) * 3;
    const auto src_extent =
        static_cast<std::size_t>(from.height - 1) * src_stride + static_cast<std::size_t>(from.width) * 3;
    const auto dst_extent = 2 * dst.stride()[1] + static_cast<std::size_t>(dst.size().height - 1) * dst.stride()[2] +
                            static_cast<std::size_t>(dst.size().width) * sizeof(float);
    const auto capacity = std::min(static_cast<std::size_t>([state_->device maxBufferLength]),
                                   static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));
    KZ_CHECK_RETURN(src_extent <= capacity && dst_extent <= capacity, false,
                    "metal: image exceeds device buffer or kernel indexing limits");
    NSUInteger src_offset = 0;
    NSUInteger dst_offset = 0;
    id<MTLBuffer> src_buffer = wrap_host(state_->device, src_origin, src_extent, src_offset);
    id<MTLBuffer> dst_buffer = wrap_host(state_->device, output, dst_extent, dst_offset);
    KZ_CHECK_RETURN(src_buffer != nil && dst_buffer != nil, false, "metal: cannot wrap host image memory");
    Params params{};
    params.src_row = static_cast<std::uint32_t>(src_stride);
    params.dst_row = static_cast<std::uint32_t>(dst.stride()[2] / sizeof(float));
    params.dst_plane = static_cast<std::uint32_t>(dst.stride()[1] / sizeof(float));
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
      // The kernel swaps the fill as a source pixel; default_color is in input channel order.
      params.fill[c] = std::clamp(std::nearbyint(param.default_color[params.swap != 0U ? 2 - c : c]), 0.0F, 255.0F);
      params.scale[c] = 1.0F / param.std[c];
    }
    id<MTLCommandBuffer> command = [state_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    KZ_CHECK_RETURN(command != nil && encoder != nil, false, "metal: command allocation failed");
    [encoder setComputePipelineState:state_->pipeline];
    [encoder setBuffer:src_buffer offset:src_offset atIndex:0];
    [encoder setBuffer:dst_buffer offset:dst_offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(params.dw, params.dh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [command commit];
    // The wrapped host images must not be released or reused before the GPU is done with them.
    [command waitUntilCompleted];
    KZ_CHECK_RETURN([command status] == MTLCommandBufferStatusCompleted, false, "metal: execution failed: {}",
                    [[[command error] localizedDescription] UTF8String]);
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
