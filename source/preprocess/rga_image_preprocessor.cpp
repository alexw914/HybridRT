// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/rga_image_preprocessor.cpp
// Description: Implements the Rockchip RGA image operators.
// -----------------------------------------------------------------------------

#include "include/preprocess/rga_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#include <im2d.hpp>

#include "include/memory/dma_buffer_memory.hpp"

namespace hrt::preprocess {

namespace {

auto rga_ok(IM_STATUS status, const char* what) -> bool {
  if (status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR)
    return true;
  KZ_LOG_ERROR("rga: {} failed: {}", what, imStrError_t(status));
  return false;
}

// RGA addresses RGB888 rows only at a pixel stride that is a multiple of 16, and RGA2 only pages below 4 GB.
auto is_inter_u8(const core::Image& image) -> bool {
  constexpr std::size_t kStrideAlignment = 16;
  const auto& stride = image.stride();
  const auto* dma_buffer = dynamic_cast<const memory::DmaBufferMemory*>(image.memory().get());
  return image.data_type() == core::DataType::Uint8 &&
         (image.format() == core::ImageFormat::InterRgb || image.format() == core::ImageFormat::InterBgr) &&
         stride[2] != 0 && stride[1] % stride[2] == 0 && (stride[1] / stride[2]) % kStrideAlignment == 0 &&
         dma_buffer != nullptr && dma_buffer->below_4gb();
}

auto rga_format(core::ImageFormat format) -> int {
  return format == core::ImageFormat::InterRgb ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888;
}

// Pixel row stride and row count from the byte strides {total, row, pixel}.
auto rga_buffer(const core::Image& image, core::Stream* stream) -> rga_buffer_t {
  const auto& stride = image.stride();
  const auto w_stride = static_cast<int>(stride[1] / stride[2]);
  const auto h_stride = static_cast<int>(stride[0] / stride[1]);
  const auto size = image.size();
  return wrapbuffer_fd_t(static_cast<int>(image.memory()->mutable_device(stream)), size.width, size.height, w_stride,
                         h_stride, rga_format(image.format()));
}

auto rga_rect(const core::Rect& rect) -> im_rect {
  return {rect.x, rect.y, rect.width, rect.height};
}

auto select_cores() -> bool {
#ifdef HRT_RGA3_ONLY
  return rga_ok(imconfig(IM_CONFIG_SCHEDULER_CORE, IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1), "imconfig");
#else
  return true;
#endif
}

} // namespace

auto RgaImagePreprocessor::accepts(core::MemoryType type) const -> bool {
  // Host memory is staged into a dma-buf: RGA2 cannot map pages above 4 GB and librga crashes on their addresses.
  return type == core::MemoryType::DmaBuffer;
}

auto RgaImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  switch (op) {
  case ImageOperation::Fill:
    return is_inter_u8(dst);
  case ImageOperation::Resize:
    return is_inter_u8(dst) && is_inter_u8(src);
  default:
    return false;
  }
}

auto RgaImagePreprocessor::adjust_roi(core::Rect& dst_roi, core::Rect& src_roi) const -> void {
  for (auto* value : {&dst_roi.x, &dst_roi.y, &src_roi.x, &src_roi.y}) {
    if (*value == 1)
      *value = 0;
  }
}

auto RgaImagePreprocessor::fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color,
                                core::Stream* stream) -> bool {
  // Packed as 0xAABBGGRR in memory order of the destination format.
  const auto r = static_cast<int>(image.format() == core::ImageFormat::InterRgb ? color[0] : color[2]);
  const auto g = static_cast<int>(color[1]);
  const auto b = static_cast<int>(image.format() == core::ImageFormat::InterRgb ? color[2] : color[0]);
  return select_cores() && rga_ok(imfill(rga_buffer(image, stream), rga_rect(roi), (b << 16) | (g << 8) | r), "imfill");
}

auto RgaImagePreprocessor::resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src,
                                  const core::Rect& src_roi, InterpolationMode mode, core::Stream* stream) -> bool {
  const auto scale_x = static_cast<double>(dst_roi.width) / src_roi.width;
  const auto scale_y = static_cast<double>(dst_roi.height) / src_roi.height;
  if (mode != InterpolationMode::Linear || scale_x > 16 || scale_y > 16 || scale_x < 0.0625 || scale_y < 0.0625)
    return cpu_.resize(dst, dst_roi, src, src_roi, mode, stream);
  return select_cores() && rga_ok(improcess(rga_buffer(src, stream), rga_buffer(dst, stream), {}, rga_rect(src_roi),
                                            rga_rect(dst_roi), {}, IM_SYNC),
                                  "improcess");
}

auto RgaImagePreprocessor::convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  return cpu_.convert(dst, src, stream);
}

auto RgaImagePreprocessor::normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean,
                                     const Array<float, 3>& stddev, core::Stream* stream) -> bool {
  return cpu_.normalize(image, roi, mean, stddev, stream);
}

} // namespace hrt::preprocess
