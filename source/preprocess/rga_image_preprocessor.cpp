// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/rga_image_preprocessor.cpp
// Description: Implements the Rockchip RGA image operators.
// -----------------------------------------------------------------------------

#include "include/preprocess/rga_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#include <im2d.hpp>

namespace hrt::preprocess {

namespace {

auto rga_ok(IM_STATUS status, const char* what) -> bool {
  if (status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR)
    return true;
  KZ_LOG_ERROR("rga: {} failed: {}", what, imStrError_t(status));
  return false;
}

constexpr int kMinSize = 128;
constexpr int kMaxSrcSize = 8176;
constexpr int kMaxDstSize = 8128;

// RGA3 addresses RGB888 rows only at a pixel stride that is a multiple of 16.
auto is_inter_u8(const core::Image& image) -> bool {
  constexpr std::size_t kStrideAlignment = 16;
  const auto& stride = image.stride();
  return image.data_type() == core::DataType::Uint8 &&
         (image.format() == core::ImageFormat::InterRgb || image.format() == core::ImageFormat::InterBgr) &&
         stride[2] != 0 && stride[1] % stride[2] == 0 && (stride[1] / stride[2]) % kStrideAlignment == 0 &&
         image.memory()->memory_type() == core::MemoryType::DmaBuffer;
}

auto is_yuv_u8(const core::Image& image) -> bool {
  constexpr std::size_t kStrideAlignment = 16;
  const auto size = image.size();
  return image.data_type() == core::DataType::Uint8 &&
         (image.format() == core::ImageFormat::Nv12 || image.format() == core::ImageFormat::Nv21) &&
         image.memory()->memory_type() == core::MemoryType::DmaBuffer && size.width >= kMinSize &&
         size.height >= kMinSize && size.width <= kMaxDstSize && size.height <= kMaxDstSize && size.width % 2 == 0 &&
         size.height % 2 == 0 && image.stride()[1] % kStrideAlignment == 0;
}

auto rga_format(core::ImageFormat format) -> int {
  switch (format) {
  case core::ImageFormat::InterRgb:
    return RK_FORMAT_RGB_888;
  case core::ImageFormat::InterBgr:
    return RK_FORMAT_BGR_888;
  case core::ImageFormat::Nv12:
    return RK_FORMAT_YCbCr_420_SP;
  case core::ImageFormat::Nv21:
    return RK_FORMAT_YCrCb_420_SP;
  default:
    return RK_FORMAT_UNKNOWN;
  }
}

// Pixel row stride and row count from the byte strides.
auto rga_buffer(const core::Image& image, core::Stream* stream, bool write) -> rga_buffer_t {
  const auto& stride = image.stride();
  const auto size = image.size();
  const bool yuv = image.format() == core::ImageFormat::Nv12 || image.format() == core::ImageFormat::Nv21;
  const auto w_stride = static_cast<int>(yuv ? stride[1] : stride[1] / stride[2]);
  const auto h_stride = yuv ? size.height : static_cast<int>(stride[0] / stride[1]);
  const auto fd = write ? image.memory()->mutable_device(stream) : image.memory()->device(stream);
  return wrapbuffer_fd_t(static_cast<int>(fd), size.width, size.height, w_stride, h_stride, rga_format(image.format()));
}

auto rga_rect(const core::Rect& rect) -> im_rect {
  return {rect.x, rect.y, rect.width, rect.height};
}

auto select_cores() -> bool {
  return rga_ok(imconfig(IM_CONFIG_SCHEDULER_CORE, IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1), "imconfig");
}

} // namespace

auto RgaImagePreprocessor::accepts(core::MemoryType type) const -> bool {
  return type == core::MemoryType::DmaBuffer;
}

auto RgaImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  if (op == ImageOperation::Resize)
    return is_inter_u8(dst) && is_inter_u8(src);
  if (op == ImageOperation::Convert)
    return is_yuv_u8(src) && is_inter_u8(dst) && src.size() == dst.size();
  return false;
}

auto RgaImagePreprocessor::adjust_roi(core::Rect& dst_roi, core::Rect& src_roi, const core::Image& /*dst*/,
                                      const core::Image& /*src*/) const -> void {
  for (auto* value : {&dst_roi.x, &dst_roi.y, &src_roi.x, &src_roi.y}) {
    if (*value == 1)
      *value = 0;
  }
}

auto RgaImagePreprocessor::resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src,
                                  const core::Rect& src_roi, InterpolationMode mode, core::Stream* stream) -> bool {
  const auto scale_x = static_cast<double>(dst_roi.width) / src_roi.width;
  const auto scale_y = static_cast<double>(dst_roi.height) / src_roi.height;
  const bool unsupported = mode != InterpolationMode::Linear || scale_x > 8 || scale_y > 8 || scale_x < 0.125 ||
                           scale_y < 0.125 || src_roi.width < kMinSize || src_roi.height < kMinSize ||
                           dst_roi.width < kMinSize || dst_roi.height < kMinSize || src.size().width > kMaxSrcSize ||
                           src.size().height > kMaxSrcSize || dst.size().width > kMaxDstSize ||
                           dst.size().height > kMaxDstSize;
  if (unsupported)
    return cpu_resize(dst, dst_roi, src, src_roi, mode, stream);
  return select_cores() && rga_ok(improcess(rga_buffer(src, stream, false), rga_buffer(dst, stream, true), {},
                                            rga_rect(src_roi), rga_rect(dst_roi), {}, IM_SYNC),
                                  "improcess");
}

auto RgaImagePreprocessor::convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool {
  return select_cores() &&
         rga_ok(imcvtcolor(rga_buffer(src, stream, false), rga_buffer(dst, stream, true), rga_format(src.format()),
                           rga_format(dst.format()), IM_YUV_TO_RGB_BT601_LIMIT, IM_SYNC),
                "imcvtcolor");
}

} // namespace hrt::preprocess
