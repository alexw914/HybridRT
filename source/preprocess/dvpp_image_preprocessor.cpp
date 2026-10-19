// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/preprocess/dvpp_image_preprocessor.cpp
// Description: Implements the Ascend DVPP (VPC) image operators.
// -----------------------------------------------------------------------------

#include "include/preprocess/dvpp_image_preprocessor.hpp"

#include <kitzoo/log/logger.hpp>

#include <acl/dvpp/hi_dvpp.h>
#include <optional>

#include "include/backend/ascend/acl_common.hpp"

namespace hrt::preprocess {

namespace {

auto pixel_format(core::ImageFormat format) -> std::optional<hi_pixel_format> {
  switch (format) {
  case core::ImageFormat::InterRgb:
    return HI_PIXEL_FORMAT_RGB_888;
  case core::ImageFormat::InterBgr:
    return HI_PIXEL_FORMAT_BGR_888;
  case core::ImageFormat::Nv12:
    return HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  case core::ImageFormat::Nv21:
    return HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
  case core::ImageFormat::Gray:
    return HI_PIXEL_FORMAT_YUV_400;
  default:
    return std::nullopt;
  }
}

auto is_vpc_image(const core::Image& image) -> bool {
  if (image.data_type() != core::DataType::Uint8 || !pixel_format(image.format()).has_value())
    return false;
  // VPC addresses rows in 16-byte steps; anything else falls back to the CPU.
  const auto& stride = image.stride();
  const auto row = image.format() == core::ImageFormat::Gray ? stride[2] : stride[1];
  return row % 16 == 0;
}

auto picture(const core::Image& image, core::Stream* stream, bool write) -> hi_vpc_pic_info {
  const auto& stride = image.stride();
  const auto gray = image.format() == core::ImageFormat::Gray;
  const auto row = gray ? stride[2] : stride[1];
  auto rows = (gray ? stride[1] : stride[0]) / row;
  if (image.format() == core::ImageFormat::Nv12 || image.format() == core::ImageFormat::Nv21)
    rows = rows / 3 * 2;
  hi_vpc_pic_info info{};
  const auto& memory = image.memory();
  info.picture_address = reinterpret_cast<void*>(write ? memory->mutable_device(stream) : memory->device(stream));
  info.picture_buffer_size = static_cast<hi_u32>(memory->bytes());
  info.picture_width = static_cast<hi_u32>(image.size().width);
  info.picture_height = static_cast<hi_u32>(image.size().height);
  info.picture_width_stride = static_cast<hi_u32>(row);
  info.picture_height_stride = static_cast<hi_u32>(rows);
  info.picture_format = *pixel_format(image.format());
  return info;
}

auto even(int& offset, int& length) -> void {
  if (offset % 2 == 1) {
    offset -= 1;
    length += 1;
  }
  length &= ~1;
}

} // namespace

DvppImagePreprocessor::DvppImagePreprocessor() {
  if (!backend::DvppRuntime::instance().ready())
    return;
  hi_vpc_chn_attr attr{};
  hi_vpc_chn channel = 0;
  // Not fatal: without a channel supports() is false and the CPU takes over.
  const auto status = hi_mpi_vpc_sys_create_chn(&channel, &attr);
  KZ_CHECK_RETURN(status == ACL_SUCCESS, , "dvpp: hi_mpi_vpc_sys_create_chn failed: 0x{:X}",
                  static_cast<unsigned int>(status));
  channel_ = channel;
  // A channel can open where the media service cannot run, so prove VPC works once; a
  // failure here leaves supports() false and the CPU operators take over.
  const core::Image src(core::MemoryType::Acl, core::ImageFormat::InterBgr, core::DataType::Uint8, {64, 64});
  core::Image dst(core::MemoryType::Acl, core::ImageFormat::InterBgr, core::DataType::Uint8, {32, 32});
  const ImageTransform transform{.dst_roi = {0, 0, 32, 32}, .src_roi = {0, 0, 64, 64}, .input_size = {32, 32}};
  if (!src.valid() || !dst.valid() || !fused(dst, transform, src, ImagePreprocessorParam{}, nullptr)) {
    KZ_LOG_WARN("dvpp: VPC is unusable on this device; preprocessing falls back to the CPU");
    static_cast<void>(hi_mpi_vpc_destroy_chn(channel_));
    channel_ = -1;
  }
}

DvppImagePreprocessor::~DvppImagePreprocessor() {
  if (channel_ >= 0 && hi_mpi_vpc_destroy_chn(channel_) != ACL_SUCCESS)
    KZ_LOG_ERROR("dvpp: hi_mpi_vpc_destroy_chn failed");
}

auto DvppImagePreprocessor::supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool {
  return op == ImageOperation::Fused && channel_ >= 0 && is_vpc_image(dst) && is_vpc_image(src);
}

auto DvppImagePreprocessor::adjust_roi(core::Rect& dst_roi, core::Rect& src_roi, const core::Image& dst,
                                       const core::Image& src) const -> void {
  if (!supports(ImageOperation::Fused, dst, src))
    return;
  even(src_roi.x, src_roi.width);
  even(src_roi.y, src_roi.height);
  even(dst_roi.x, dst_roi.width);
  even(dst_roi.y, dst_roi.height);
}

auto DvppImagePreprocessor::fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                                  const ImagePreprocessorParam& param, core::Stream* stream) -> bool {
  if (param.interpolation_mode == InterpolationMode::Cubic) {
    KZ_LOG_ERROR("dvpp: cubic interpolation is not supported");
    return false;
  }
  const auto& from = transform.src_roi;
  const auto& to = transform.dst_roi;
  auto source = picture(src, stream, false);
  hi_vpc_crop_resize_border_region region{};
  region.dest_pic_info = picture(dst, stream, true);
  region.crop_region = {static_cast<hi_u32>(from.y), static_cast<hi_u32>(from.x), static_cast<hi_u32>(from.width),
                        static_cast<hi_u32>(from.height)};
  region.resize_info = {static_cast<hi_u32>(to.width), static_cast<hi_u32>(to.height),
                        param.interpolation_mode == InterpolationMode::Nearest ? 1U : 0U};
  region.dest_top_offset = static_cast<hi_u32>(to.y);
  region.dest_left_offset = static_cast<hi_u32>(to.x);
  region.border_type = HI_BORDER_CONSTANT;
  for (std::size_t i = 0; i < 3; ++i)
    region.scalar_value.val[i] = static_cast<double>(param.default_color[i]);
  region.scalar_value.val[3] = 0.0;
  hi_u32 task = 0;
  auto status = hi_mpi_vpc_crop_resize_make_border(channel_, &source, &region, 1, &task, 0);
  KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "dvpp: hi_mpi_vpc_crop_resize_make_border failed: 0x{:X}",
                  static_cast<unsigned int>(status));
  status = hi_mpi_vpc_get_process_result(channel_, task, -1);
  KZ_CHECK_RETURN(status == ACL_SUCCESS, false, "dvpp: hi_mpi_vpc_get_process_result failed: 0x{:X}",
                  static_cast<unsigned int>(status));
  return true;
}

// Only the fused step runs on DVPP; the preprocessor never selects these.
auto DvppImagePreprocessor::fill(core::Image& /*image*/, const core::Rect& /*roi*/, const Array<float, 3>& /*color*/,
                                 core::Stream* /*stream*/) -> bool {
  return false;
}

auto DvppImagePreprocessor::resize(core::Image& /*dst*/, const core::Rect& /*dst_roi*/, const core::Image& /*src*/,
                                   const core::Rect& /*src_roi*/, InterpolationMode /*mode*/, core::Stream* /*stream*/)
    -> bool {
  return false;
}

auto DvppImagePreprocessor::convert(core::Image& /*dst*/, const core::Image& /*src*/, core::Stream* /*stream*/)
    -> bool {
  return false;
}

auto DvppImagePreprocessor::normalize(core::Image& /*image*/, const core::Rect& /*roi*/,
                                      const Array<float, 3>& /*mean*/, const Array<float, 3>& /*stddev*/,
                                      core::Stream* /*stream*/) -> bool {
  return false;
}

} // namespace hrt::preprocess
