// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/preprocess/image_preprocessor.hpp
// Description: Declares image preprocessing parameters and the per-device
//              image preprocessor.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_IMAGE_PREPROCESSOR_HPP

#include <hrt/core/image.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/json/reader.hpp>

#include <optional>

namespace hrt::preprocess {

enum class ImagePreprocessorType {
  Auto,
  Cpu,
  Cuda,
  Rga,
  Dvpp,
  Metal,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ImagePreprocessorType, {
                                                        {ImagePreprocessorType::Auto, "auto"},
                                                        {ImagePreprocessorType::Cpu, "cpu"},
                                                        {ImagePreprocessorType::Cuda, "cuda"},
                                                        {ImagePreprocessorType::Rga, "rga"},
                                                        {ImagePreprocessorType::Dvpp, "dvpp"},
                                                        {ImagePreprocessorType::Metal, "metal"},
                                                    })

enum class InterpolationMode {
  Linear,
  Nearest,
  Cubic,
};

NLOHMANN_JSON_SERIALIZE_ENUM(InterpolationMode, {
                                                    {InterpolationMode::Linear, "linear"},
                                                    {InterpolationMode::Nearest, "nearest"},
                                                    {InterpolationMode::Cubic, "cubic"},
                                                })

enum class ResizeType {
  Normal,
  KeepRatio,
  CenterCrop,
  LetterBox,
  // Grow the source rect to the input aspect ratio, scale by pad_scale and
  // clip to the image.
  PadOutside,
  TextLine, // Fixed height, proportional width capped at input width, right padding.
};

NLOHMANN_JSON_SERIALIZE_ENUM(ResizeType, {
                                             {ResizeType::Normal, "normal"},
                                             {ResizeType::KeepRatio, "keep_ratio"},
                                             {ResizeType::CenterCrop, "center_crop"},
                                             {ResizeType::LetterBox, "letter_box"},
                                             {ResizeType::LetterBox, "center_keep_ratio"},
                                             {ResizeType::PadOutside, "pad_outside"},
                                             {ResizeType::TextLine, "text_line"},
                                         })

enum class NormalizeType {
  Full,
  Roi,
  None,
};

NLOHMANN_JSON_SERIALIZE_ENUM(NormalizeType, {
                                                {NormalizeType::Full, "full"},
                                                {NormalizeType::Roi, "roi"},
                                                {NormalizeType::None, "none"},
                                            })

struct ImagePreprocessorParam {
  ImagePreprocessorType device = ImagePreprocessorType::Auto;
  InterpolationMode interpolation_mode = InterpolationMode::Linear;
  ResizeType resize_type = ResizeType::Normal;
  core::ImageFormat dst_image_format = core::ImageFormat::PlanarRgb;
  NormalizeType normalize_type = NormalizeType::Full;
  Array<float, 3> mean{0.0F, 0.0F, 0.0F};
  Array<float, 3> std{1.0F, 1.0F, 1.0F};
  Array<float, 3> default_color{0.0F, 0.0F, 0.0F};
  int resize_short_size = -1;
  float pad_scale = 1.0F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ImagePreprocessorParam, device, interpolation_mode, resize_type,
                                                dst_image_format, normalize_type, mean, std, default_color,
                                                resize_short_size, pad_scale)

// Where the source region landed in the network input.
struct ImageTransform {
  core::Rect dst_roi;
  core::Rect src_roi;
  core::Size input_size;

  // Maps network-input coordinates back to the source image.
  KZ_NODISCARD auto to_source(const core::Rect2f& rect) const -> core::Rect2f;

  KZ_NODISCARD auto to_source(const core::Point2f& point) const -> core::Point2f;

  // to_source() clipped to src_roi; empty when the box leaves it.
  KZ_NODISCARD auto clip_to_source(const core::Rect2f& rect) const -> core::Rect2f;
};

auto calc_resize_position(core::Rect& dst_roi, core::Rect& src_roi, const core::Rect& dst_rect,
                          const core::Rect& src_rect, core::Size src_size, const ImagePreprocessorParam& param) -> bool;

enum class ImageOperation {
  Fill,
  Resize,
  Convert,
  Normalize,
  Fused,
};

// Image operators of one device, plus the shared flow in process(): it moves
// data the device cannot access and runs unsupported steps on the CPU.
// External operator libraries plug in as subclasses. Not thread-safe.
class ImagePreprocessor {
public:
  // Auto picks the device of `target` memory. Returns nullptr when `type` is
  // not compiled in.
  static auto create(ImagePreprocessorType type, core::MemoryType target = core::MemoryType::Cpu)
      -> UniquePtr<ImagePreprocessor>;

  virtual ~ImagePreprocessor();

  ImagePreprocessor(const ImagePreprocessor&) = delete;
  auto operator=(const ImagePreprocessor&) -> ImagePreprocessor& = delete;
  ImagePreprocessor(ImagePreprocessor&&) = delete;
  auto operator=(ImagePreprocessor&&) -> ImagePreprocessor& = delete;

  // An empty `src_rect` selects the whole source; `param.device` is ignored.
  // Fails when the region read leaves the source: only pad_outside clips
  // `src_rect`, so clip boxes for the others with core::intersect.
  auto process(core::Image& dst, const core::Image& src, const core::Rect& src_rect,
               const ImagePreprocessorParam& param, core::Stream* stream = nullptr) -> std::optional<ImageTransform>;

  KZ_NODISCARD virtual auto type() const -> ImagePreprocessorType = 0;

  // Memory type of scratch images.
  KZ_NODISCARD virtual auto memory_type() const -> core::MemoryType = 0;

  KZ_NODISCARD virtual auto accepts(core::MemoryType type) const -> bool = 0;

  // For Fill and Normalize, `dst` and `src` are the same image.
  KZ_NODISCARD virtual auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const
      -> bool = 0;

  // Narrows the regions to what this device can address, for the images it would run on.
  virtual auto adjust_roi(core::Rect& dst_roi, core::Rect& src_roi, const core::Image& dst,
                          const core::Image& src) const -> void;

  virtual auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool = 0;

  // Within one layout family; swaps RGB/BGR order when the formats differ.
  virtual auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
                      InterpolationMode mode, core::Stream* stream) -> bool = 0;

  // Same size, any format and data type.
  virtual auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool = 0;

  // In place on a float image: x = (x - mean) / std.
  virtual auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean,
                         const Array<float, 3>& stddev, core::Stream* stream) -> bool = 0;

  virtual auto fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
                     const ImagePreprocessorParam& param, core::Stream* stream) -> bool;

protected:
  ImagePreprocessor();

private:
  virtual auto supports_fused(const core::Image& dst, const core::Image& src, const ImagePreprocessorParam& param) const
      -> bool;

  virtual auto convert_normalize(core::Image& dst, const core::Image& src, const core::Rect& roi,
                                 const Array<float, 3>& mean, const Array<float, 3>& stddev, core::Stream* stream)
      -> bool;

  auto cpu() -> ImagePreprocessor*;

  auto scratch(UniquePtr<core::Image>& slot, core::MemoryType memory_type, core::ImageFormat format,
               core::DataType data_type, core::Size size) -> core::Image*;

  auto run(core::Image& dst, const core::Image& src, const ImageTransform& transform,
           const ImagePreprocessorParam& param, core::Stream* stream) -> bool;

  UniquePtr<ImagePreprocessor> cpu_;
  UniquePtr<core::Image> staged_src_;
  UniquePtr<core::Image> staged_dst_;
  UniquePtr<core::Image> host_src_;
  UniquePtr<core::Image> host_dst_;
  UniquePtr<core::Image> converted_;
  UniquePtr<core::Image> resized_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_IMAGE_PREPROCESSOR_HPP
