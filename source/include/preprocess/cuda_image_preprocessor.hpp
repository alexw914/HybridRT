// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/cuda_image_preprocessor.hpp
// Description: Declares the NPP-based CUDA image operators.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>

namespace hrt::preprocess {

// Operates on uint8 and float32 images in CUDA memory; NV12 is accepted as a
// conversion source.
class CudaImagePreprocessor final : public ImagePreprocessor {
public:
  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Cuda; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::Cuda; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override { return type == core::MemoryType::Cuda; }

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

private:
  // Changes only the data type; `dst` has the source format.
  auto convert_type(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool;

  // Changes only the channel layout; both images share a data type.
  auto convert_layout(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool;

  auto scratch(UniquePtr<core::Image>& slot, core::ImageFormat format, core::DataType data_type, core::Size size)
      -> core::Image*;

  UniquePtr<core::Image> color_;
  UniquePtr<core::Image> typed_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP
