// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/cuda_image_preprocessor.hpp
// Description: Declares the CUDA image preprocessor, a fused kernel.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

#include <kitzoo/core/macro.hpp>

#include "include/backend/cuda/kernels/image_kernels.hpp"

namespace hrt::preprocess {

// Runs every supported conversion in one fused kernel on CUDA memory; it has
// no step operators, so the shared flow leaves other conversions (gray or
// planar inputs and sources, other source formats) to the CPU.
class CudaImagePreprocessor final : public ImagePreprocessor {
public:
  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

  // Interleaved BGR/RGB or NV12/NV21/I420/YV12 uint8 to a 3-channel uint8 or
  // float32 input in one kernel: color conversion, resize (linear, nearest or
  // cubic), padding, channel order and normalization.
  auto fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
             const ImagePreprocessorParam& param, core::Stream* stream) -> bool override;

  // All samples in one launch of up to backend::kPreprocessParamsPerLaunch params per kernel.
  auto fused_batch(std::span<core::Image* const> dsts, std::span<const ImageTransform> transforms,
                   std::span<const core::Image* const> srcs, const ImagePreprocessorParam& param, core::Stream* stream)
      -> bool override;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Cuda; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::Cuda; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override { return type == core::MemoryType::Cuda; }

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

private:
  KZ_NODISCARD auto supports_fused(const core::Image& dst, const core::Image& src,
                                   const ImagePreprocessorParam& param) const -> bool override;

  Vector<backend::PreprocessParam> params_; // Reused parameter list of fused_batch().
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_CUDA_IMAGE_PREPROCESSOR_HPP
