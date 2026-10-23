// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/preprocess/metal_image_preprocessor.hpp
// Description: Declares synchronous Metal preprocessing of host images.
// -----------------------------------------------------------------------------

#ifndef HRT_PREPROCESS_METAL_IMAGE_PREPROCESSOR_HPP
#define HRT_PREPROCESS_METAL_IMAGE_PREPROCESSOR_HPP

#include <hrt/preprocess/image_preprocessor.hpp>

namespace hrt::preprocess {

class MetalImagePreprocessor final : public ImagePreprocessor {
public:
  static auto create() -> UniquePtr<ImagePreprocessor>;

  MetalImagePreprocessor();

  ~MetalImagePreprocessor() override;

  auto fill(core::Image& image, const core::Rect& roi, const Array<float, 3>& color, core::Stream* stream)
      -> bool override;

  auto resize(core::Image& dst, const core::Rect& dst_roi, const core::Image& src, const core::Rect& src_roi,
              InterpolationMode mode, core::Stream* stream) -> bool override;

  auto convert(core::Image& dst, const core::Image& src, core::Stream* stream) -> bool override;

  auto normalize(core::Image& image, const core::Rect& roi, const Array<float, 3>& mean, const Array<float, 3>& stddev,
                 core::Stream* stream) -> bool override;

  auto fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
             const ImagePreprocessorParam& param, core::Stream* stream) -> bool override;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override;

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override;

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override;

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

private:
  KZ_NODISCARD auto supports_fused(const core::Image& dst, const core::Image& src,
                                   const ImagePreprocessorParam& param) const -> bool override;

  struct State;
  UniquePtr<State> state_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_METAL_IMAGE_PREPROCESSOR_HPP
