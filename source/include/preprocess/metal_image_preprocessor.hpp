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

  ~MetalImagePreprocessor() override;

  MetalImagePreprocessor(const MetalImagePreprocessor&) = delete;
  auto operator=(const MetalImagePreprocessor&) -> MetalImagePreprocessor& = delete;
  MetalImagePreprocessor(MetalImagePreprocessor&&) = delete;
  auto operator=(MetalImagePreprocessor&&) -> MetalImagePreprocessor& = delete;

  KZ_NODISCARD auto type() const -> ImagePreprocessorType override { return ImagePreprocessorType::Metal; }

  KZ_NODISCARD auto memory_type() const -> core::MemoryType override { return core::MemoryType::Cpu; }

  KZ_NODISCARD auto accepts(core::MemoryType type) const -> bool override { return type == core::MemoryType::Cpu; }

  KZ_NODISCARD auto supports(ImageOperation op, const core::Image& dst, const core::Image& src) const -> bool override;

protected:
  auto fused(core::Image& dst, const ImageTransform& transform, const core::Image& src,
             const ImagePreprocessorParam& param, core::Stream* stream) -> bool override;

  auto fused_batch(std::span<core::Image* const> dsts, std::span<const ImageTransform> transforms,
                   std::span<const core::Image* const> srcs, const ImagePreprocessorParam& param, core::Stream* stream)
      -> bool override;

private:
  MetalImagePreprocessor();

  KZ_NODISCARD auto supports_fused(const core::Image& dst, const core::Image& src,
                                   const ImagePreprocessorParam& param) const -> bool override;

  struct State;
  UniquePtr<State> state_;
};

} // namespace hrt::preprocess

#endif // HRT_PREPROCESS_METAL_IMAGE_PREPROCESSOR_HPP
