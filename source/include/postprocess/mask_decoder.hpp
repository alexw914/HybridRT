// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/mask_decoder.hpp
// Description: Declares instance mask decoding from coefficients and
//              prototypes, shared by the instance segmentation decoders.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_MASK_DECODER_HPP
#define HRT_POSTPROCESS_MASK_DECODER_HPP

#include <hrt/postprocess/segment.hpp>

#include <opencv2/core.hpp>

namespace hrt::postprocess {

// Keeps its scratch buffers between masks; not thread-safe.
class MaskDecoder {
public:
  struct Prototypes {
    const float* data;
    int channels;
    int height;
    int width;
  };

  // Mask = coefficients . prototypes inside the box, sampled bilinearly into
  // the box's source pixels and thresholded as a logit.
  auto make(const Prototypes& prototypes, const Vector<float>& coefficients, const core::Rect2f& net_box,
            const core::Rect2f& source_box, const preprocess::ImageTransform& transform, float mask_logit) -> Mask;

private:
  // Bilinear neighbors and the weight of the second along one axis.
  struct Tap {
    int first = 0;
    int second = 0;
    float weight = 0.0F;
  };

  static auto taps(Vector<Tap>& out, int count, double scale, double offset, int size) -> void;

  cv::Mat logits_;
  Vector<Tap> columns_;
  Vector<Tap> rows_;
  Vector<float> across_;
};

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_MASK_DECODER_HPP
