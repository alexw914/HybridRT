// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/mask_decoder.hpp
// Description: Declares instance mask decoding from coefficients and
//              prototypes, shared by the instance segmentation decoders.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_MASK_DECODER_HPP
#define HRT_POSTPROCESS_MASK_DECODER_HPP

#include <hrt/engine/engine.hpp>
#include <hrt/postprocess/segment.hpp>

#include <opencv2/core.hpp>
#include <optional>

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

// Outputs of a raw per-stride segmentation model: `heads` in stride order and
// the prototype output [1, masks, mh, mw].
struct RawSegmentOutputs {
  Vector<std::size_t> heads;
  std::size_t prototypes = 0;
  std::size_t masks = 0;
};

// Finds the prototypes among the outputs (the one whose channel count makes
// every other output box_channels + classes + masks) and checks one head per
// stride against the input size; logs and returns nullopt when they do not fit.
auto raw_segment_outputs(const engine::Engine& engine, const Vector<float>& strides, std::size_t box_channels,
                         std::size_t classes, const char* name) -> std::optional<RawSegmentOutputs>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_MASK_DECODER_HPP
