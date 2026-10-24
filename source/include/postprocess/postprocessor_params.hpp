// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/postprocessor_params.hpp
// Description: Declares the "postprocess" config sections shared by the YOLO and
//              DETR decoders.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_POSTPROCESSOR_PARAMS_HPP
#define HRT_POSTPROCESS_POSTPROCESSOR_PARAMS_HPP

#include <hrt/postprocess/postprocessor.hpp>

namespace hrt::postprocess {

// Anchor-based heads: per anchor x, y, w, h, [theta,] objectness, classes.
// One stride and one anchor list per output head.
struct YoloV5Param {
  unsigned int class_num = 0;
  Vector<core::Point2f> strides; // Inferred in init from the input and head sizes; not a config key.
  // Per head: w0, h0, w1, h1, ...
  Vector<Vector<float>> anchors;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV5Param, class_num, anchors, confidence_threshold, iou_threshold)

// Anchor-free heads: 4 * reg_max distance bins, [theta,] classes.
struct YoloV8Param {
  unsigned int class_num = 0;
  Vector<core::Point2f> strides; // Inferred in init from the input and head sizes; not a config key.
  // Zero requests inference from output channels during initialization.
  std::size_t reg_max = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV8Param, class_num, reg_max, confidence_threshold, iou_threshold)

// YOLO26 raw heads: no distance bins (the four box channels are distances in
// cells), one output per stride.
struct Yolo26Param {
  unsigned int class_num = 0;
  Vector<core::Point2f> strides; // Inferred in init from the input and head sizes; not a config key.
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Yolo26Param, class_num, confidence_threshold, iou_threshold)

struct GeneralDetectParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralDetectParam, class_num, confidence_threshold, iou_threshold)

enum class DetrOutputFormat {
  Combined,
  Separate,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DetrOutputFormat, {
                                                   {DetrOutputFormat::Combined, "combined"},
                                                   {DetrOutputFormat::Separate, "separate"},
                                               })

enum class DetrScoreActivation {
  None,
  Sigmoid,
  Softmax,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DetrScoreActivation, {
                                                      {DetrScoreActivation::None, "none"},
                                                      {DetrScoreActivation::Sigmoid, "sigmoid"},
                                                      {DetrScoreActivation::Softmax, "softmax"},
                                                  })

struct DetrParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  DetrOutputFormat output_format = DetrOutputFormat::Combined;
  DetrScoreActivation score_activation = DetrScoreActivation::None;
  int background_class = -1;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DetrParam, class_num, confidence_threshold, output_format,
                                                score_activation, background_class)

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_POSTPROCESSOR_PARAMS_HPP
