// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/params.hpp
// Description: Declares the "postprocess" config sections of the postprocessors and
//              their factories.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_PARAMS_HPP
#define HRT_POSTPROCESS_PARAMS_HPP

#include <hrt/postprocess/classify.hpp>
#include <hrt/postprocess/detect.hpp>
#include <hrt/postprocess/embedding.hpp>
#include <hrt/postprocess/keypoint.hpp>
#include <hrt/postprocess/ocr.hpp>
#include <hrt/postprocess/rotated.hpp>
#include <hrt/postprocess/segment.hpp>

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

// Built-in postprocessor of `type` for each task, or nullptr.
auto create_detect(std::string_view type) -> UniquePtr<DetectPostprocessor>;

auto create_rotated(std::string_view type) -> UniquePtr<RotatedPostprocessor>;

auto create_classify(std::string_view type) -> UniquePtr<ClassifyPostprocessor>;

auto create_embedding(std::string_view type) -> UniquePtr<EmbeddingPostprocessor>;
auto create_embedding() -> UniquePtr<EmbeddingPostprocessor>;

auto create_keypoint(std::string_view type) -> UniquePtr<KeypointPostprocessor>;

auto create_segment(std::string_view type) -> UniquePtr<SegmentPostprocessor>;

auto create_ocr(std::string_view type) -> UniquePtr<OcrPostprocessor>;
auto create_ocr() -> UniquePtr<OcrPostprocessor>;

auto create_yolov5_detect() -> UniquePtr<DetectPostprocessor>;

auto create_yolov8_detect() -> UniquePtr<DetectPostprocessor>;

auto create_yolo26_detect() -> UniquePtr<DetectPostprocessor>;

auto create_general_detect() -> UniquePtr<DetectPostprocessor>;

auto create_detr_detect() -> UniquePtr<DetectPostprocessor>;

auto create_e2e_detect() -> UniquePtr<DetectPostprocessor>;

auto create_yolov5_rotated() -> UniquePtr<RotatedPostprocessor>;

auto create_yolov8_rotated() -> UniquePtr<RotatedPostprocessor>;

auto create_yolo26_rotated() -> UniquePtr<RotatedPostprocessor>;

auto create_general_rotated() -> UniquePtr<RotatedPostprocessor>;

auto create_e2e_rotated() -> UniquePtr<RotatedPostprocessor>;

auto create_general_classify() -> UniquePtr<ClassifyPostprocessor>;

auto create_general_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto create_general_yolo_pose() -> UniquePtr<KeypointPostprocessor>;

auto create_yolov8_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto create_yolo26_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto create_rtmo_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto create_e2e_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto create_general_yolo_segment() -> UniquePtr<SegmentPostprocessor>;

auto create_yolov8_segment() -> UniquePtr<SegmentPostprocessor>;

auto create_yolo26_segment() -> UniquePtr<SegmentPostprocessor>;

auto create_e2e_segment() -> UniquePtr<SegmentPostprocessor>;

auto create_semantic_segment() -> UniquePtr<SegmentPostprocessor>;

auto nms_to_source(DetectResultMap& result, const Vector<float>& iou_threshold,
                   const preprocess::ImageTransform& transform) -> void;

// Maps network-space rotated boxes back to the source; fails without uniform
// scaling.
auto rotated_to_source(RotatedResultMap& result, const preprocess::ImageTransform& transform) -> bool;

auto nms_rotated_to_source(RotatedResultMap& result, const Vector<float>& iou_threshold,
                           const preprocess::ImageTransform& transform) -> bool;

auto resolve_yolov5_heads(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

auto resolve_yolov8_heads(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

// resolve_yolov8_heads() for YOLO26 heads (reg_max 1); fills strides and per-class thresholds.
auto resolve_yolo26_heads(Yolo26Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_PARAMS_HPP
