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
#include <hrt/postprocess/keypoint.hpp>
#include <hrt/postprocess/rotated.hpp>
#include <hrt/postprocess/segment.hpp>

namespace hrt::postprocess {

// Anchor-based heads: per anchor x, y, w, h, [theta,] objectness, classes.
struct YoloV5Param {
  unsigned int class_num = 0;
  unsigned int head_num = 0;
  Vector<int> anchors_per_head;
  Vector<float> strides_per_head;
  // Per head: w0, h0, w1, h1, ...
  Vector<Vector<float>> anchors;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV5Param, class_num, head_num, anchors_per_head, strides_per_head,
                                                anchors, confidence_threshold, iou_threshold)

// Anchor-free heads: 4 * reg_max distance bins, [theta,] classes.
struct YoloV8Param {
  unsigned int class_num = 0;
  Vector<float> strides;
  std::size_t reg_max = 16;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(YoloV8Param, class_num, strides, reg_max, confidence_threshold,
                                                iou_threshold)

struct GeneralDetectParam {
  unsigned int class_num = 0;
  Vector<float> confidence_threshold;
  Vector<float> iou_threshold;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralDetectParam, class_num, confidence_threshold, iou_threshold)

// Built-in postprocessor of `type` for each task, or nullptr.
auto make_detect(std::string_view type) -> UniquePtr<DetectPostprocessor>;

auto make_rotated(std::string_view type) -> UniquePtr<RotatedPostprocessor>;

auto make_classify(std::string_view type) -> UniquePtr<ClassifyPostprocessor>;

auto make_keypoint(std::string_view type) -> UniquePtr<KeypointPostprocessor>;

auto make_segment(std::string_view type) -> UniquePtr<SegmentPostprocessor>;

auto make_yolov5_detect() -> UniquePtr<DetectPostprocessor>;

auto make_yolov8_detect() -> UniquePtr<DetectPostprocessor>;

auto make_general_detect() -> UniquePtr<DetectPostprocessor>;

auto make_yolov5_rotated() -> UniquePtr<RotatedPostprocessor>;

auto make_yolov8_rotated() -> UniquePtr<RotatedPostprocessor>;

auto make_general_rotated() -> UniquePtr<RotatedPostprocessor>;

auto make_general_classify() -> UniquePtr<ClassifyPostprocessor>;

auto make_general_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto make_yolo_keypoint() -> UniquePtr<KeypointPostprocessor>;

auto make_instance_segment() -> UniquePtr<SegmentPostprocessor>;

auto make_semantic_segment() -> UniquePtr<SegmentPostprocessor>;

auto finish_detect(DetectResultMap& result, const Vector<float>& iou_threshold,
                   const preprocess::ImageTransform& transform) -> void;

auto finish_rotated(RotatedResultMap& result, const Vector<float>& iou_threshold,
                    const preprocess::ImageTransform& transform) -> bool;

auto check_yolov5(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

auto check_yolov8(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_PARAMS_HPP
