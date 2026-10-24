// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/postprocessor_creators.hpp
// Description: Declares the entry points of the built-in decoders; postprocessor_factory.cpp
//              registers them, checks their config and initializes them.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_POSTPROCESSOR_CREATORS_HPP
#define HRT_POSTPROCESS_POSTPROCESSOR_CREATORS_HPP

#include <hrt/postprocess/postprocessor.hpp>

namespace hrt::postprocess {

auto create_embedding() -> UniquePtr<EmbeddingPostprocessor>;

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

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_POSTPROCESSOR_CREATORS_HPP
