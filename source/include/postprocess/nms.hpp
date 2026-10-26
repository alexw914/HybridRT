// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/nms.hpp
// Description: Declares the greedy non-maximum suppression shared by the
//              postprocessors and its mapping of kept boxes to the source.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_NMS_HPP
#define HRT_POSTPROCESS_NMS_HPP

#include <hrt/postprocess/result_types.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <cstddef>

namespace hrt::postprocess {

// Indices of the rects kept by greedy NMS, highest score first.
auto nms_indices(const Vector<core::Rect2f>& rects, const Vector<float>& scores, float iou_threshold)
    -> Vector<std::size_t>;

// Greedy NMS within one class, highest score first.
auto nms(DetectResult& boxes, float iou_threshold) -> void;

// Greedy NMS within one class using polygon IoU.
auto nms_rotated(RotatedResult& boxes, float iou_threshold) -> void;

auto nms_reverse(DetectResultMap& result, const Vector<float>& iou_threshold,
                 const preprocess::ImageTransform& transform) -> void;

auto nms_rotated_reverse(RotatedResultMap& result, const Vector<float>& iou_threshold,
                         const preprocess::ImageTransform& transform) -> bool;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_NMS_HPP
