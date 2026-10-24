// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/view.hpp
// Description: Declares drawing of model results on host BGR images.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_VIEW_HPP
#define HRT_CORE_VIEW_HPP

#include <hrt/postprocess/result_types.hpp>

#include <opencv2/core/mat.hpp>

namespace hrt::core {

// Draws in place on a host CV_8UC3 BGR image; result coordinates are source-image pixels.

auto draw(cv::Mat& canvas, const postprocess::DetectResultMap& result) -> void;
auto draw(cv::Mat& canvas, const postprocess::RotatedResultMap& result) -> void;
auto draw(cv::Mat& canvas, const postprocess::ClassifyResult& result) -> void;
auto draw(cv::Mat& canvas, const postprocess::KeypointResultMap& result) -> void;
auto draw(cv::Mat& canvas, const postprocess::SegmentResultMap& result) -> void;

} // namespace hrt::core

#endif // HRT_CORE_VIEW_HPP
