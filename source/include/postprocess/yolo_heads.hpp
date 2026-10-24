// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/postprocess/yolo_heads.hpp
// Description: Declares validation of YOLO output heads and inference of their strides.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_YOLO_HEADS_HPP
#define HRT_POSTPROCESS_YOLO_HEADS_HPP

#include <hrt/engine/engine.hpp>

#include <cstddef>

#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

auto resolve_yolov5_heads(YoloV5Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

auto resolve_yolov8_heads(YoloV8Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

// resolve_yolov8_heads() for YOLO26 heads (reg_max 1); fills strides and per-class thresholds.
auto resolve_yolo26_heads(Yolo26Param& param, const engine::Engine& engine, std::size_t extra_channels) -> bool;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_YOLO_HEADS_HPP
