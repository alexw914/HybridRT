// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/result_types.hpp
// Description: Declares the results of every built-in postprocessing task, in
//              source-image coordinates.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_RESULT_TYPES_HPP
#define HRT_POSTPROCESS_RESULT_TYPES_HPP

#include <hrt/core/define.hpp>

#include <cstddef>
#include <cstdint>

namespace hrt::postprocess {

struct ObjectBox {
  core::Rect2f rect;
  float score = 0.0F;
  std::int64_t type = 0;
  std::int64_t track_id = -1;
};

using DetectResult = Vector<ObjectBox>;

using DetectResultMap = Map<std::size_t, DetectResult>;

struct RotatedBox {
  core::RotatedRect rect;
  float score = 0.0F;
  std::int64_t type = 0;
  std::int64_t track_id = -1;
};

using RotatedResult = Vector<RotatedBox>;

using RotatedResultMap = Map<std::size_t, RotatedResult>;

// One score vector per engine output, indexed by class_id.
using ClassifyResult = Vector<Vector<float>>;

// The single engine output as float, otherwise unchanged.
using EmbeddingResult = Vector<float>;

// Visibility values as the model outputs them.
enum class KeypointVisible {
  Unlabeled = 0,
  Invisible = 1,
  Visible = 2,
};

struct Keypoint {
  core::Point2f point;
  KeypointVisible visible = KeypointVisible::Visible;
  float score = 1.0F;
};

// General keypoints use class 0 and a box covering the full input region.
struct KeypointObject {
  ObjectBox box;
  Vector<Keypoint> keypoints;
};

using KeypointResult = Vector<KeypointObject>;

using KeypointResultMap = Map<std::int64_t, KeypointResult>;

// Binary, row-major pixels over `rect` in source-image coordinates; 1 marks the class.
struct Mask {
  core::Rect rect;
  Vector<std::uint8_t> data;
};

struct SegmentObject {
  ObjectBox box;
  Mask mask;
};

using SegmentResult = Vector<SegmentObject>;

// Keys are class IDs; semantic results contain one full-input-region mask per class.
using SegmentResultMap = Map<std::int64_t, SegmentResult>;

struct OcrText {
  core::Rect2f rect; // Axis-aligned text box in source pixels.
  String text;       // UTF-8; empty for detection-only results.
  float detection_score = 0.0F;
  float recognition_score = 0.0F;
};

using OcrResult = Vector<OcrText>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_RESULT_TYPES_HPP
