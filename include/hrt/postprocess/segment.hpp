// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/postprocess/segment.hpp
// Description: Declares segmentation results and postprocessors.
// -----------------------------------------------------------------------------

#ifndef HRT_POSTPROCESS_SEGMENT_HPP
#define HRT_POSTPROCESS_SEGMENT_HPP

#include <hrt/postprocess/detect.hpp>
#include <hrt/postprocess/postprocessor.hpp>

#include <cstdint>
#include <string_view>

namespace hrt::postprocess {

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

using SegmentPostprocessor = Postprocessor<SegmentResultMap>;

// Aborts on an unknown type, a config check_config rejects, or engine outputs
// that do not fit the config.
auto create_segment_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<SegmentPostprocessor>;

} // namespace hrt::postprocess

#endif // HRT_POSTPROCESS_SEGMENT_HPP
