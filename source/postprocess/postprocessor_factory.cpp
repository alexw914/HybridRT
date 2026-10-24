// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/postprocessor_factory.cpp
// Description: Registers the built-in postprocessors by postprocess.type and
//              implements their typed creation and configuration checks.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor_factory.hpp>

#include <algorithm>
#include <array>
#include <variant>

#include "include/postprocess/common.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

template <typename Result>
using Creator = UniquePtr<Postprocessor<Result>> (*)();

using AnyCreator =
    std::variant<Creator<DetectResultMap>, Creator<RotatedResultMap>, Creator<ClassifyResult>, Creator<EmbeddingResult>,
                 Creator<KeypointResultMap>, Creator<SegmentResultMap>, Creator<OcrResult>>;

struct Entry {
  std::string_view type;
  AnyCreator creator;
};

// Every built-in type, once; the size follows the list.
constexpr auto kEntries = std::to_array<Entry>({
    {"YoloV5Detect", create_yolov5_detect},
    {"YoloV8Detect", create_yolov8_detect},
    {"Yolo26Detect", create_yolo26_detect},
    {"GeneralDetect", create_general_detect},
    {"DetrDetect", create_detr_detect},
    {"E2EDetect", create_e2e_detect},
    {"YoloV5Rotated", create_yolov5_rotated},
    {"YoloV8Rotated", create_yolov8_rotated},
    {"Yolo26Rotated", create_yolo26_rotated},
    {"GeneralRotated", create_general_rotated},
    {"E2ERotated", create_e2e_rotated},
    {"GeneralClassify", create_general_classify},
    {"Embedding", create_embedding},
    {"GeneralKeypoint", create_general_keypoint},
    {"GeneralYoloPose", create_general_yolo_pose},
    {"YoloV8Keypoint", create_yolov8_keypoint},
    {"Yolo26Keypoint", create_yolo26_keypoint},
    {"RtmoKeypoint", create_rtmo_keypoint},
    {"E2EKeypoint", create_e2e_keypoint},
    {"GeneralYoloSegment", create_general_yolo_segment},
    {"YoloV8Segment", create_yolov8_segment},
    {"Yolo26Segment", create_yolo26_segment},
    {"E2ESegment", create_e2e_segment},
    {"SemanticSegment", create_semantic_segment},
    {"Ocr", create_ocr},
});

auto find_entry(std::string_view type) -> const Entry* {
  const auto found =
      std::find_if(kEntries.begin(), kEntries.end(), [type](const Entry& entry) -> bool { return entry.type == type; });
  return found == kEntries.end() ? nullptr : &*found;
}

} // namespace

template <typename Result>
auto create_postprocessor(std::string_view type, const kitzoo::json::Json& config, const engine::Engine& engine)
    -> UniquePtr<Postprocessor<Result>> {
  const auto* entry = find_entry(type);
  KZ_CHECK_MSG(entry != nullptr, "postprocess: unknown type \"{}\"", type);
  const auto* creator = std::get_if<Creator<Result>>(&entry->creator);
  KZ_CHECK_MSG(creator != nullptr, "postprocess: \"{}\" does not produce the requested result type", type);
  auto postprocessor = (*creator)();
  KZ_CHECK_MSG(postprocessor != nullptr, "postprocess: cannot create \"{}\"", type);
  const auto problems = postprocessor->check_config(config);
  KZ_CHECK_MSG(problems.empty(), "postprocess: invalid {} config: {}", type, core::join(problems));
  KZ_CHECK_MSG(engine.input_size().has_value(), "postprocess: {} needs one 4-D image input", type);
  KZ_CHECK_MSG(postprocessor->init(config, engine), "postprocess: {} does not fit the engine", type);
  return postprocessor;
}

template <typename Result>
auto supports_postprocessor(std::string_view type) -> bool {
  const auto* entry = find_entry(type);
  return entry != nullptr && std::holds_alternative<Creator<Result>>(entry->creator);
}

auto check_postprocess_config(std::string_view type, const kitzoo::json::Json& config) -> Vector<String> {
  const auto* entry = find_entry(type);
  if (entry == nullptr)
    return {"postprocess.type: unknown \"" + String(type) + "\""};
  const auto postprocessor =
      std::visit([](auto creator) -> UniquePtr<PostprocessorBase> { return creator(); }, entry->creator);
  return postprocessor->check_config(config);
}

#define HRT_POSTPROCESS_INSTANTIATE(Result)                                                                            \
  template auto create_postprocessor<Result>(std::string_view, const kitzoo::json::Json&, const engine::Engine&)       \
      -> UniquePtr<Postprocessor<Result>>;                                                                             \
  template auto supports_postprocessor<Result>(std::string_view)->bool;

HRT_POSTPROCESS_INSTANTIATE(DetectResultMap)
HRT_POSTPROCESS_INSTANTIATE(RotatedResultMap)
HRT_POSTPROCESS_INSTANTIATE(ClassifyResult)
HRT_POSTPROCESS_INSTANTIATE(EmbeddingResult)
HRT_POSTPROCESS_INSTANTIATE(KeypointResultMap)
HRT_POSTPROCESS_INSTANTIATE(SegmentResultMap)
HRT_POSTPROCESS_INSTANTIATE(OcrResult)

#undef HRT_POSTPROCESS_INSTANTIATE

} // namespace hrt::postprocess
