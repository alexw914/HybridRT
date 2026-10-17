// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/yolo_fixture.hpp
// Description: Loads the YOLO task algos of the reference manifest shared by
//              the benchmarks.
// -----------------------------------------------------------------------------

#ifndef HRT_BENCHMARKS_YOLO_FIXTURE_HPP
#define HRT_BENCHMARKS_YOLO_FIXTURE_HPP

#include <hrt/model/image_model.hpp>

#include <cstddef>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <string_view>

namespace hrt::bench {

// The benchmarked tasks; the benchmark argument indexes this list.
inline constexpr Array<const char*, 5> kTasks{"detect", "segment", "pose", "obb", "classify"};

// Their algos in assets/template/hrtc_template.json.
inline constexpr Array<const char*, 5> kAlgos{"yolov8n_detect", "yolov8n_segment", "yolov8n_pose", "yolov8n_obb",
                                              "yolov8n_classify"};

enum class Provider {
  Cpu,
  CoreMl,
};

// The template algo of `task` on the ORT `provider`, or null when its local
// model under assets/ is missing and `need_model`. Model paths are relative to
// the repository root, which becomes the working directory.
inline auto algo_entry(std::size_t task, Provider provider = Provider::Cpu,
                       bool need_model = true) -> kitzoo::json::Json {
  std::filesystem::current_path(HRT_SOURCE_DIR);
  const kitzoo::json::Reader reader(std::filesystem::path("assets/template/hrtc_template.json"));
  const auto& json = reader.raw();
  const auto algos = json.is_object() ? json.find("algos") : json.end();
  if (algos == json.end() || !algos->is_array() || task >= kAlgos.size())
    return {};
  for (auto algo : *algos) {
    if (algo.value("name", String{}) != kAlgos.at(task))
      continue;
    auto& backend = algo["config"]["engine"]["backends"][0];
    if (need_model && !std::filesystem::is_regular_file(backend.value("data", String{})))
      return {};
    if (provider == Provider::CoreMl)
      backend["config"]["execution_provider"] = "coreml";
    return algo;
  }
  return {};
}

// The test image of `task` as BGR: the airport for OBB, the bus otherwise.
inline auto task_image(std::size_t task) -> cv::Mat {
  const auto name = std::string_view(kTasks.at(task)) == "obb" ? "airport.jpg" : "bus.jpg";
  return cv::imread((std::filesystem::path(HRT_SOURCE_DIR) / "assets/yolo/images" / name).string(), cv::IMREAD_COLOR);
}

inline auto wrap_bgr(const cv::Mat& bgr) -> std::optional<core::Image> {
  return core::Image::wrap(core::MemoryType::Cpu, bgr.data, core::ImageFormat::InterBgr, {bgr.cols, bgr.rows},
                           core::DataType::Uint8, bgr.step);
}

// Calls `fn.template operator()<Result>()` with the result type of `task`.
template <typename Fn>
auto with_result(std::size_t task, Fn&& fn) -> void {
  switch (task) {
  case 0:
    fn.template operator()<postprocess::DetectResultMap>();
    break;
  case 1:
    fn.template operator()<postprocess::SegmentResultMap>();
    break;
  case 2:
    fn.template operator()<postprocess::KeypointResultMap>();
    break;
  case 3:
    fn.template operator()<postprocess::RotatedResultMap>();
    break;
  default:
    fn.template operator()<postprocess::ClassifyResult>();
    break;
  }
}

} // namespace hrt::bench

#endif // HRT_BENCHMARKS_YOLO_FIXTURE_HPP
