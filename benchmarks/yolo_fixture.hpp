// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: benchmarks/yolo_fixture.hpp
// Description: Loads the YOLO task fixtures of assets/yolo shared by the
//              benchmarks.
// -----------------------------------------------------------------------------

#ifndef HRT_BENCHMARKS_YOLO_FIXTURE_HPP
#define HRT_BENCHMARKS_YOLO_FIXTURE_HPP

#include <hrt/model/image_model.hpp>

#include <cstddef>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <string_view>

namespace hrt::bench {

// The algos of assets/yolo/yolo.json, in manifest order; the benchmark
// argument indexes this list.
inline constexpr Array<const char*, 5> kTasks{"detect", "segment", "pose", "obb", "classify"};

// Entry `task` of "algos" in assets/yolo/<manifest>, or null when the fixture
// is missing. Its model paths are relative to the repository root, which
// becomes the working directory.
inline auto algo_entry(std::string_view manifest, std::size_t task) -> kitzoo::json::Json {
  std::filesystem::current_path(HRT_SOURCE_DIR);
  const kitzoo::json::Reader reader(std::filesystem::path("assets/yolo") / manifest);
  const auto& json = reader.raw();
  const auto algos = json.is_object() ? json.find("algos") : json.end();
  if (algos == json.end() || !algos->is_array() || task >= algos->size())
    return {};
  return (*algos)[task];
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
