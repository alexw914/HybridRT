// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/general_keypoint.cpp
// Description: Implements keypoint decoding from heatmaps or coordinate
//              outputs and the keypoint postprocessor factory.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

enum class KeypointOutput {
  // One heatmap per keypoint: [.., K, H, W].
  Heatmap,
  // SimCC-style x, y and visibility distributions: three [1, K, bins].
  PointWithVisible,
  // Normalized coordinates as x row then y row: [.., 2, K].
  PointChannelFirst,
  // Normalized (x, y) pairs: [.., K, 2].
  PointChannelLast,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KeypointOutput, {
                                                 {KeypointOutput::Heatmap, "heatmap"},
                                                 {KeypointOutput::PointWithVisible, "point_with_visible"},
                                                 {KeypointOutput::PointChannelFirst, "point_channel_first"},
                                                 {KeypointOutput::PointChannelLast, "point_channel_last"},
                                             })

struct GeneralKeypointParam {
  KeypointOutput general_keypoint_type = KeypointOutput::Heatmap;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralKeypointParam, general_keypoint_type)

auto argmax(const float* begin, std::int64_t count) -> std::int64_t {
  return std::distance(begin, std::max_element(begin, begin + count));
}

class GeneralKeypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<GeneralKeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    type_ = config.get<GeneralKeypointParam>().general_keypoint_type;
    if (engine.proto().output_layout != engine::DataLayout::Nchw) {
      KZ_LOG_ERROR("general_keypoint: needs NCHW outputs");
      return false;
    }
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(!outputs.empty(), false, "general_keypoint: needs at least one output");
    KZ_CHECK_RETURN(type_ != KeypointOutput::PointWithVisible || outputs.size() == 3, false,
                    "general_keypoint: point_with_visible needs 3 outputs");
    for (const auto& output : outputs) {
      const auto& shape = output.shape;
      const std::size_t dimensions = type_ == KeypointOutput::Heatmap ? 3 : 2;
      KZ_CHECK_RETURN(
          shape.size() >= dimensions &&
              std::all_of(shape.begin(), shape.end(), [](std::int64_t value) -> bool { return value > 0; }) &&
              std::all_of(shape.begin(), shape.end() - static_cast<std::ptrdiff_t>(dimensions),
                          [](std::int64_t value) -> bool { return value == 1; }),
          false, "general_keypoint: needs positive dimensions and one keypoint set per output");
      switch (type_) {
      case KeypointOutput::Heatmap:
        KZ_CHECK_RETURN(shape[shape.size() - 2] <= std::numeric_limits<int>::max() &&
                            shape.back() <= std::numeric_limits<int>::max(),
                        false, "general_keypoint: heatmap dimensions exceed OpenCV limits");
        break;
      case KeypointOutput::PointChannelFirst:
        KZ_CHECK_RETURN(shape[shape.size() - 2] == 2, false, "general_keypoint: needs [..., 2, K] coordinates");
        break;
      case KeypointOutput::PointChannelLast:
        KZ_CHECK_RETURN(shape.back() == 2, false, "general_keypoint: needs [..., K, 2] coordinates");
        break;
      case KeypointOutput::PointWithVisible:
        KZ_CHECK_RETURN(shape.size() == 3 && shape[1] == outputs[0].shape[1], false,
                        "general_keypoint: needs three [1, K, bins] outputs with matching K");
        break;
      }
    }
    KZ_CHECK_RETURN(type_ != KeypointOutput::PointWithVisible || outputs[2].shape[2] == 3, false,
                    "general_keypoint: visibility needs 3 states (unlabeled, invisible, visible)");
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    while (scratch_.size() < outputs.size())
      scratch_.emplace_back(core::MemoryType::Cpu, core::DataType::Float32);
    Vector<const float*> data(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      data[i] = float_data(*outputs[i], scratch_[i]);
      if (data[i] == nullptr)
        return false;
    }
    auto& objects = result[0];
    switch (type_) {
    case KeypointOutput::Heatmap:
      for (std::size_t i = 0; i < outputs.size(); ++i)
        decode_heatmaps(objects.emplace_back().keypoints, data[i], outputs[i]->shape());
      break;
    case KeypointOutput::PointWithVisible:
      decode_point_with_visible(objects.emplace_back().keypoints, data, outputs);
      break;
    case KeypointOutput::PointChannelFirst:
    case KeypointOutput::PointChannelLast:
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& shape = outputs[i]->shape();
        auto& points = objects.emplace_back().keypoints;
        const auto first = type_ == KeypointOutput::PointChannelFirst;
        const auto count = first ? shape.back() : shape[shape.size() - 2];
        for (std::int64_t k = 0; k < count; ++k) {
          const auto x = first ? data[i][k] : data[i][k * 2];
          const auto y = first ? data[i][count + k] : data[i][k * 2 + 1];
          points.push_back({{x, y}, KeypointVisible::Visible, 1.0F});
        }
      }
      break;
    }
    const auto width = static_cast<float>(transform.input_size.width);
    const auto height = static_cast<float>(transform.input_size.height);
    for (auto& object : objects) {
      const auto& region = transform.src_roi;
      object.box = {{static_cast<float>(region.x), static_cast<float>(region.y), static_cast<float>(region.width),
                     static_cast<float>(region.height)},
                    1.0F,
                    0,
                    -1};
      for (auto& keypoint : object.keypoints)
        keypoint.point = to_source(core::Point2f{keypoint.point.x * width, keypoint.point.y * height}, transform);
    }
    return true;
  }

private:
  // Argmax refined with DARK (Gaussian-smoothed Taylor expansion).
  auto decode_heatmaps(Vector<Keypoint>& points, const float* data, const core::Shape& shape) -> void {
    const auto height = shape[shape.size() - 2];
    const auto width = shape.back();
    const auto maps = shape[shape.size() - 3];
    auto safe_log = [](float v) -> float { return std::log(std::clamp(v, 0.001F, 1.0F)); };
    for (std::int64_t k = 0; k < maps; ++k) {
      const auto* map = data + k * height * width;
      const auto index = argmax(map, height * width);
      const auto score = map[index];
      const auto px = index % width;
      const auto py = index / width;
      const cv::Mat raw(static_cast<int>(height), static_cast<int>(width), CV_32FC1, const_cast<float*>(map));
      cv::GaussianBlur(raw, blurred_, cv::Size(11, 11), 3.4);
      auto at = [&](std::int64_t y, std::int64_t x) -> float {
        return safe_log(blurred_.at<float>(static_cast<int>(y), static_cast<int>(x)));
      };
      const auto x1 = std::min(px + 1, width - 1);
      const auto x0 = std::max<std::int64_t>(px - 1, 0);
      const auto y1 = std::min(py + 1, height - 1);
      const auto y0 = std::max<std::int64_t>(py - 1, 0);
      const auto center = at(py, px);
      const auto dx = 0.5F * (at(py, x1) - at(py, x0));
      const auto dy = 0.5F * (at(y1, px) - at(y0, px));
      const auto dxx = at(py, x1) + at(py, x0) - 2.0F * center;
      const auto dyy = at(y1, px) + at(y0, px) - 2.0F * center;
      const auto dxy =
          0.5F * (at(y1, x1) - at(py, x1) - at(y1, px) + 2.0F * center - at(py, x0) - at(y0, px) + at(y0, x0));
      cv::Matx22f hessian(dxx + 1e-6F, dxy, dxy, dyy + 1e-6F);
      const auto offset = hessian.inv() * cv::Matx21f(dx, dy);
      points.push_back({{(static_cast<float>(px) - offset(0, 0)) / static_cast<float>(width),
                         (static_cast<float>(py) - offset(1, 0)) / static_cast<float>(height)},
                        KeypointVisible::Visible,
                        score});
    }
  }

  static auto decode_point_with_visible(Vector<Keypoint>& points, const Vector<const float*>& data,
                                        const core::Buffers& outputs) -> void {
    const auto count = outputs[0]->shape()[1];
    const auto x_bins = outputs[0]->shape()[2];
    const auto y_bins = outputs[1]->shape()[2];
    const auto v_bins = outputs[2]->shape()[2];
    for (std::int64_t k = 0; k < count; ++k) {
      const auto x = argmax(data[0] + k * x_bins, x_bins);
      const auto y = argmax(data[1] + k * y_bins, y_bins);
      const auto v = argmax(data[2] + k * v_bins, v_bins);
      points.push_back(
          {{static_cast<float>(x) / static_cast<float>(x_bins), static_cast<float>(y) / static_cast<float>(y_bins)},
           static_cast<KeypointVisible>(v),
           data[2][k * v_bins + 2]});
    }
  }

  KeypointOutput type_ = KeypointOutput::Heatmap;
  Vector<core::Buffer> scratch_;
  cv::Mat blurred_;
};

} // namespace

auto make_general_keypoint() -> UniquePtr<KeypointPostprocessor> {
  return std::make_unique<GeneralKeypoint>();
}

auto make_keypoint(std::string_view type) -> UniquePtr<KeypointPostprocessor> {
  if (type == "GeneralKeypoint")
    return make_general_keypoint();
  if (type == "YoloKeypoint")
    return make_yolo_keypoint();
  return nullptr;
}

auto create_keypoint_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<KeypointPostprocessor> {
  return checked(make_keypoint(type), type, config, engine);
}

} // namespace hrt::postprocess
