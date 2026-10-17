// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/keypoint/general_keypoint.cpp
// Description: Implements top-down keypoint decoding from heatmaps, SimCC
//              distributions or coordinate outputs and the keypoint
//              postprocessor factory.
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
  // One heatmap per keypoint: [.., K, H, W] (HRNet, SimpleBaseline, ViTPose).
  Heatmap,
  // SimCC x and y distributions: [1, K, W * r] and [1, K, H * r] (RTMPose).
  Simcc,
  // SimCC-style x, y and visibility distributions: three [1, K, bins].
  PointWithVisible,
  // Normalized coordinates as x row then y row: [.., 2, K].
  PointChannelFirst,
  // Normalized (x, y) pairs: [.., K, 2].
  PointChannelLast,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KeypointOutput, {
                                                 {KeypointOutput::Heatmap, "heatmap"},
                                                 {KeypointOutput::Simcc, "simcc"},
                                                 {KeypointOutput::PointWithVisible, "point_with_visible"},
                                                 {KeypointOutput::PointChannelFirst, "point_channel_first"},
                                                 {KeypointOutput::PointChannelLast, "point_channel_last"},
                                             })

enum class HeatmapRefine {
  None,
  Dark,
};

NLOHMANN_JSON_SERIALIZE_ENUM(HeatmapRefine, {
                                                {HeatmapRefine::None, "none"},
                                                {HeatmapRefine::Dark, "dark"},
                                            })

enum class KeypointScoreActivation {
  None,
  Sigmoid,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KeypointScoreActivation, {
                                                          {KeypointScoreActivation::None, "none"},
                                                          {KeypointScoreActivation::Sigmoid, "sigmoid"},
                                                      })

// heatmap_refine applies to heatmaps; score_activation and keypoint_threshold
// to heatmap and simcc scores, whose keypoints are visible above the threshold.
struct GeneralKeypointParam {
  KeypointOutput general_keypoint_type = KeypointOutput::Heatmap;
  HeatmapRefine heatmap_refine = HeatmapRefine::Dark;
  KeypointScoreActivation score_activation = KeypointScoreActivation::None;
  float keypoint_threshold = 0.0F;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralKeypointParam, general_keypoint_type, heatmap_refine,
                                                score_activation, keypoint_threshold)

auto argmax(const float* begin, std::int64_t count) -> std::int64_t {
  return std::distance(begin, std::max_element(begin, begin + count));
}

class GeneralKeypoint final : public KeypointPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<GeneralKeypointParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<GeneralKeypointParam>();
    const auto type = param_.general_keypoint_type;
    if (engine.proto().output_layout != engine::DataLayout::Nchw) {
      KZ_LOG_ERROR("general_keypoint: needs NCHW outputs");
      return false;
    }
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(!outputs.empty(), false, "general_keypoint: needs at least one output");
    KZ_CHECK_RETURN(type != KeypointOutput::Simcc || outputs.size() == 2, false,
                    "general_keypoint: simcc needs 2 outputs (x, y)");
    KZ_CHECK_RETURN(type != KeypointOutput::PointWithVisible || outputs.size() == 3, false,
                    "general_keypoint: point_with_visible needs 3 outputs");
    for (const auto& output : outputs) {
      const auto& shape = output.shape;
      const std::size_t dimensions = type == KeypointOutput::Heatmap ? 3 : 2;
      KZ_CHECK_RETURN(
          shape.size() >= dimensions &&
              std::all_of(shape.begin(), shape.end(), [](std::int64_t value) -> bool { return value > 0; }) &&
              std::all_of(shape.begin(), shape.end() - static_cast<std::ptrdiff_t>(dimensions),
                          [](std::int64_t value) -> bool { return value == 1; }),
          false, "general_keypoint: needs positive dimensions and one keypoint set per output");
      switch (type) {
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
      case KeypointOutput::Simcc:
      case KeypointOutput::PointWithVisible:
        KZ_CHECK_RETURN(shape.size() == 3 && shape[1] == outputs[0].shape[1], false,
                        "general_keypoint: needs [1, K, bins] outputs with matching K");
        break;
      }
    }
    KZ_CHECK_RETURN(type != KeypointOutput::PointWithVisible || outputs[2].shape[2] == 3, false,
                    "general_keypoint: visibility needs 3 states (unlabeled, invisible, visible)");
    return true;
  }

protected:
  auto decode(KeypointResultMap& result, const core::Buffers& outputs,
              const preprocess::ImageTransform& transform) -> bool override {
    while (scratch_.size() < outputs.size())
      scratch_.emplace_back(core::MemoryType::Cpu, core::DataType::Float32);
    Vector<const float*> data(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      data[i] = float_data(*outputs[i], scratch_[i]);
      if (data[i] == nullptr)
        return false;
    }
    auto& objects = result[0];
    const auto type = param_.general_keypoint_type;
    switch (type) {
    case KeypointOutput::Heatmap:
      for (std::size_t i = 0; i < outputs.size(); ++i)
        decode_heatmaps(objects.emplace_back().keypoints, data[i], outputs[i]->shape());
      break;
    case KeypointOutput::Simcc:
      decode_simcc(objects.emplace_back().keypoints, data, outputs);
      break;
    case KeypointOutput::PointWithVisible:
      decode_point_with_visible(objects.emplace_back().keypoints, data, outputs);
      break;
    case KeypointOutput::PointChannelFirst:
    case KeypointOutput::PointChannelLast:
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& shape = outputs[i]->shape();
        auto& points = objects.emplace_back().keypoints;
        const auto first = type == KeypointOutput::PointChannelFirst;
        const auto count = first ? shape.back() : shape[shape.size() - 2];
        for (std::int64_t k = 0; k < count; ++k) {
          const auto x = first ? data[i][k] : data[i][k * 2];
          const auto y = first ? data[i][count + k] : data[i][k * 2 + 1];
          points.push_back({{x, y}, KeypointVisible::Visible, 1.0F});
        }
      }
      break;
    }
    const auto scored = type == KeypointOutput::Heatmap || type == KeypointOutput::Simcc;
    const auto width = static_cast<float>(transform.input_size.width);
    const auto height = static_cast<float>(transform.input_size.height);
    for (auto& object : objects) {
      const auto& region = transform.src_roi;
      object.box = {{static_cast<float>(region.x), static_cast<float>(region.y), static_cast<float>(region.width),
                     static_cast<float>(region.height)},
                    1.0F,
                    0,
                    -1};
      for (auto& keypoint : object.keypoints) {
        keypoint.point = to_source(core::Point2f{keypoint.point.x * width, keypoint.point.y * height}, transform);
        if (!scored)
          continue;
        if (param_.score_activation == KeypointScoreActivation::Sigmoid)
          keypoint.score = sigmoid(keypoint.score);
        keypoint.visible =
            keypoint.score > param_.keypoint_threshold ? KeypointVisible::Visible : KeypointVisible::Invisible;
      }
    }
    return true;
  }

private:
  // Argmax per map; peak cells normalized by the map size, as the MSRA codec
  // scales them by input / heatmap size.
  auto decode_heatmaps(Vector<Keypoint>& points, const float* data, const core::Shape& shape) -> void {
    const auto height = shape[shape.size() - 2];
    const auto width = shape.back();
    const auto maps = shape[shape.size() - 3];
    for (std::int64_t k = 0; k < maps; ++k) {
      const auto* map = data + k * height * width;
      const auto index = argmax(map, height * width);
      auto x = static_cast<float>(index % width);
      auto y = static_cast<float>(index / width);
      if (param_.heatmap_refine == HeatmapRefine::Dark)
        refine_dark(x, y, map, height, width);
      points.push_back(
          {{x / static_cast<float>(width), y / static_cast<float>(height)}, KeypointVisible::Visible, map[index]});
    }
  }

  // DARK as in mmpose refine_keypoints_dark: second-order Taylor expansion of
  // the log of the map blurred by an 11 x 11 Gaussian (sigma 2, zero border),
  // at peaks at least 2 cells inside. Only a maximum of the blurred map
  // (negative definite Hessian) moves, by less than one cell; logit heatmaps
  // whose blurred values are not positive keep the argmax.
  auto refine_dark(float& x, float& y, const float* map, std::int64_t height, std::int64_t width) -> void {
    const auto px = static_cast<int>(x);
    const auto py = static_cast<int>(y);
    if (px < 2 || py < 2 || px > width - 3 || py > height - 3)
      return;
    const cv::Mat raw(static_cast<int>(height), static_cast<int>(width), CV_32FC1, const_cast<float*>(map));
    cv::GaussianBlur(raw, blurred_, cv::Size(11, 11), 0.0, 0.0, cv::BORDER_CONSTANT);
    auto at = [&](int dy, int dx) -> float { return std::log(std::max(blurred_.at<float>(py + dy, px + dx), 1e-10F)); };
    const auto center = at(0, 0);
    const auto dx = 0.5F * (at(0, 1) - at(0, -1));
    const auto dy = 0.5F * (at(1, 0) - at(-1, 0));
    const auto dxx = 0.25F * (at(0, 2) - 2.0F * center + at(0, -2));
    const auto dyy = 0.25F * (at(2, 0) - 2.0F * center + at(-2, 0));
    const auto dxy = 0.25F * (at(1, 1) - at(-1, 1) - at(1, -1) + at(-1, -1));
    const auto det = dxx * dyy - dxy * dxy;
    if (!(dxx < 0.0F && det > 0.0F))
      return;
    const auto offset_x = (dyy * dx - dxy * dy) / det;
    const auto offset_y = (dxx * dy - dxy * dx) / det;
    if (std::abs(offset_x) >= 1.0F || std::abs(offset_y) >= 1.0F)
      return;
    x -= offset_x;
    y -= offset_y;
  }

  // Bins normalized by the bin count, as mmpose divides them by the split
  // ratio (bins = input size * ratio); score min(max x, max y).
  static auto decode_simcc(Vector<Keypoint>& points, const Vector<const float*>& data,
                           const core::Buffers& outputs) -> void {
    const auto count = outputs[0]->shape()[1];
    const auto x_bins = outputs[0]->shape()[2];
    const auto y_bins = outputs[1]->shape()[2];
    for (std::int64_t k = 0; k < count; ++k) {
      const auto* xs = data[0] + k * x_bins;
      const auto* ys = data[1] + k * y_bins;
      const auto x = argmax(xs, x_bins);
      const auto y = argmax(ys, y_bins);
      points.push_back(
          {{static_cast<float>(x) / static_cast<float>(x_bins), static_cast<float>(y) / static_cast<float>(y_bins)},
           KeypointVisible::Visible,
           std::min(xs[x], ys[y])});
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

  GeneralKeypointParam param_;
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
  if (type == "RtmoKeypoint")
    return make_rtmo_keypoint();
  return nullptr;
}

auto create_keypoint_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<KeypointPostprocessor> {
  return checked(make_keypoint(type), type, config, engine);
}

} // namespace hrt::postprocess
