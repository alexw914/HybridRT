// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/ocr/ocr.cpp
// Description: Implements PP-OCR DB box detection and CTC text decoding.
// -----------------------------------------------------------------------------

#include <hrt/postprocess/postprocessor.hpp>

#include <kitzoo/utilities/str.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <type_traits>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include "include/postprocess/common.hpp"
#include "include/postprocess/postprocessor_creators.hpp"

namespace hrt::postprocess {

namespace {

struct DetectionParam {
  String stage = "detection";
  String decoder = "db";
  float threshold = 0.2F;
  float box_threshold = 0.4F;
  float unclip_ratio = 1.4F;
  float min_size = 3.0F;
  int max_candidates = 3000;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DetectionParam, stage, decoder, threshold, box_threshold, unclip_ratio,
                                                min_size, max_candidates)

struct RecognitionParam {
  String stage = "recognition";
  String decoder = "ctc";
  String dictionary;
  bool append_space = true;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RecognitionParam, stage, decoder, dictionary, append_space)

auto ordered_points(const cv::RotatedRect& rect) -> Array<cv::Point2f, 4> {
  Array<cv::Point2f, 4> points;
  rect.points(points.data());
  std::sort(points.begin(), points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) -> bool { return a.x == b.x ? a.y < b.y : a.x < b.x; });
  if (points[0].y > points[1].y)
    std::swap(points[0], points[1]);
  if (points[2].y > points[3].y)
    std::swap(points[2], points[3]);
  return {points[0], points[2], points[3], points[1]};
}

// How a map is read: float32 as is, or int8 as integers ranked like their
// dequantized values (positive scale), dequantized only where a value is kept.
struct MapValues {
  float scale = 1.0F;
  std::int32_t zero_point = 0;

  KZ_NODISCARD auto value(float stored) const -> float { return stored; }

  KZ_NODISCARD auto value(std::int8_t stored) const -> float { return core::dequantize(stored, scale, zero_point); }
};

// Stored-value bounds of an int8 map: [low, high] holds the values in [0, 1]
// and values above `above` exceed `threshold`.
struct Int8Bounds {
  std::int32_t low = 128;
  std::int32_t high = -129;
  std::int32_t above = 127;
};

auto int8_bounds(const MapValues& map, float threshold) -> Int8Bounds {
  Int8Bounds bounds;
  for (std::int32_t q = -128; q < 128; ++q) {
    const auto value = map.value(static_cast<std::int8_t>(q));
    if (value >= 0.0F && value <= 1.0F) {
      bounds.low = std::min(bounds.low, q);
      bounds.high = q;
    }
    if (value <= threshold)
      bounds.above = q;
  }
  if (map.value(std::int8_t{-128}) > threshold)
    bounds.above = -129;
  return bounds;
}

// 255 where data > above or `all` is set, else 0; false when a value is
// outside [low, high] (or NaN). Branch-free, with & rather than &&, and int8
// bounds for int8 maps keep the loop vectorized at full lane width.
template <typename T>
auto binarize(const T* data, std::size_t count, T low, T high, T above, std::uint8_t all, std::uint8_t* out) -> bool {
  unsigned invalid = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto value = data[i];
    invalid |= static_cast<unsigned>(!((value >= low) & (value <= high)));
    out[i] = static_cast<std::uint8_t>((value > above ? 255 : 0) | all);
  }
  return invalid == 0;
}

class Ocr final : public OcrPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "ocr"))
      return false;
    detection_ = config["stage"] == "detection";
    const auto& outputs = engine.output_protos();
    KZ_CHECK_RETURN(outputs.size() == 1, false, "Ocr: expected one output");
    shape_ = outputs[0].shape;
    scratch_ = create_scratch(engine);
    if (detection_) {
      det_ = config.get<DetectionParam>();
      KZ_CHECK_RETURN(engine.proto().output_layout == engine::DataLayout::Nchw && shape_.size() == 4 &&
                          shape_[0] >= 1 && shape_[1] == 1 && shape_[2] > 0 && shape_[3] > 0 && shape_[2] <= 32768 &&
                          shape_[3] <= 32768,
                      false, "Ocr DB: expected [batch, 1, H, W]");
      return true;
    }
    const auto param = config.get<RecognitionParam>();
    characters_ = {String{}}; // CTC blank is class zero.
    auto lines = kitzoo::util::split(std::string_view(param.dictionary).substr(7), "\n");
    if (!lines.empty() && lines.back().empty())
      lines.pop_back();
    for (auto line : lines) {
      if (line.ends_with('\r'))
        line.remove_suffix(1);
      KZ_CHECK_RETURN(!line.empty(), false, "Ocr CTC: dictionary contains an empty entry");
      characters_.emplace_back(line);
    }
    if (param.append_space)
      characters_.emplace_back(" ");
    KZ_CHECK_RETURN(shape_.size() == 3 && shape_[0] >= 1 && shape_[1] > 0 &&
                        shape_[2] == static_cast<std::int64_t>(characters_.size()),
                    false, "Ocr CTC: expected [batch, T, dictionary + blank + optional space]");
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    if (!config.is_object() || !config.contains("stage") || !config["stage"].is_string())
      return {"postprocess.stage: expected detection or recognition"};
    if (config["stage"] == "detection") {
      core::Problems problems;
      core::check_param<DetectionParam>(config, "postprocess", {"type"}, problems);
      if (!problems.empty())
        return problems;
      const auto param = config.get<DetectionParam>();
      if (param.decoder != "db")
        problems.emplace_back("postprocess.decoder: detection requires db");
      if (!(param.threshold >= 0 && param.threshold <= 1 && param.box_threshold >= 0 && param.box_threshold <= 1))
        problems.emplace_back("postprocess: thresholds must be in [0, 1]");
      if (!(param.unclip_ratio >= 0 && param.unclip_ratio <= 10 && param.min_size > 0 && param.min_size <= 1000 &&
            param.max_candidates > 0 && param.max_candidates <= 10000))
        problems.emplace_back("postprocess: invalid DB size, unclip ratio or candidate limit");
      return problems;
    }
    if (config["stage"] != "recognition")
      return {"postprocess.stage: expected detection or recognition"};
    core::Problems problems;
    core::check_param<RecognitionParam>(config, "postprocess", {"type"}, problems);
    if (!problems.empty())
      return problems;
    const auto param = config.get<RecognitionParam>();
    if (param.decoder != "ctc")
      problems.emplace_back("postprocess.decoder: recognition requires ctc");
    if (!param.dictionary.starts_with("string:") || param.dictionary.size() == 7)
      problems.emplace_back("postprocess.dictionary: expected packaged string: dictionary text");
    return problems;
  }

protected:
  auto decode(OcrResult& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto& output = *outputs[0];
    // Contiguous int8 maps are read in place; others are dequantized first.
    if (output.data_type() == core::DataType::Int8 && output.is_contiguous() && output.scale() > 0.0F) {
      const auto* data = static_cast<const std::int8_t*>(output.memory()->cpu());
      const MapValues map{output.scale(), output.zero_point()};
      return detection_ ? detect(result, data, map, transform) : recognize(result, data, map);
    }
    const core::Buffer* floats = &output;
    if (floats->data_type() != core::DataType::Float32) {
      dequantize_buffer(*floats, scratch_[0]);
      floats = &scratch_[0];
    }
    const auto* data = static_cast<const float*>(floats->memory()->cpu());
    return detection_ ? detect(result, data, MapValues{}, transform) : recognize(result, data, MapValues{});
  }

private:
  template <typename T>
  auto recognize(OcrResult& result, const T* data, const MapValues& map) -> bool {
    auto& text = result.emplace_back();
    std::size_t previous = 0;
    std::size_t count = 0;
    double score = 0;
    for (std::int64_t t = 0; t < shape_[1]; ++t) {
      const auto* row = data + t * shape_[2];
      std::size_t index = 0;
      if constexpr (std::is_same_v<T, float>)
        index = static_cast<std::size_t>(std::max_element(row, row + shape_[2]) - row);
      else
        index = first_max(row, static_cast<std::size_t>(shape_[2]));
      const auto probability = map.value(row[index]);
      // Only the maximum is checked: logits usually exceed 1 there, and a full scan costs more than decoding.
      KZ_CHECK_RETURN(std::isfinite(probability) && probability >= 0 && probability <= 1, false,
                      "Ocr CTC: expected probabilities in [0, 1]");
      if (index != 0 && index != previous) {
        text.text += characters_[index];
        score += static_cast<double>(probability);
        ++count;
      }
      previous = index;
    }
    text.recognition_score = count == 0 ? 0 : static_cast<float>(score / static_cast<double>(count));
    return true;
  }

  template <typename T>
  auto detect(OcrResult& result, const T* data, const MapValues& map, const preprocess::ImageTransform& transform)
      -> bool {
    const auto height = static_cast<int>(shape_[2]);
    const auto width = static_cast<int>(shape_[3]);
    const auto pixels = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    bitmap_.create(height, width, CV_8U);
    bool valid = false;
    if constexpr (std::is_same_v<T, float>) {
      valid = binarize(data, pixels, 0.0F, 1.0F, det_.threshold, std::uint8_t{0}, bitmap_.ptr<std::uint8_t>());
    } else {
      const auto bounds = int8_bounds(map, det_.threshold);
      // An empty [low, high] rejects any map; an `above` below -128 sets every pixel.
      valid = bounds.low <= bounds.high &&
              binarize(data, pixels, static_cast<std::int8_t>(bounds.low), static_cast<std::int8_t>(bounds.high),
                       static_cast<std::int8_t>(std::max(bounds.above, -128)),
                       static_cast<std::uint8_t>(bounds.above < -128 ? 255 : 0), bitmap_.ptr<std::uint8_t>());
    }
    KZ_CHECK_RETURN(valid, false, "Ocr DB: expected finite probabilities in [0, 1]");
    contours_.clear();
    cv::findContours(bitmap_, contours_, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    const cv::Mat probability(height, width, std::is_same_v<T, float> ? CV_32F : CV_8S, const_cast<T*>(data));
    for (std::size_t i = 0; i < std::min(contours_.size(), static_cast<std::size_t>(det_.max_candidates)); ++i) {
      if (contours_[i].size() < 3)
        continue;
      auto rect = cv::minAreaRect(contours_[i]);
      if (std::min(rect.size.width, rect.size.height) < det_.min_size)
        continue;
      auto points = ordered_points(rect);
      const auto bounds = cv::boundingRect(points) & cv::Rect(0, 0, width, height);
      if (bounds.empty())
        continue;
      mask_.create(bounds.size(), CV_8U);
      mask_.setTo(0);
      Array<cv::Point, 4> polygon;
      for (std::size_t j = 0; j < 4; ++j)
        polygon[j] = {static_cast<int>(points[j].x) - bounds.x, static_cast<int>(points[j].y) - bounds.y};
      cv::fillConvexPoly(mask_, polygon, 1);
      // The mean of int8 values dequantizes like one value.
      const auto mean = cv::mean(probability(bounds), mask_)[0];
      const auto score = std::is_same_v<T, float>
                             ? static_cast<float>(mean)
                             : static_cast<float>(static_cast<double>(map.scale) * (mean - map.zero_point));
      if (score < det_.box_threshold)
        continue;
      // DB quad mode offsets a rectangle by area * ratio / perimeter. Its enclosing
      // rectangle grows by twice that distance, without a general polygon dependency.
      const auto distance = rect.size.area() * det_.unclip_ratio / (2 * (rect.size.width + rect.size.height));
      rect.size.width += 2 * distance;
      rect.size.height += 2 * distance;
      if (std::min(rect.size.width, rect.size.height) < det_.min_size + 2)
        continue;
      points = ordered_points(rect);
      for (auto& point : points) {
        const core::Point2f input{point.x * static_cast<float>(transform.input_size.width) / static_cast<float>(width),
                                  point.y * static_cast<float>(transform.input_size.height) /
                                      static_cast<float>(height)};
        const auto source = transform.reverse(input);
        point.x = std::clamp(source.x, static_cast<float>(transform.src_roi.x),
                             static_cast<float>(transform.src_roi.x + transform.src_roi.width - 1));
        point.y = std::clamp(source.y, static_cast<float>(transform.src_roi.y),
                             static_cast<float>(transform.src_roi.y + transform.src_roi.height - 1));
      }
      // PP-OCR drops clipped boxes whose truncated top or left side is at most 3 pixels.
      if (static_cast<int>(cv::norm(points[0] - points[1])) <= 3 ||
          static_cast<int>(cv::norm(points[0] - points[3])) <= 3)
        continue;
      const auto [left, right] = std::minmax({points[0].x, points[1].x, points[2].x, points[3].x});
      const auto [top, bottom] = std::minmax({points[0].y, points[1].y, points[2].y, points[3].y});
      auto& text = result.emplace_back();
      text.rect = {left, top, right - left, bottom - top};
      text.detection_score = score;
    }
    // PP-OCR reading order: sort by top then left, then reorder boxes within 10 pixels vertically by left edge.
    std::stable_sort(result.begin(), result.end(), [](const OcrText& a, const OcrText& b) -> bool {
      return a.rect.y == b.rect.y ? a.rect.x < b.rect.x : a.rect.y < b.rect.y;
    });
    for (std::size_t i = 0; i + 1 < result.size(); ++i) {
      for (auto j = i + 1; j > 0; --j) {
        auto& upper = result[j - 1].rect;
        auto& lower = result[j].rect;
        if (std::abs(lower.y - upper.y) >= 10 || lower.x >= upper.x)
          break;
        std::swap(result[j - 1], result[j]);
      }
    }
    return true;
  }

  bool detection_ = true;
  DetectionParam det_;
  core::Shape shape_;
  Vector<String> characters_;
  Vector<core::Buffer> scratch_;
  cv::Mat bitmap_;
  cv::Mat mask_;
  Vector<Vector<cv::Point>> contours_;
};

} // namespace

auto create_ocr() -> UniquePtr<OcrPostprocessor> {
  return std::make_unique<Ocr>();
}

} // namespace hrt::postprocess
