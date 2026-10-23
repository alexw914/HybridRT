// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/detect/detr_detect.cpp
// Description: Decodes normalized DETR queries with configurable score activation
//              and background handling, without NMS.
// -----------------------------------------------------------------------------

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

class DetrDetect final : public DetectPostprocessor {
public:
  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!is_float_outputs(engine, "detr_detect"))
      return false;
    param_ = config.get<DetrParam>();
    if (!config.contains("confidence_threshold"))
      param_.confidence_threshold = {0.25F};
    if (param_.background_class < -1 ||
        static_cast<std::int64_t>(param_.background_class) > static_cast<std::int64_t>(param_.class_num) ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    input_size_ = *engine.input_size();
    score_channels_ = static_cast<std::size_t>(param_.class_num) + (param_.background_class >= 0 ? 1U : 0U);
    const auto& outputs = engine.output_protos();
    const auto valid = [](const core::BufferProto& output, std::size_t channels) -> bool {
      return output.shape.size() == 3 && output.shape[0] >= 1 && output.shape[1] > 0 &&
             output.shape[2] == static_cast<std::int64_t>(channels);
    };
    if (param_.output_format == DetrOutputFormat::Combined) {
      KZ_CHECK_RETURN(outputs.size() == 1 && valid(outputs[0], 4 + score_channels_), false,
                      "detr_detect: combined format needs one [batch, N, 4 + score channels] output");
    } else {
      KZ_CHECK_RETURN(
          outputs.size() == 2 && valid(outputs[0], 4) && valid(outputs[1], score_channels_) &&
              outputs[0].shape[1] == outputs[1].shape[1],
          false, "detr_detect: separate format needs output 0 [batch, N, 4] boxes and output 1 [batch, N, scores]");
    }
    count_ = static_cast<std::size_t>(outputs[0].shape[1]);
    return true;
  }

  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_param<DetrParam>(config, "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const bool combined = param_.output_format == DetrOutputFormat::Combined;
    const HeadGridView boxes(*outputs[0], std::nullopt);
    const HeadGridView scores(*outputs[combined ? 0 : 1], std::nullopt);
    const std::int64_t score_first = combined ? 4 : 0;
    logits_.resize(score_channels_);
    for (std::size_t i = 0; i < count_; ++i) {
      const Array<float, 4> box{boxes.at(i, 0), boxes.at(i, 1), boxes.at(i, 2), boxes.at(i, 3)};
      if (!std::isfinite(box[0]) || !std::isfinite(box[1]) || !std::isfinite(box[2]) || !std::isfinite(box[3]) ||
          box[2] <= 0.0F || box[3] <= 0.0F)
        continue;
      for (std::size_t c = 0; c < score_channels_; ++c)
        logits_[c] = scores.at(i, score_first + static_cast<std::int64_t>(c));
      const auto* logits = logits_.data();
      float max_logit = std::numeric_limits<float>::lowest();
      float denominator = 0.0F;
      if (param_.score_activation == DetrScoreActivation::Softmax) {
        if (!std::all_of(logits, logits + score_channels_, [](float value) -> bool { return std::isfinite(value); }))
          continue;
        max_logit = *std::max_element(logits, logits + score_channels_);
        for (std::size_t c = 0; c < score_channels_; ++c)
          denominator += std::exp(logits[c] - max_logit);
      }
      std::size_t cls = 0;
      float score = -1.0F;
      for (std::size_t c = 0; c < score_channels_; ++c) {
        if (param_.background_class >= 0 && c == static_cast<std::size_t>(param_.background_class))
          continue;
        auto value = logits[c];
        if (!std::isfinite(value))
          continue;
        if (param_.score_activation == DetrScoreActivation::Sigmoid)
          value = core::sigmoid(value);
        else if (param_.score_activation == DetrScoreActivation::Softmax)
          value = std::exp(value - max_logit) / denominator;
        if (value > score) {
          cls = c - (param_.background_class >= 0 && c > static_cast<std::size_t>(param_.background_class) ? 1U : 0U);
          score = value;
        }
      }
      if (!(score > param_.confidence_threshold[cls]))
        continue;
      const auto rect = transform.clip_to_source(core::center_box(
          box[0] * static_cast<float>(input_size_.width), box[1] * static_cast<float>(input_size_.height),
          box[2] * static_cast<float>(input_size_.width), box[3] * static_cast<float>(input_size_.height)));
      if (!rect.empty())
        result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
    }
    return true;
  }

private:
  DetrParam param_;
  core::Size input_size_;
  std::size_t count_ = 0;
  std::size_t score_channels_ = 0;
  Vector<float> logits_;
};

} // namespace

auto create_detr_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<DetrDetect>();
}

} // namespace hrt::postprocess
