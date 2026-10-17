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
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<DetrParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    param_ = config.get<DetrParam>();
    if (!config.contains("confidence_threshold"))
      param_.confidence_threshold = {0.25F};
    if (param_.class_num == 0 || param_.background_class < -1 ||
        static_cast<std::int64_t>(param_.background_class) > static_cast<std::int64_t>(param_.class_num) ||
        !expand_per_class(param_.confidence_threshold, param_.class_num, "confidence_threshold"))
      return false;
    const auto input = engine_input_size(engine);
    if (!input || input->width <= 0 || input->height <= 0)
      return false;
    input_size_ = *input;
    score_channels_ = static_cast<std::size_t>(param_.class_num) + (param_.background_class >= 0 ? 1U : 0U);
    const auto& outputs = engine.output_protos();
    const auto valid = [](const core::BufferProto& output, std::size_t channels) -> bool {
      return output.shape.size() == 3 && output.shape[0] == 1 && output.shape[1] > 0 &&
             output.shape[2] == static_cast<std::int64_t>(channels) &&
             (output.data_type == core::DataType::Float32 || output.data_type == core::DataType::Int8);
    };
    if (param_.output_format == DetrOutputFormat::Combined) {
      KZ_CHECK_RETURN(outputs.size() == 1 && valid(outputs[0], 4 + score_channels_), false,
                      "detr_detect: combined format needs one [1, N, 4 + score channels] output");
      boxes_index_ = 0;
      scores_index_ = 0;
    } else {
      KZ_CHECK_RETURN(outputs.size() == 2 && param_.boxes_output < 2 && param_.scores_output < 2 &&
                          param_.boxes_output != param_.scores_output && valid(outputs[param_.boxes_output], 4) &&
                          valid(outputs[param_.scores_output], score_channels_) &&
                          outputs[param_.boxes_output].shape[1] == outputs[param_.scores_output].shape[1],
                      false, "detr_detect: separate format needs matching [1, N, 4] boxes and [1, N, scores]");
      boxes_index_ = param_.boxes_output;
      scores_index_ = param_.scores_output;
    }
    output_shapes_.clear();
    for (const auto& output : outputs)
      output_shapes_.push_back(output.shape);
    return true;
  }

protected:
  auto decode(DetectResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    KZ_CHECK_RETURN(outputs.size() == output_shapes_.size(), false, "detr_detect: output count changed");
    for (std::size_t i = 0; i < outputs.size(); ++i)
      KZ_CHECK_RETURN(outputs[i]->shape() == output_shapes_[i], false, "detr_detect: output shape changed");
    KZ_CHECK_RETURN(transform.dst_roi.width > 0 && transform.dst_roi.height > 0, false,
                    "detr_detect: destination region is empty");
    const auto* boxes = float_data(*outputs[boxes_index_], boxes_scratch_);
    if (boxes == nullptr)
      return false;
    const bool combined = param_.output_format == DetrOutputFormat::Combined;
    const auto* scores = combined ? boxes + 4 : float_data(*outputs[scores_index_], scores_scratch_);
    if (scores == nullptr)
      return false;
    const auto count = static_cast<std::size_t>(output_shapes_[boxes_index_][1]);
    const auto box_step = combined ? 4 + score_channels_ : 4;
    const auto score_step = combined ? box_step : score_channels_;
    const core::Rect2f bounds{static_cast<float>(transform.src_roi.x), static_cast<float>(transform.src_roi.y),
                              static_cast<float>(transform.src_roi.width),
                              static_cast<float>(transform.src_roi.height)};
    for (std::size_t i = 0; i < count; ++i) {
      const auto* box = boxes + i * box_step;
      if (!std::isfinite(box[0]) || !std::isfinite(box[1]) || !std::isfinite(box[2]) || !std::isfinite(box[3]) ||
          box[2] <= 0.0F || box[3] <= 0.0F)
        continue;
      const auto* logits = scores + i * score_step;
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
          value = sigmoid(value);
        else if (param_.score_activation == DetrScoreActivation::Softmax)
          value = std::exp(value - max_logit) / denominator;
        if (value > score) {
          cls = c - (param_.background_class >= 0 && c > static_cast<std::size_t>(param_.background_class) ? 1U : 0U);
          score = value;
        }
      }
      if (!(score > param_.confidence_threshold[cls]))
        continue;
      const auto [l, t, r, b] =
          center_box(box[0] * static_cast<float>(input_size_.width), box[1] * static_cast<float>(input_size_.height),
                     box[2] * static_cast<float>(input_size_.width), box[3] * static_cast<float>(input_size_.height));
      const auto rect = core::intersect(to_source(core::Rect2f::from_points({l, t}, {r, b}), transform), bounds);
      if (!rect.empty())
        result[cls].push_back({rect, score, static_cast<std::int64_t>(cls)});
    }
    return true;
  }

private:
  DetrParam param_;
  core::Size input_size_;
  Vector<core::Shape> output_shapes_;
  std::size_t boxes_index_ = 0;
  std::size_t scores_index_ = 0;
  std::size_t score_channels_ = 0;
  core::Buffer boxes_scratch_{core::MemoryType::Cpu, core::DataType::Float32};
  core::Buffer scores_scratch_{core::MemoryType::Cpu, core::DataType::Float32};
};

} // namespace

auto make_detr_detect() -> UniquePtr<DetectPostprocessor> {
  return std::make_unique<DetrDetect>();
}

} // namespace hrt::postprocess
