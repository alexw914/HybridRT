// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/classify/general_classify.cpp
// Description: Implements classification score extraction and the classify
//              postprocessor factory.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <cmath>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

enum class ClassifyActivation {
  None,
  Sigmoid,
  Softmax,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ClassifyActivation, {
                                                     {ClassifyActivation::None, "none"},
                                                     {ClassifyActivation::Sigmoid, "sigmoid"},
                                                     {ClassifyActivation::Softmax, "softmax"},
                                                 })

struct GeneralClassifyParam {
  ClassifyActivation classify_postprocess_type = ClassifyActivation::None;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(GeneralClassifyParam, classify_postprocess_type)

class GeneralClassify final : public ClassifyPostprocessor {
public:
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    return check_section<GeneralClassifyParam>(config);
  }

  auto init(const kitzoo::json::Json& config, const engine::Engine& engine) -> bool override {
    if (!float_outputs(engine, "general_classify"))
      return false;
    activation_ = config.get<GeneralClassifyParam>().classify_postprocess_type;
    scratch_.assign(engine.output_protos().size(), nullptr);
    return true;
  }

protected:
  auto decode(ClassifyResult& result, const core::Buffers& outputs, const preprocess::ImageTransform& /*transform*/)
      -> bool override {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto floats = float_view(outputs[i], scratch_[i]);
      const auto* data = static_cast<const float*>(floats->memory()->cpu());
      auto& scores = result.emplace_back(data, data + floats->volume());
      if (activation_ == ClassifyActivation::Sigmoid) {
        std::transform(scores.begin(), scores.end(), scores.begin(), core::sigmoid);
      } else if (activation_ == ClassifyActivation::Softmax && !scores.empty()) {
        const auto max = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0F;
        for (auto& score : scores) {
          score = std::exp(score - max);
          sum += score;
        }
        for (auto& score : scores)
          score /= sum;
      }
    }
    return true;
  }

private:
  ClassifyActivation activation_ = ClassifyActivation::None;
  core::Buffers scratch_;
};

} // namespace

auto make_general_classify() -> UniquePtr<ClassifyPostprocessor> {
  return std::make_unique<GeneralClassify>();
}

auto make_classify(std::string_view type) -> UniquePtr<ClassifyPostprocessor> {
  return type == "GeneralClassify" ? make_general_classify() : nullptr;
}

auto create_classify_postprocessor(std::string_view type, const kitzoo::json::Json& config,
                                   const engine::Engine& engine) -> UniquePtr<ClassifyPostprocessor> {
  return checked(make_classify(type), type, config, engine);
}

} // namespace hrt::postprocess
