// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/semantic_segment.cpp
// Description: Decodes semantic segmentation into per-class binary masks.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <bitset>
#include <opencv2/imgproc.hpp>

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// One output: class scores [1, C, H, W] (argmax) or class indexes
// [1, 1, H, W] / [1, H, W], at any resolution covering the network input.
class SemanticSegment final : public SegmentPostprocessor {
public:
  // No parameters besides num and type.
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_json(config, kitzoo::json::Json::object(), "postprocess", {"type"}, problems);
    return problems;
  }

  auto init(const kitzoo::json::Json& /*config*/, const engine::Engine& engine) -> bool override {
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || outputs.size() != 1 || !engine_input_size(engine) ||
        (outputs[0].shape.size() != 3 && outputs[0].shape.size() != 4) || outputs[0].shape[0] != 1) {
      KZ_LOG_ERROR("semantic_segment: needs one image input and one [1, C, H, W] or [1, H, W] output");
      return false;
    }
    return true;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto& output = *outputs[0];
    const auto& shape = output.shape();
    const auto channels = shape.size() == 4 ? static_cast<int>(shape[1]) : 1;
    const auto height = static_cast<int>(shape[shape.size() - 2]);
    const auto width = static_cast<int>(shape.back());
    if (!labels_at_output(output, channels, height, width))
      return false;

    // Source pixel center -> network coordinate -> output pixel.
    const auto& dst = transform.dst_roi;
    const auto& src = transform.src_roi;
    const auto sx = static_cast<double>(width) / transform.input_size.width;
    const auto sy = static_cast<double>(height) / transform.input_size.height;
    const auto rw = static_cast<double>(dst.width) / src.width;
    const auto rh = static_cast<double>(dst.height) / src.height;
    const cv::Matx23d inverse(rw * sx, 0.0, (0.5 * rw + dst.x) * sx - 0.5, 0.0, rh * sy, (0.5 * rh + dst.y) * sy - 0.5);
    cv::warpAffine(labels_, source_labels_, inverse, {src.width, src.height}, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    std::bitset<65536> present;
    const auto* labels = source_labels_.ptr<std::uint16_t>();
    const auto pixels = source_labels_.total();
    for (std::size_t i = 0; i < pixels; ++i)
      present.set(labels[i]);
    for (std::size_t cls = 0; cls < present.size(); ++cls) {
      if (!present.test(cls))
        continue;
      auto& object = result[static_cast<std::int64_t>(cls)].emplace_back();
      object.box = {{static_cast<float>(src.x), static_cast<float>(src.y), static_cast<float>(src.width),
                     static_cast<float>(src.height)},
                    1.0F,
                    static_cast<std::int64_t>(cls),
                    -1};
      object.mask.rect = src;
      object.mask.data.resize(pixels);
      cv::Mat mask(src.height, src.width, CV_8U, object.mask.data.data());
      cv::compare(source_labels_, static_cast<double>(cls), mask, cv::CMP_EQ);
      cv::bitwise_and(mask, cv::Scalar(1), mask);
    }
    return true;
  }

private:
  auto labels_at_output(const core::Buffer& output, int channels, int height, int width) -> bool {
    labels_.create(height, width, CV_16U);
    const auto plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (channels == 1 && (output.data_type() == core::DataType::Int32 || output.data_type() == core::DataType::Int64)) {
      if (!output.is_contiguous())
        return false;
      for (std::size_t i = 0; i < plane; ++i) {
        const auto value = output.data_type() == core::DataType::Int32
                               ? static_cast<std::int64_t>(static_cast<const std::int32_t*>(output.memory()->cpu())[i])
                               : static_cast<const std::int64_t*>(output.memory()->cpu())[i];
        labels_.ptr<std::uint16_t>()[i] = static_cast<std::uint16_t>(value);
      }
      return true;
    }
    const auto* data = float_data(output, scratch_);
    if (data == nullptr)
      return false;
    auto* out = labels_.ptr<std::uint16_t>();
    for (std::size_t i = 0; i < plane; ++i) {
      if (channels == 1) {
        out[i] = static_cast<std::uint16_t>(std::lround(data[i]));
        continue;
      }
      std::size_t best = 0;
      for (std::size_t c = 1; c < static_cast<std::size_t>(channels); ++c) {
        if (data[c * plane + i] > data[best * plane + i])
          best = c;
      }
      out[i] = static_cast<std::uint16_t>(best);
    }
    return true;
  }

  core::Buffer scratch_{core::MemoryType::Cpu, core::DataType::Float32};
  cv::Mat labels_;
  cv::Mat source_labels_;
};

} // namespace

auto make_semantic_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<SemanticSegment>();
}

} // namespace hrt::postprocess
