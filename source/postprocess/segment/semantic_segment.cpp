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
    if (engine.proto().output_layout != engine::DataLayout::Nchw || outputs.size() != 1 || !engine.input_size() ||
        (outputs[0].shape.size() != 3 && outputs[0].shape.size() != 4) || outputs[0].shape[0] != 1) {
      KZ_LOG_ERROR("semantic_segment: needs one image input and one [1, C, H, W] or [1, H, W] output");
      return false;
    }
    const auto type = outputs[0].data_type;
    const auto labels = type == core::DataType::Int32 || type == core::DataType::Int64;
    const auto channels = outputs[0].shape.size() == 4 ? outputs[0].shape[1] : 1;
    KZ_CHECK_RETURN(type == core::DataType::Float32 || type == core::DataType::Int8 || (labels && channels == 1), false,
                    "semantic_segment: needs float32 or int8 scores, or int32 or int64 labels in one channel");
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
    if (!labels_at_output(outputs[0], channels, height, width))
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
  auto labels_at_output(const SharedPtr<core::Buffer>& buffer, int channels, int height, int width) -> bool {
    const auto& output = *buffer;
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
    auto* out = labels_.ptr<std::uint16_t>();
    // Quantized scores keep their order for a positive scale, so int8 maps
    // skip dequantization.
    if (channels > 1 && output.data_type() == core::DataType::Int8 && output.is_contiguous() && output.scale > 0.0F) {
      argmax_channels(static_cast<const std::int8_t*>(output.memory()->cpu()), channels, plane, out);
      return true;
    }
    const auto floats = float_view(buffer, scratch_);
    if (floats == nullptr)
      return false;
    const auto* data = static_cast<const float*>(floats->memory()->cpu());
    if (channels == 1) {
      for (std::size_t i = 0; i < plane; ++i)
        out[i] = static_cast<std::uint16_t>(std::lround(data[i]));
      return true;
    }
    argmax_channels(data, channels, plane, out);
    return true;
  }

  // Channel-major argmax: each pass reads one channel plane in order against
  // the running maximum, instead of striding a plane per channel per pixel.
  template <typename T>
  auto argmax_channels(const T* data, int channels, std::size_t plane, std::uint16_t* out) -> void {
    best_.assign(data, data + plane);
    std::fill(out, out + plane, std::uint16_t{0});
    auto* best = best_.data();
    for (int c = 1; c < channels; ++c) {
      const auto* channel = data + static_cast<std::size_t>(c) * plane;
      const auto label = static_cast<std::uint16_t>(c);
      for (std::size_t i = 0; i < plane; ++i) {
        const auto value = static_cast<float>(channel[i]);
        const auto greater = value > best[i];
        best[i] = greater ? value : best[i];
        out[i] = greater ? label : out[i];
      }
    }
  }

  SharedPtr<core::Buffer> scratch_;
  Vector<float> best_;
  cv::Mat labels_;
  cv::Mat source_labels_;
};

} // namespace

auto make_semantic_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<SemanticSegment>();
}

} // namespace hrt::postprocess
