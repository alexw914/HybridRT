// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/segment/semantic_segment.cpp
// Description: Decodes semantic segmentation into per-class binary masks.
// -----------------------------------------------------------------------------

#include <kitzoo/core/macro.hpp>
#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <bitset>
#include <opencv2/imgproc.hpp>
#include <type_traits>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#include "include/postprocess/common.hpp"
#include "include/postprocess/params.hpp"

namespace hrt::postprocess {

namespace {

// Float argmax over 16 pixels per step with NEON or SSE2, the running maxima
// and labels kept in registers across all channels; returns the pixels done.
// Selection matches the scalar loop: a later channel wins only when strictly
// greater, so NaN never replaces the maximum.
auto argmax_vector(const float* data, int channels, std::size_t plane, std::uint16_t* out) -> std::size_t {
  std::size_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (; i + 16 <= plane; i += 16) {
    Array<float32x4_t, 4> best{vld1q_f32(data + i), vld1q_f32(data + i + 4), vld1q_f32(data + i + 8),
                               vld1q_f32(data + i + 12)};
    Array<uint16x8_t, 2> labels{vdupq_n_u16(0), vdupq_n_u16(0)};
    for (int c = 1; c < channels; ++c) {
      const auto* channel = data + static_cast<std::size_t>(c) * plane + i;
      const auto label = vdupq_n_u16(static_cast<std::uint16_t>(c));
      Array<uint32x4_t, 4> greater{};
      for (std::size_t k = 0; k < 4; ++k) {
        const auto value = vld1q_f32(channel + k * 4);
        greater[k] = vcgtq_f32(value, best[k]);
        best[k] = vbslq_f32(greater[k], value, best[k]);
      }
      for (std::size_t k = 0; k < 2; ++k)
        labels[k] = vbslq_u16(vcombine_u16(vmovn_u32(greater[k * 2]), vmovn_u32(greater[k * 2 + 1])), label, labels[k]);
    }
    vst1q_u16(out + i, labels[0]);
    vst1q_u16(out + i + 8, labels[1]);
  }
#elif defined(__SSE2__) || defined(_M_X64)
  for (; i + 16 <= plane; i += 16) {
    // GCC drops SIMD type attributes when they are used as template arguments.
    __m128 best[4]{_mm_loadu_ps(data + i), _mm_loadu_ps(data + i + 4), _mm_loadu_ps(data + i + 8),
                   _mm_loadu_ps(data + i + 12)};
    __m128i labels[2]{_mm_setzero_si128(), _mm_setzero_si128()};
    for (int c = 1; c < channels; ++c) {
      const auto* channel = data + static_cast<std::size_t>(c) * plane + i;
      const auto label = _mm_set1_epi16(static_cast<short>(c));
      __m128 greater[4]{};
      for (std::size_t k = 0; k < 4; ++k) {
        const auto value = _mm_loadu_ps(channel + k * 4);
        greater[k] = _mm_cmpgt_ps(value, best[k]);
        best[k] = _mm_or_ps(_mm_and_ps(greater[k], value), _mm_andnot_ps(greater[k], best[k]));
      }
      for (std::size_t k = 0; k < 2; ++k) {
        // All-ones and zero lanes stay so when packed to 16 bits.
        const auto mask = _mm_packs_epi32(_mm_castps_si128(greater[k * 2]), _mm_castps_si128(greater[k * 2 + 1]));
        labels[k] = _mm_or_si128(_mm_and_si128(mask, label), _mm_andnot_si128(mask, labels[k]));
      }
    }
    _mm_storeu_si128(static_cast<__m128i*>(static_cast<void*>(out + i)), labels[0]);
    _mm_storeu_si128(static_cast<__m128i*>(static_cast<void*>(out + i + 8)), labels[1]);
  }
#else
  static_cast<void>(data);
  static_cast<void>(channels);
  static_cast<void>(plane);
  static_cast<void>(out);
#endif
  return i;
}

// Channel argmax of [C, plane] scores: blocks of pixels against every channel,
// so the running maxima stay in cache instead of a pass over the whole map
// per channel. Float scores take argmax_vector() first.
template <typename T>
auto argmax_channels(const T* data, int channels, std::size_t plane, std::uint16_t* out) -> void {
  constexpr std::size_t kBlock = 64;
  std::size_t first = 0;
  if constexpr (std::is_same_v<T, float>)
    first = argmax_vector(data, channels, plane, out);
  Array<T, kBlock> best{};
  Array<std::uint16_t, kBlock> labels{};
  for (auto start = first; start < plane; start += kBlock) {
    const auto count = std::min(kBlock, plane - start);
    std::copy_n(data + start, count, best.begin());
    std::fill_n(labels.begin(), count, std::uint16_t{0});
    for (int c = 1; c < channels; ++c) {
      const auto* channel = data + static_cast<std::size_t>(c) * plane + start;
      const auto label = static_cast<std::uint16_t>(c);
      for (std::size_t k = 0; k < count; ++k) {
        const auto greater = channel[k] > best[k];
        best[k] = greater ? channel[k] : best[k];
        labels[k] = greater ? label : labels[k];
      }
    }
    std::copy_n(labels.begin(), count, out + start);
  }
}

// One output: class scores [1, C, H, W] (argmax) or class indexes
// [1, 1, H, W] / [1, H, W], at any resolution covering the network input.
class SemanticSegment final : public SegmentPostprocessor {
public:
  auto init(const kitzoo::json::Json& /*config*/, const engine::Engine& engine) -> bool override {
    const auto& outputs = engine.output_protos();
    if (engine.proto().output_layout != engine::DataLayout::Nchw || outputs.size() != 1 ||
        (outputs[0].shape.size() != 3 && outputs[0].shape.size() != 4) || outputs[0].shape[0] < 1) {
      KZ_LOG_ERROR("semantic_segment: needs one [batch, C, H, W] or [batch, H, W] output");
      return false;
    }
    const auto type = outputs[0].data_type;
    const auto labels = type == core::DataType::Int32 || type == core::DataType::Int64;
    const auto channels = outputs[0].shape.size() == 4 ? outputs[0].shape[1] : 1;
    KZ_CHECK_RETURN(type == core::DataType::Float32 || type == core::DataType::Int8 || (labels && channels == 1), false,
                    "semantic_segment: needs float32 or int8 scores, or int32 or int64 labels in one channel");
    // Contiguous multi-channel int8 scores are ranked in place.
    scratch_.assign(1, core::Buffer(core::MemoryType::Cpu, core::DataType::Float32));
    if (type == core::DataType::Int8 && (channels == 1 || !outputs[0].stride.empty()))
      scratch_[0].reshape(outputs[0].shape);
    return true;
  }

  // No parameters besides num and type.
  KZ_NODISCARD auto check_config(const kitzoo::json::Json& config) const -> Vector<String> override {
    core::Problems problems;
    core::check_json(config, kitzoo::json::Json::object(), "postprocess", {"type"}, problems);
    return problems;
  }

protected:
  auto decode(SegmentResultMap& result, const core::Buffers& outputs, const preprocess::ImageTransform& transform)
      -> bool override {
    const auto& output = *outputs[0];
    const auto& shape = output.shape();
    const auto channels = shape.size() == 4 ? static_cast<int>(shape[1]) : 1;
    const auto height = static_cast<int>(shape[shape.size() - 2]);
    const auto width = static_cast<int>(shape.back());
    if (!output_labels(output, channels, height, width))
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
      object.box = {core::Rect2f(src), 1.0F, static_cast<std::int64_t>(cls), -1};
      object.mask.rect = src;
      object.mask.data.resize(pixels);
      cv::Mat mask(src.height, src.width, CV_8U, object.mask.data.data());
      cv::compare(source_labels_, static_cast<double>(cls), mask, cv::CMP_EQ);
      cv::bitwise_and(mask, cv::Scalar(1), mask);
    }
    return true;
  }

private:
  auto output_labels(const core::Buffer& output, int channels, int height, int width) -> bool {
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
    // Quantized scores keep their order (the scale is positive), so int8 maps
    // skip dequantization.
    if (channels > 1 && output.data_type() == core::DataType::Int8 && output.is_contiguous()) {
      argmax_channels(static_cast<const std::int8_t*>(output.memory()->cpu()), channels, plane, out);
      return true;
    }
    const core::Buffer* floats = &output;
    if (floats->data_type() != core::DataType::Float32) {
      dequantize_buffer(*floats, scratch_[0]);
      floats = &scratch_[0];
    }
    const auto* data = static_cast<const float*>(floats->memory()->cpu());
    if (channels == 1) {
      for (std::size_t i = 0; i < plane; ++i)
        out[i] = static_cast<std::uint16_t>(std::lround(data[i]));
      return true;
    }
    argmax_channels(data, channels, plane, out);
    return true;
  }

  Vector<core::Buffer> scratch_;
  cv::Mat labels_;
  cv::Mat source_labels_;
};

} // namespace

auto create_semantic_segment() -> UniquePtr<SegmentPostprocessor> {
  return std::make_unique<SemanticSegment>();
}

} // namespace hrt::postprocess
