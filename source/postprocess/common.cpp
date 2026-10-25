// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/postprocess/common.cpp
// Description: Implements shared postprocessing helpers and host staging of
//              engine outputs.
// -----------------------------------------------------------------------------

#include "include/postprocess/common.hpp"

#include <kitzoo/log/logger.hpp>

#include <algorithm>
#include <numeric>

// The vector int8 dequantization follows the target: AArch64 NEON, SSE2 on
// x86, the scalar loop elsewhere.
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#include <hrt/postprocess/postprocessor.hpp>

#include "include/core/batch_scheduler.hpp"

namespace hrt::postprocess {

PostprocessorBase::~PostprocessorBase() = default;

auto PostprocessorBase::to_host(const core::Buffers& outputs, core::Stream* stream) -> const core::Buffers* {
  view_.resize(outputs.size());
  host_.resize(outputs.size());
  bool copied = false;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    if (output->memory()->have_cpu()) {
      view_[i] = output;
      continue;
    }
    if (host_[i] == nullptr)
      host_[i] = std::make_shared<core::Buffer>(core::MemoryType::Cpu);
    if (!core::Buffer::copy(*host_[i], *output, core::CopyPolicy::AutoDstStride, stream))
      return nullptr;
    host_[i]->set_quantization(output->scale(), output->zero_point());
    view_[i] = host_[i];
    copied = true;
  }
  if (copied && stream != nullptr)
    stream->synchronize();
  return &view_;
}

auto PostprocessorBase::sample_outputs(const core::Buffers& host, std::size_t index) -> const core::Buffers* {
  const auto single = [](const auto& buffer) -> bool { return buffer->shape().empty() || buffer->shape()[0] == 1; };
  if (index == 0 && std::all_of(host.begin(), host.end(), single))
    return &host;
  sample_.resize(host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    sample_[i] = core::batch_sample(*host[i], index, false);
    if (sample_[i] == nullptr)
      return nullptr;
  }
  return &sample_;
}

auto expand_per_class(Vector<float>& values, std::size_t class_num, const char* name) -> bool {
  KZ_CHECK_RETURN(class_num > 0, false, "postprocess: class_num must be positive");
  if (values.size() == 1 && class_num > 1)
    values.resize(class_num, values.front());
  if (values.size() != class_num) {
    KZ_LOG_ERROR("postprocess: {} needs 1 or {} values, got {}", name, class_num, values.size());
    return false;
  }
  if (!std::all_of(values.begin(), values.end(), [](float v) -> bool { return v >= 0.0F && v <= 1.0F; })) {
    KZ_LOG_ERROR("postprocess: {} values must be in [0, 1]", name);
    return false;
  }
  return true;
}

auto is_float_outputs(const engine::Engine& engine, std::string_view name) -> bool {
  for (const auto& output : engine.output_protos()) {
    KZ_CHECK_RETURN(output.data_type == core::DataType::Float32 || output.data_type == core::DataType::Int8, false,
                    "{}: outputs must be float32 or int8, got {}", name, to_string(output.data_type));
  }
  return true;
}

auto is_4d_outputs(const engine::Engine& engine, std::string_view name) -> bool {
  const auto& outputs = engine.output_protos();
  KZ_CHECK_RETURN(!outputs.empty() &&
                      std::all_of(outputs.begin(), outputs.end(),
                                  [](const core::BufferProto& output) -> bool { return output.shape.size() == 4; }),
                  false, "{}: needs 4-D head outputs", name);
  return true;
}

auto head_channels(const core::BufferProto& proto, engine::DataLayout layout) -> std::int64_t {
  return proto.shape[layout == engine::DataLayout::Nchw ? 1 : 3];
}

auto head_stride(const core::BufferProto& proto, engine::DataLayout layout, core::Size input) -> core::Point2f {
  const auto rows = proto.shape[layout == engine::DataLayout::Nchw ? 2 : 1];
  const auto cols = proto.shape[layout == engine::DataLayout::Nchw ? 3 : 2];
  return {static_cast<float>(static_cast<double>(input.width) / static_cast<double>(cols)),
          static_cast<float>(static_cast<double>(input.height) / static_cast<double>(rows))};
}

HeadGridView::HeadGridView(const core::Buffer& buffer, std::optional<engine::DataLayout> layout)
    : int8_(buffer.data_type() == core::DataType::Int8), scale_(buffer.scale()), zero_point_(buffer.zero_point()) {
  const auto& shape = buffer.shape();
  const auto& stride = buffer.stride();
  const auto dims = shape.size();
  const auto item = int8_ ? sizeof(std::int8_t) : sizeof(float);
  if (!layout) {
    rows_ = 1;
    cols_ = shape[dims - 1];
    col_step_ = 1;
    channel_step_ = stride[dims - 1] / item;
  } else if (*layout == engine::DataLayout::Nchw) {
    rows_ = shape[dims - 2];
    cols_ = shape[dims - 1];
    row_step_ = stride[dims - 1] / item;
    col_step_ = 1;
    channel_step_ = stride[dims - 2] / item;
  } else {
    rows_ = shape[dims - 3];
    cols_ = shape[dims - 2];
    row_step_ = stride[dims - 2] / item;
    col_step_ = stride[dims - 1] / item;
    channel_step_ = 1;
  }
  data_ = buffer.memory()->cpu();
}

auto HeadGridView::raw_bound(float threshold) const -> float {
  if (!int8_)
    return threshold;
  if (!(scale_ > 0.0F))
    return std::numeric_limits<float>::lowest();
  const auto estimate = std::floor(static_cast<double>(threshold) / static_cast<double>(scale_) + zero_point_);
  auto bound = static_cast<std::int32_t>(std::clamp(estimate, -128.0, 128.0));
  // Rounding down admits the zero-point bin for small positive thresholds,
  // defeating block skipping on quantized decoded heads. Match float rounding.
  while (bound < 128 && core::dequantize(static_cast<std::int8_t>(bound), scale_, zero_point_) < threshold)
    ++bound;
  while (bound > -128 && core::dequantize(static_cast<std::int8_t>(bound - 1), scale_, zero_point_) >= threshold)
    --bound;
  return static_cast<float>(bound);
}

namespace {

// raw_bound() in the element type: every int8 reaches -129 and none reaches 128.
template <typename T>
auto typed_bound(float bound) {
  if constexpr (std::is_same_v<T, float>)
    return bound;
  else
    return static_cast<std::int32_t>(std::clamp(bound, -129.0F, 128.0F));
}

template <typename T>
auto scan_max_cells(const HeadGridView& grid, std::size_t first, std::size_t count, float raw_bound, Vector<T>& best,
                    Vector<std::uint32_t>& cells) -> void {
  const auto rows = grid.rows();
  const auto cols = grid.cols();
  const auto bound = typed_bound<T>(raw_bound);
  cells.clear();
  if (count == 1) {
    const auto step = grid.col_step();
    for (std::int64_t y = 0; y < rows; ++y) {
      const auto* in = grid.row<T>(y, first);
      for (std::int64_t x = 0; x < cols; ++x) {
        if (in[static_cast<std::size_t>(x) * step] >= bound)
          cells.push_back(static_cast<std::uint32_t>(y * cols + x));
      }
    }
    return;
  }
  best.assign(static_cast<std::size_t>(rows * cols), std::numeric_limits<T>::lowest());
  if (grid.col_step() == 1) {
    for (std::size_t c = first; c < first + count; ++c) {
      for (std::int64_t y = 0; y < rows; ++y) {
        const auto* in = grid.row<T>(y, c);
        auto* out = best.data() + y * cols;
        for (std::int64_t x = 0; x < cols; ++x)
          out[x] = std::max(out[x], in[x]);
      }
    }
  } else {
    const auto step = grid.col_step();
    for (std::int64_t y = 0; y < rows; ++y) {
      for (std::int64_t x = 0; x < cols; ++x) {
        auto value = std::numeric_limits<T>::lowest();
        for (std::size_t c = first; c < first + count; ++c)
          value = std::max(value, grid.row<T>(y, c)[static_cast<std::size_t>(x) * step]);
        best[static_cast<std::size_t>(y * cols + x)] = value;
      }
    }
  }
  for (std::size_t i = 0; i < best.size(); ++i) {
    if (best[i] >= bound)
      cells.push_back(static_cast<std::uint32_t>(i));
  }
}

// Whether any of 16 consecutive values reaches `bound`, a typed_bound(); a
// filter before the scalar check, so it may report blocks the check rejects
// but must not miss one. NEON and SSE2 compare all 16 at once. An int8 bound
// spans [-129, 128]; saturated to int8, -129 still passes every value and 128
// only adds blocks.
template <typename T>
auto any_at_least(const T* values, decltype(typed_bound<T>(0.0F)) bound) -> bool {
#if defined(__ARM_NEON) && defined(__aarch64__)
  if constexpr (std::is_same_v<T, float>) {
    const auto limit = vdupq_n_f32(bound);
    const auto hits =
        vorrq_u32(vorrq_u32(vcgeq_f32(vld1q_f32(values), limit), vcgeq_f32(vld1q_f32(values + 4), limit)),
                  vorrq_u32(vcgeq_f32(vld1q_f32(values + 8), limit), vcgeq_f32(vld1q_f32(values + 12), limit)));
    return vmaxvq_u32(hits) != 0;
  } else {
    const auto limit = vdupq_n_s8(static_cast<std::int8_t>(std::clamp(bound, -128, 127)));
    return vmaxvq_u8(vcgeq_s8(vld1q_s8(values), limit)) != 0;
  }
#elif defined(__SSE2__) || defined(_M_X64)
  if constexpr (std::is_same_v<T, float>) {
    const auto limit = _mm_set1_ps(bound);
    const auto hits = _mm_or_ps(
        _mm_or_ps(_mm_cmpge_ps(_mm_loadu_ps(values), limit), _mm_cmpge_ps(_mm_loadu_ps(values + 4), limit)),
        _mm_or_ps(_mm_cmpge_ps(_mm_loadu_ps(values + 8), limit), _mm_cmpge_ps(_mm_loadu_ps(values + 12), limit)));
    return _mm_movemask_ps(hits) != 0;
  } else {
    const auto limit = _mm_set1_epi8(static_cast<char>(std::clamp(bound, -128, 127)));
    const auto bytes = _mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(values)));
    return _mm_movemask_epi8(_mm_or_si128(_mm_cmpgt_epi8(bytes, limit), _mm_cmpeq_epi8(bytes, limit))) != 0;
  }
#else
  bool any = false;
  for (std::size_t k = 0; k < 16; ++k)
    any |= values[k] >= bound;
  return any;
#endif
}

template <typename T>
auto scan_channel_cells(const HeadGridView& grid, std::size_t channel, float threshold, Vector<std::uint32_t>& cells)
    -> void {
  const auto* values = grid.row<T>(0, channel);
  const auto step = grid.col_step();
  const auto bound = typed_bound<T>(grid.raw_bound(threshold));
  const auto cols = static_cast<std::size_t>(grid.cols());
  cells.clear();
  std::size_t x = 0;
  const auto check = [&](std::size_t end) -> void {
    for (; x < end; ++x) {
      if (values[x * step] >= bound && grid.at(0, static_cast<std::int64_t>(x), channel) > threshold)
        cells.push_back(static_cast<std::uint32_t>(x));
    }
  };
  // Scores are mostly below the threshold: contiguous rows skip 16 values at
  // a time when none reaches the bound.
  if (step == 1) {
    while (x + 16 <= cols) {
      if (any_at_least<T>(values + x, bound))
        check(x + 16);
      else
        x += 16;
    }
  }
  check(cols);
}

} // namespace

auto CellScanner::scan_max_score(const HeadGridView& grid, std::size_t first, std::size_t count, float bound)
    -> const Vector<std::uint32_t>& {
  if (grid.int8())
    scan_max_cells(grid, first, count, bound, int8_best_, cells_);
  else
    scan_max_cells(grid, first, count, bound, float_best_, cells_);
  return cells_;
}

auto CellScanner::scan_channel(const HeadGridView& grid, std::size_t channel, float threshold)
    -> const Vector<std::uint32_t>& {
  if (grid.int8())
    scan_channel_cells<std::int8_t>(grid, channel, threshold, cells_);
  else
    scan_channel_cells<float>(grid, channel, threshold, cells_);
  return cells_;
}

auto create_scratch(const engine::Engine& engine) -> Vector<core::Buffer> {
  Vector<core::Buffer> scratch;
  for (const auto& output : engine.output_protos()) {
    auto& buffer = scratch.emplace_back(core::MemoryType::Cpu, core::DataType::Float32);
    if (output.data_type == core::DataType::Int8)
      buffer.reshape(output.shape);
  }
  return scratch;
}

namespace {

// core::dequantize() over `count` values, 16 per step with AArch64 NEON or
// SSE2 and the tail in the scalar loop. The vector paths subtract the zero
// point and then multiply without fusing, so they match the scalar loop bit
// for bit.
auto dequantize_values(const std::int8_t* in, std::size_t count, float scale, std::int32_t zero_point, float* out)
    -> void {
  std::size_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
  // Each table moves four bytes into the top byte of a 32-bit lane and zeroes
  // the rest (index 255 is out of range); a conversion with 24 fraction bits
  // then yields the signed byte exactly, without widening steps.
  static constexpr Array<Array<std::uint8_t, 16>, 4> kLanes{{
      {255, 255, 255, 0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3},
      {255, 255, 255, 4, 255, 255, 255, 5, 255, 255, 255, 6, 255, 255, 255, 7},
      {255, 255, 255, 8, 255, 255, 255, 9, 255, 255, 255, 10, 255, 255, 255, 11},
      {255, 255, 255, 12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15},
  }};
  const Array<uint8x16_t, 4> lanes{vld1q_u8(kLanes[0].data()), vld1q_u8(kLanes[1].data()), vld1q_u8(kLanes[2].data()),
                                   vld1q_u8(kLanes[3].data())};
  const auto zero = vdupq_n_f32(static_cast<float>(zero_point));
  const auto factor = vdupq_n_f32(scale);
  for (; i + 16 <= count; i += 16) {
    const auto bytes = vld1q_u8(static_cast<const std::uint8_t*>(static_cast<const void*>(in + i)));
    for (std::size_t k = 0; k < 4; ++k) {
      const auto values = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(bytes, lanes[k])), 24);
      vst1q_f32(out + i + k * 4, vmulq_f32(vsubq_f32(values, zero), factor));
    }
  }
#elif defined(__SSE2__) || defined(_M_X64)
  const auto zero = _mm_set1_ps(static_cast<float>(zero_point));
  const auto factor = _mm_set1_ps(scale);
  const auto store = [&](__m128i values, float* dst) -> void {
    _mm_storeu_ps(dst, _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(values), zero), factor));
  };
  for (; i + 16 <= count; i += 16) {
    const auto bytes = _mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(in + i)));
    // SSE2 has no sign extension: each value goes to the high half of a wider
    // lane and an arithmetic shift brings it down.
    const auto low = _mm_srai_epi16(_mm_unpacklo_epi8(bytes, bytes), 8);
    const auto high = _mm_srai_epi16(_mm_unpackhi_epi8(bytes, bytes), 8);
    store(_mm_srai_epi32(_mm_unpacklo_epi16(low, low), 16), out + i);
    store(_mm_srai_epi32(_mm_unpackhi_epi16(low, low), 16), out + i + 4);
    store(_mm_srai_epi32(_mm_unpacklo_epi16(high, high), 16), out + i + 8);
    store(_mm_srai_epi32(_mm_unpackhi_epi16(high, high), 16), out + i + 12);
  }
#endif
  for (; i < count; ++i)
    out[i] = core::dequantize(in[i], scale, zero_point);
}

} // namespace

auto dequantize_buffer(const core::Buffer& output, core::Buffer& scratch) -> void {
  const auto* base = static_cast<const std::uint8_t*>(output.memory()->cpu());
  auto* out = static_cast<float*>(scratch.memory()->mutable_cpu());
  const auto convert = [&](const std::uint8_t* in, std::size_t count) -> void {
    dequantize_values(static_cast<const std::int8_t*>(static_cast<const void*>(in)), count, output.scale(),
                      output.zero_point(), out);
    out += count;
  };
  const auto count = static_cast<std::size_t>(output.volume());
  if (output.is_contiguous()) {
    convert(base, count);
    return;
  }
  // Rows of the last dimension are contiguous; the outer index of a row is
  // decomposed against the shape, dimension k - 1 stepping stride[k] bytes.
  const auto& shape = output.shape();
  const auto& stride = output.stride();
  const auto cols = static_cast<std::size_t>(shape.back());
  for (std::size_t row = 0; cols != 0 && row < count / cols; ++row) {
    std::size_t offset = 0;
    auto rest = row;
    for (auto k = shape.size() - 1; k > 0; --k) {
      const auto dim = static_cast<std::size_t>(shape[k - 1]);
      offset += rest % dim * stride[k];
      rest /= dim;
    }
    convert(base + offset, cols);
  }
}

} // namespace hrt::postprocess
