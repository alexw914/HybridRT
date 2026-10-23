// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/preprocess/image_preprocessor_test.cpp
// Description: Tests preprocessing parameters and resize geometry.
// -----------------------------------------------------------------------------

#include <hrt/core/context.hpp>
#include <hrt/preprocess/image_preprocessor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>

namespace {

auto param_with(hrt::preprocess::ResizeType type) -> hrt::preprocess::ImagePreprocessorParam {
  hrt::preprocess::ImagePreprocessorParam param;
  param.resize_type = type;
  return param;
}

TEST(ImagePreprocessorTest, ParsesParams) {
  auto param = kitzoo::json::Json::parse(R"({
    "resize_type": "center_keep_ratio", "dst_image_format": "inter_bgr",
    "std": [255, 255, 255], "pad_scale": 1.5
  })")
                   .get<hrt::preprocess::ImagePreprocessorParam>();
  EXPECT_EQ(param.resize_type, hrt::preprocess::ResizeType::LetterBox);
  EXPECT_EQ(param.dst_image_format, hrt::core::ImageFormat::InterBgr);
  EXPECT_FLOAT_EQ(param.std[2], 255.0F);
  EXPECT_FLOAT_EQ(param.pad_scale, 1.5F);
  EXPECT_EQ(param.interpolation_mode, hrt::preprocess::InterpolationMode::Linear);
  EXPECT_EQ(param.device, hrt::preprocess::ImagePreprocessorType::Auto);
}

TEST(ImagePreprocessorTest, LetterBoxCentersTheImage) {
  hrt::core::Rect dst_roi;
  hrt::core::Rect src_roi;
  ASSERT_TRUE(hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 640, 640}, {0, 0, 1280, 720}, {1280, 720},
                                                    param_with(hrt::preprocess::ResizeType::LetterBox)));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 140, 640, 360}));
  EXPECT_EQ(src_roi, (hrt::core::Rect{0, 0, 1280, 720}));
}

TEST(ImagePreprocessorTest, LetterBoxRoundsTheScaledSizeAsUltralytics) {
  const auto param = param_with(hrt::preprocess::ResizeType::LetterBox);
  hrt::core::Rect dst_roi;
  hrt::core::Rect src_roi;
  // 768 * 640 / 1366 = 359.82 rounds to 360, as Ultralytics' new_unpad.
  ASSERT_TRUE(
      hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 640, 640}, {0, 0, 1366, 768}, {1366, 768}, param));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 140, 640, 360}));
  // 727 / 2 = 363.5 rounds half to even, as Python's round(), leaving 138 rows of padding on each side.
  ASSERT_TRUE(
      hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 640, 640}, {0, 0, 1280, 727}, {1280, 727}, param));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 138, 640, 364}));
  // A thin source keeps at least one row.
  ASSERT_TRUE(
      hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 640, 640}, {0, 0, 2000, 1}, {2000, 1}, param));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 319, 640, 1}));
}

TEST(ImagePreprocessorTest, KeepRatioAnchorsTopLeft) {
  hrt::core::Rect dst_roi;
  hrt::core::Rect src_roi;
  ASSERT_TRUE(hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 640, 640}, {0, 0, 720, 1280}, {720, 1280},
                                                    param_with(hrt::preprocess::ResizeType::KeepRatio)));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 0, 360, 640}));
}

TEST(ImagePreprocessorTest, CenterCropUsesShortSide) {
  auto param = param_with(hrt::preprocess::ResizeType::CenterCrop);
  param.resize_short_size = 256;
  hrt::core::Rect dst_roi;
  hrt::core::Rect src_roi;
  ASSERT_TRUE(
      hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 224, 224}, {0, 0, 512, 256}, {512, 256}, param));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{0, 0, 224, 224}));
  EXPECT_EQ(src_roi, (hrt::core::Rect{144, 16, 224, 224}));

  param.resize_short_size = -1;
  EXPECT_FALSE(
      hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 224, 224}, {0, 0, 512, 256}, {512, 256}, param));
}

TEST(ImagePreprocessorTest, PadOutsideClipsAtImageBorder) {
  hrt::core::Rect dst_roi;
  hrt::core::Rect src_roi;
  // A 50x100 box at the left edge padded to a square spans x in [-25, 75),
  // so the left quarter of the input stays as padding.
  ASSERT_TRUE(hrt::preprocess::calc_resize_position(dst_roi, src_roi, {0, 0, 200, 200}, {0, 0, 50, 100}, {640, 480},
                                                    param_with(hrt::preprocess::ResizeType::PadOutside)));
  EXPECT_EQ(src_roi, (hrt::core::Rect{0, 0, 75, 100}));
  EXPECT_EQ(dst_roi, (hrt::core::Rect{50, 0, 150, 200}));
}

TEST(ImagePreprocessorTest, TextLineUsesCeilingWidthAndZeroNormalizedPadding) {
  hrt::core::Image source(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterBgr, hrt::core::DataType::Uint8,
                          {5, 3});
  auto* pixels = static_cast<std::uint8_t*>(source.memory()->mutable_cpu());
  std::fill(pixels, pixels + source.buffer()->bytes(), std::uint8_t{255});
  hrt::core::Image output(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::PlanarBgr, hrt::core::DataType::Float32,
                          {16, 4});
  auto processor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  hrt::preprocess::ImagePreprocessorParam param;
  param.dst_image_format = hrt::core::ImageFormat::PlanarBgr;
  param.resize_type = hrt::preprocess::ResizeType::TextLine;
  param.normalize_type = hrt::preprocess::NormalizeType::Roi;
  param.mean = {127.5F, 127.5F, 127.5F};
  param.std = param.mean;
  const auto transform = processor->process(output, source, {}, param);
  ASSERT_TRUE(transform);
  EXPECT_EQ(transform->dst_roi, (hrt::core::Rect{0, 0, 7, 4}));
  const auto* data = static_cast<const float*>(output.memory()->cpu());
  EXPECT_FLOAT_EQ(data[0], 1);
  EXPECT_FLOAT_EQ(data[15], 0);
}

} // namespace

namespace {

// BGR image of one color, optionally with padded rows.
struct HostImage {
  hrt::Vector<std::uint8_t> pixels;
  hrt::core::Image image;
};

auto uniform_bgr(int width, int height, hrt::Array<std::uint8_t, 3> bgr, std::size_t pitch = 0) -> HostImage {
  const auto row = pitch == 0 ? static_cast<std::size_t>(width) * 3 : pitch;
  hrt::Vector<std::uint8_t> pixels(row * static_cast<std::size_t>(height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto* px = pixels.data() + static_cast<std::size_t>(y) * row + static_cast<std::size_t>(x) * 3;
      px[0] = bgr[0];
      px[1] = bgr[1];
      px[2] = bgr[2];
    }
  }
  auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr,
                                      {width, height}, hrt::core::DataType::Uint8, pitch);
  return {std::move(pixels), std::move(*image)};
}

auto planar_at(const hrt::core::Image& image, int plane, int x, int y) -> float {
  const auto* base = static_cast<const std::byte*>(image.memory()->cpu());
  const auto& stride = image.stride();
  const auto* row = base + static_cast<std::size_t>(plane) * stride[1] + static_cast<std::size_t>(y) * stride[2];
  return reinterpret_cast<const float*>(row)[x];
}

TEST(ImagePreprocessorTest, LetterBoxesBgrIntoNormalizedPlanarRgb) {
  // Wrapped with padded rows to exercise strided host input.
  auto src = uniform_bgr(8, 4, {10, 20, 30}, 32);
  hrt::core::Image dst(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::PlanarRgb, hrt::core::DataType::Float32,
                       {8, 8});

  hrt::preprocess::ImagePreprocessorParam param;
  param.resize_type = hrt::preprocess::ResizeType::LetterBox;
  param.std = {255.0F, 255.0F, 255.0F};
  param.default_color = {114.0F, 114.0F, 114.0F};
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);

  auto transform = preprocessor->process(dst, src.image, {}, param);
  ASSERT_TRUE(transform);
  EXPECT_EQ(transform->dst_roi, (hrt::core::Rect{0, 2, 8, 4}));
  EXPECT_EQ(transform->src_roi, (hrt::core::Rect{0, 0, 8, 4}));
  EXPECT_EQ(transform->input_size, (hrt::core::Size{8, 8}));
  EXPECT_FLOAT_EQ(planar_at(dst, 0, 3, 3), 30.0F / 255.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 1, 3, 3), 20.0F / 255.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 2, 3, 3), 10.0F / 255.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 0, 0, 0), 114.0F / 255.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 2, 7, 7), 114.0F / 255.0F);
}

TEST(ImagePreprocessorTest, NormalizesOnlyTheRoiWhenAsked) {
  auto src = uniform_bgr(8, 4, {10, 20, 30});
  hrt::core::Image dst(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::PlanarBgr, hrt::core::DataType::Float32,
                       {8, 8});

  hrt::preprocess::ImagePreprocessorParam param;
  param.resize_type = hrt::preprocess::ResizeType::LetterBox;
  param.normalize_type = hrt::preprocess::NormalizeType::Roi;
  param.mean = {10.0F, 10.0F, 10.0F};
  param.std = {2.0F, 2.0F, 2.0F};
  param.default_color = {1.0F, 2.0F, 3.0F};
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);

  ASSERT_TRUE(preprocessor->process(dst, src.image, {}, param));
  EXPECT_FLOAT_EQ(planar_at(dst, 0, 4, 4), 0.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 2, 4, 4), 10.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 0, 0, 0), 1.0F);
  EXPECT_FLOAT_EQ(planar_at(dst, 2, 0, 7), 3.0F);
}

TEST(ImagePreprocessorTest, ConvertsAndNormalizesAllByteValuesWithChannelOrderAndPadding) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto src = uniform_bgr(256, 4, {0, 0, 0}, 784);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 256; ++x) {
      auto* pixel = src.pixels.data() + y * 784 + x * 3;
      pixel[0] = static_cast<std::uint8_t>(x);
      pixel[1] = static_cast<std::uint8_t>(255 - x);
      pixel[2] = static_cast<std::uint8_t>((x + 127) % 256);
    }
  }
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  ASSERT_NE(processor, nullptr);
  for (const auto source_format : {ImageFormat::InterRgb, ImageFormat::InterBgr}) {
    auto source = Image::wrap(MemoryType::Cpu, src.pixels.data(), source_format, {256, 4}, DataType::Uint8, 784);
    ASSERT_TRUE(source);
    for (const auto target_format : {ImageFormat::PlanarRgb, ImageFormat::PlanarBgr}) {
      Image dst(MemoryType::Cpu, target_format, DataType::Float32, {256, 8}, {25344, 8448, 1056});
      ASSERT_TRUE(dst.valid());
      for (const auto mode : {NormalizeType::Full, NormalizeType::Roi, NormalizeType::None}) {
        for (const bool identity : {false, true}) {
          ImagePreprocessorParam param;
          param.resize_type = ResizeType::LetterBox;
          param.dst_image_format = target_format;
          param.normalize_type = mode;
          param.mean = identity ? hrt::Array<float, 3>{0, 0, 0} : hrt::Array<float, 3>{10, 20, 30};
          param.std = identity ? hrt::Array<float, 3>{1, 1, 1} : hrt::Array<float, 3>{2, 3, 4};
          param.default_color = {11, 22, 33};
          ASSERT_TRUE(processor->process(dst, *source, {}, param));
          const auto swap = (source_format == ImageFormat::InterRgb) != (target_format == ImageFormat::PlanarRgb);
          for (int c = 0; c < 3; ++c) {
            const auto channel = static_cast<std::size_t>(c);
            const auto input_channel = swap ? 2 - c : c;
            for (int y = 0; y < 8; ++y) {
              const auto padding = y < 2 || y >= 6;
              for (int x = 0; x < 256; ++x) {
                float expected = padding ? param.default_color[channel]
                                         : src.pixels[static_cast<std::size_t>((y - 2) * 784 + x * 3 + input_channel)];
                if (mode == NormalizeType::Full || (mode == NormalizeType::Roi && !padding))
                  expected = (expected - param.mean[channel]) / param.std[channel];
                EXPECT_NEAR(planar_at(dst, c, x, y), expected, 1.0e-5F);
              }
            }
          }
        }
      }
    }
  }
}

TEST(ImagePreprocessorTest, NormalizesInterleavedFloatOutputThroughFallback) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto src = uniform_bgr(8, 4, {10, 20, 30});
  Image dst(MemoryType::Cpu, ImageFormat::InterRgb, DataType::Float32, {8, 4});
  ImagePreprocessorParam param;
  param.dst_image_format = ImageFormat::InterRgb;
  param.mean = {10, 20, 30};
  param.std = {2, 3, 4};
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  ASSERT_TRUE(processor->process(dst, src.image, {}, param));
  const auto* values = static_cast<const float*>(dst.memory()->cpu());
  for (int i = 0; i < 8 * 4; ++i) {
    EXPECT_FLOAT_EQ(values[i * 3], 10.0F);
    EXPECT_FLOAT_EQ(values[i * 3 + 1], 0.0F);
    EXPECT_FLOAT_EQ(values[i * 3 + 2], -5.0F);
  }
}

// Element (c, x, y) of a planar, gray or interleaved uint8 or float32 host
// image; stride[i] steps dim i - 1 and the last dim is packed.
auto value_at(const hrt::core::Image& image, int c, int x, int y) -> float {
  const auto* base = static_cast<const std::byte*>(image.memory()->cpu());
  const auto& stride = image.stride();
  const auto element = image.data_type() == hrt::core::DataType::Float32 ? sizeof(float) : 1;
  const auto planar = image.format() == hrt::core::ImageFormat::PlanarRgb ||
                      image.format() == hrt::core::ImageFormat::PlanarBgr ||
                      image.format() == hrt::core::ImageFormat::Gray;
  const auto offset = planar ? static_cast<std::size_t>(c) * stride[1] + static_cast<std::size_t>(y) * stride[2] +
                                   static_cast<std::size_t>(x) * element
                             : static_cast<std::size_t>(y) * stride[1] + static_cast<std::size_t>(x) * stride[2] +
                                   static_cast<std::size_t>(c) * element;
  if (image.data_type() == hrt::core::DataType::Float32)
    return *reinterpret_cast<const float*>(base + offset);
  return static_cast<float>(*reinterpret_cast<const std::uint8_t*>(base + offset));
}

// Conversions the CUDA kernel does not run go to the CPU operators, with the same result.
TEST(ImagePreprocessorTest, CudaLeavesUnfusedConversionsToTheCpu) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto context = Context::create(DeviceType::Cuda);
  if (!context)
    GTEST_SKIP() << "CUDA unavailable";
  context->set_current();
  auto stream = context->create_stream();
  auto cuda = ImagePreprocessor::create(ImagePreprocessorType::Cuda);
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  ASSERT_TRUE(stream && cuda && cpu);
  auto bgr = uniform_bgr(24, 16, {0, 0, 0});
  for (std::size_t i = 0; i < bgr.pixels.size(); ++i)
    bgr.pixels[i] = static_cast<std::uint8_t>((i * 37) % 251);
  hrt::Vector<std::uint8_t> yuv(24 * 16 * 3 / 2);
  for (std::size_t i = 0; i < yuv.size(); ++i)
    yuv[i] = static_cast<std::uint8_t>((i * 53) % 241);
  const auto nv12 = Image::wrap(MemoryType::Cpu, yuv.data(), ImageFormat::Nv12, {24, 16});
  Image gray(MemoryType::Cpu, ImageFormat::Gray, DataType::Uint8, {24, 16});
  ASSERT_TRUE(nv12 && gray.valid() && cpu->process(gray, bgr.image, {}, {}));

  struct Case {
    const Image* source;
    ImageFormat format;
    InterpolationMode mode;
  };

  for (const auto& item : {Case{&bgr.image, ImageFormat::PlanarRgb, InterpolationMode::Cubic},
                           Case{&bgr.image, ImageFormat::Gray, InterpolationMode::Linear},
                           Case{&*nv12, ImageFormat::PlanarBgr, InterpolationMode::Cubic},
                           Case{&gray, ImageFormat::Gray, InterpolationMode::Nearest}}) {
    for (const bool device_source : {false, true}) {
      SCOPED_TRACE(testing::Message() << "format " << static_cast<int>(item.format) << " mode "
                                      << static_cast<int>(item.mode) << " device " << device_source);
      ImagePreprocessorParam param;
      param.resize_type = ResizeType::LetterBox;
      param.interpolation_mode = item.mode;
      param.dst_image_format = item.format;
      param.mean = {2, 4, 6};
      param.std = {2, 4, 8};
      param.default_color = {114, 20, 250};
      Image device(MemoryType::Cuda, item.source->format(), item.source->data_type(), item.source->size());
      ASSERT_TRUE(Buffer::copy(*device.buffer(), *item.source->buffer(), CopyPolicy::AutoDstStride, stream.get()));
      const auto& source = device_source ? device : *item.source;
      Image expected(MemoryType::Cpu, item.format, DataType::Float32, {16, 16});
      Image actual(MemoryType::Cuda, item.format, DataType::Float32, {16, 16});
      Image host(MemoryType::Cpu, item.format, DataType::Float32, {16, 16});
      ASSERT_TRUE(cpu->process(expected, *item.source, {1, 1, 20, 13}, param));
      ASSERT_TRUE(cuda->process(actual, source, {1, 1, 20, 13}, param, stream.get()));
      ASSERT_TRUE(Buffer::copy(*host.buffer(), *actual.buffer(), CopyPolicy::DirectCopy, stream.get()));
      stream->synchronize();
      EXPECT_EQ(std::memcmp(host.memory()->cpu(), expected.memory()->cpu(), expected.buffer()->bytes()), 0);
    }
  }
}

TEST(ImagePreprocessorTest, CropsTheSourceRect) {
  // Left half blue, right half red; cropping the right half yields pure red.
  auto src = uniform_bgr(8, 4, {255, 0, 0});
  auto* pixels = src.pixels.data();
  for (int y = 0; y < 4; ++y) {
    for (int x = 4; x < 8; ++x) {
      auto* px = pixels + (y * 8 + x) * 3;
      px[0] = 0;
      px[2] = 255;
    }
  }
  hrt::core::Image dst(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::InterRgb, hrt::core::DataType::Uint8,
                       {2, 2});
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  const hrt::preprocess::ImagePreprocessorParam param;

  auto transform = preprocessor->process(dst, src.image, {4, 0, 4, 4}, param);
  ASSERT_TRUE(transform);
  EXPECT_EQ(transform->src_roi, (hrt::core::Rect{4, 0, 4, 4}));
  const auto* out = static_cast<const std::uint8_t*>(dst.memory()->cpu());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(out[i * 3 + 0], 255);
    EXPECT_EQ(out[i * 3 + 2], 0);
  }
}

TEST(ImagePreprocessorTest, ProcessesABatchLikeSingleCalls) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  // Several OpenCV threads let the CPU run the samples in parallel.
  const auto threads = cv::getNumThreads();
  cv::setNumThreads(4);
  hrt::Vector<HostImage> sources;
  sources.push_back(uniform_bgr(16, 8, {10, 20, 30}));
  sources.push_back(uniform_bgr(9, 13, {40, 50, 60}, 32));
  sources.push_back(uniform_bgr(20, 20, {70, 80, 90}));
  sources.push_back(uniform_bgr(7, 3, {100, 110, 120}));
  hrt::Vector<Image> srcs;
  for (const auto& source : sources)
    srcs.push_back(source.image);
  const hrt::Vector<Rect> rects{{}, {1, 2, 6, 9}, {4, 4, 12, 8}, {}};
  ImagePreprocessorParam param;
  param.resize_type = ResizeType::LetterBox;
  param.mean = {1, 2, 3};
  param.std = {2, 4, 8};
  param.default_color = {114, 114, 114};
  auto make = []() -> hrt::Vector<Image> {
    hrt::Vector<Image> images;
    for (int i = 0; i < 4; ++i)
      images.emplace_back(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, Size{12, 10});
    return images;
  };
  auto batched = make();
  auto single = make();
  auto processor = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  for (int run = 0; run < 2; ++run) {
    const auto transforms = processor->process(batched, srcs, rects, param);
    ASSERT_TRUE(transforms);
    ASSERT_EQ(transforms->size(), srcs.size());
    auto reference = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
    for (std::size_t i = 0; i < srcs.size(); ++i) {
      SCOPED_TRACE(i);
      const auto expected = reference->process(single[i], srcs[i], rects[i], param);
      ASSERT_TRUE(expected);
      EXPECT_EQ((*transforms)[i].dst_roi, expected->dst_roi);
      EXPECT_EQ((*transforms)[i].src_roi, expected->src_roi);
      const auto bytes = batched[i].buffer()->bytes();
      EXPECT_EQ(std::memcmp(batched[i].memory()->cpu(), single[i].memory()->cpu(), bytes), 0);
    }
  }

  // Mismatched counts and a failing sample fail the whole batch.
  EXPECT_FALSE(processor->process(std::span(batched).first(3), srcs, {}, param));
  EXPECT_FALSE(processor->process(batched, srcs, std::span(rects).first(2), param));
  EXPECT_FALSE(processor->process(std::span<Image>{}, std::span<const Image>{}, {}, param));
  auto outside = rects;
  outside[2] = {10, 10, 20, 20};
  EXPECT_FALSE(processor->process(batched, srcs, outside, param));
  cv::setNumThreads(threads);
}

TEST(ImagePreprocessorTest, CudaStagesOnlyTheCroppedHostRegion) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto cuda = ImagePreprocessor::create(ImagePreprocessorType::Cuda);
  if (!cuda)
    GTEST_SKIP() << "CUDA preprocessing unavailable";
  const auto pitch = std::size_t{64 * 3 + 8};
  auto src = uniform_bgr(64, 48, {0, 0, 0}, pitch);
  for (std::size_t i = 0; i < src.pixels.size(); ++i)
    src.pixels[i] = static_cast<std::uint8_t>((i * 37) % 251);
  Image device(MemoryType::Cuda, ImageFormat::InterBgr, DataType::Uint8, {64, 48});
  ASSERT_TRUE(Buffer::copy(*device.buffer(), *src.image.buffer(), CopyPolicy::AutoDstStride));
  // Staging only the crop and its margin must not change what resize reads, borders included.
  for (const auto crop : {Rect{10, 6, 30, 20}, Rect{0, 0, 17, 48}, Rect{50, 40, 14, 8}}) {
    for (const auto interpolation : {InterpolationMode::Linear, InterpolationMode::Nearest, InterpolationMode::Cubic}) {
      ImagePreprocessorParam param;
      param.resize_type = ResizeType::LetterBox;
      param.interpolation_mode = interpolation;
      Image from_host(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {32, 32});
      Image from_device(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {32, 32});
      const auto host_transform = cuda->process(from_host, src.image, crop, param);
      const auto device_transform = cuda->process(from_device, device, crop, param);
      ASSERT_TRUE(host_transform);
      ASSERT_TRUE(device_transform);
      EXPECT_EQ(host_transform->src_roi, crop);
      EXPECT_EQ(host_transform->dst_roi, device_transform->dst_roi);
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 32; ++y)
          for (int x = 0; x < 32; ++x)
            ASSERT_EQ(planar_at(from_host, c, x, y), planar_at(from_device, c, x, y));
    }
  }
}

TEST(ImagePreprocessorTest, CudaFusedKernelMatchesCpu) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto context = Context::create(DeviceType::Cuda);
  if (!context)
    GTEST_SKIP() << "CUDA unavailable";
  context->set_current();
  auto stream = context->create_stream();
  auto cuda = ImagePreprocessor::create(ImagePreprocessorType::Cuda);
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  ASSERT_TRUE(stream && cuda && cpu);
  auto src = uniform_bgr(97, 61, {0, 0, 0}, 97 * 3 + 5);
  for (std::size_t i = 0; i < src.pixels.size(); ++i)
    src.pixels[i] = static_cast<std::uint8_t>((i * 7919 + i / 13) % 256);
  const auto rgb =
      Image::wrap(MemoryType::Cpu, src.pixels.data(), ImageFormat::InterRgb, {97, 61}, DataType::Uint8, 97 * 3 + 5);
  // YUV of even size with a padded pitch; the crop below starts at odd coordinates.
  hrt::Vector<std::uint8_t> yuv(104 * 60 * 3 / 2);
  for (std::size_t i = 0; i < yuv.size(); ++i)
    yuv[i] = static_cast<std::uint8_t>((i * 4111 + i / 7) % 256);
  const auto nv12 = Image::wrap(MemoryType::Cpu, yuv.data(), ImageFormat::Nv12, {96, 60}, DataType::Uint8, 104);
  const auto nv21 = Image::wrap(MemoryType::Cpu, yuv.data(), ImageFormat::Nv21, {96, 60}, DataType::Uint8, 104);
  ASSERT_TRUE(rgb && nv12 && nv21);
  std::size_t checked = 0;
  std::size_t exact = 0;
  for (const auto* source : hrt::Array<const Image*, 4>{&src.image, &*rgb, &*nv12, &*nv21}) {
    for (const auto format : {ImageFormat::PlanarRgb, ImageFormat::PlanarBgr, ImageFormat::InterRgb}) {
      for (const auto type : {DataType::Float32, DataType::Uint8}) {
        for (const auto resize : {ResizeType::LetterBox, ResizeType::Normal, ResizeType::KeepRatio}) {
          for (const auto mode : {InterpolationMode::Linear, InterpolationMode::Nearest}) {
            for (const auto normalize : {NormalizeType::Full, NormalizeType::Roi, NormalizeType::None}) {
              // Downscaling and upscaling, from the whole source and from a crop.
              for (const auto size : {Size{40, 32}, Size{160, 128}}) {
                for (const auto crop : {Rect{}, Rect{9, 5, 50, 41}}) {
                  if (type == DataType::Uint8 && normalize != NormalizeType::None)
                    continue;
                  SCOPED_TRACE(testing::Message()
                               << "format " << static_cast<int>(format) << " type " << static_cast<int>(type)
                               << " resize " << static_cast<int>(resize) << " mode " << static_cast<int>(mode)
                               << " normalize " << static_cast<int>(normalize) << " size " << size.width << " crop "
                               << crop.width << " source " << static_cast<int>(source->format()));
                  ImagePreprocessorParam param;
                  param.resize_type = resize;
                  param.interpolation_mode = mode;
                  param.normalize_type = normalize;
                  param.dst_image_format = format;
                  param.mean = {3, 6, 9};
                  param.std = {2, 4, 8};
                  param.default_color = {114.4F, 20, 250};
                  Image expected(MemoryType::Cpu, format, type, size);
                  Image device(MemoryType::Cuda, format, type, size);
                  Image actual(MemoryType::Cpu, format, type, size);
                  ASSERT_TRUE(cpu->process(expected, *source, crop, param));
                  ASSERT_TRUE(cuda->process(device, *source, crop, param, stream.get()));
                  ASSERT_TRUE(Buffer::copy(*actual.buffer(), *device.buffer(), CopyPolicy::DirectCopy, stream.get()));
                  stream->synchronize();
                  for (int c = 0; c < 3; ++c) {
                    // One uint8 step of rounding in the resize, scaled by normalization.
                    const auto step =
                        normalize == NormalizeType::None ? 1.0F : 1.0F / param.std[static_cast<std::size_t>(c)];
                    for (int y = 0; y < size.height; ++y) {
                      for (int x = 0; x < size.width; ++x) {
                        const auto diff = std::abs(value_at(actual, c, x, y) - value_at(expected, c, x, y));
                        ASSERT_LE(diff, step + 1.0e-4F)
                            << "c " << c << " x " << x << " y " << y << " cpu " << value_at(expected, c, x, y)
                            << " cuda " << value_at(actual, c, x, y);
                        ++checked;
                        exact += diff <= 1.0e-4F ? 1 : 0;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // The kernel follows OpenCV's vectorized fixed-point rounding; OpenCV's scalar row tails round slightly differently.
  EXPECT_GT(static_cast<double>(exact), 0.998 * static_cast<double>(checked)) << exact << " of " << checked;
}

TEST(ImagePreprocessorTest, RgaMatchesCpuForAlignedAndUnalignedRowStrides) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto rga = ImagePreprocessor::create(ImagePreprocessorType::Rga);
  if (!rga)
    GTEST_SKIP() << "RGA preprocessing unavailable";
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  // Host images are staged into dma-bufs, whose rows RGA can address at any image width.
  for (const int width : {800, 810}) {
    // Smooth gradients: RGA and OpenCV sample differently, which only matters at sharp edges.
    auto src = uniform_bgr(width, 1080, {0, 0, 0});
    for (int y = 0; y < 1080; ++y) {
      for (int x = 0; x < width; ++x) {
        auto* px = src.pixels.data() +
                   (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3;
        px[0] = static_cast<std::uint8_t>(y * 200 / 1080);
        px[1] = static_cast<std::uint8_t>(x * 200 / width);
        px[2] = static_cast<std::uint8_t>(250 - y * 200 / 1080);
      }
    }
    ImagePreprocessorParam param;
    param.resize_type = ResizeType::LetterBox;
    param.dst_image_format = ImageFormat::InterRgb;
    param.default_color = {114, 114, 114};
    Image expected(MemoryType::Cpu, ImageFormat::InterRgb, DataType::Uint8, {128, 128});
    Image actual(MemoryType::DmaBuffer, ImageFormat::InterRgb, DataType::Uint8, {128, 128});
    ASSERT_TRUE(actual.valid());
    const auto reference = cpu->process(expected, src.image, {}, param);
    const auto result = rga->process(actual, src.image, {}, param);
    ASSERT_TRUE(reference);
    ASSERT_TRUE(result) << "width " << width;
    EXPECT_EQ(result->dst_roi, reference->dst_roi);
    const auto* want = static_cast<const std::uint8_t*>(expected.memory()->cpu());
    const auto* got = static_cast<const std::uint8_t*>(actual.memory()->cpu());
    for (std::size_t i = 0; i < 128 * 128 * 3; ++i)
      ASSERT_NEAR(got[i], want[i], 2) << "width " << width << " byte " << i;
  }
}

TEST(ImagePreprocessorTest, RgaMatchesCpuForCroppedSources) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto rga = ImagePreprocessor::create(ImagePreprocessorType::Rga);
  if (!rga)
    GTEST_SKIP() << "RGA preprocessing unavailable";
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  constexpr int kWidth = 1270;
  constexpr int kHeight = 717;
  constexpr std::size_t kPitch = static_cast<std::size_t>(kWidth) * 3;
  auto host = uniform_bgr(kWidth, kHeight, {0, 0, 0});
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      auto* px = host.pixels.data() + static_cast<std::size_t>(y) * kPitch + static_cast<std::size_t>(x) * 3;
      px[0] = static_cast<std::uint8_t>(x * 200 / kWidth);
      px[1] = static_cast<std::uint8_t>(y * 200 / kHeight);
      px[2] = static_cast<std::uint8_t>(30 + (x + y) * 200 / (kWidth + kHeight));
    }
  }
  // The same pixels in a dma-buf (RGA reads the crop in place), and in one with packed, unaligned rows (RGA
  // cannot address them, so resize falls back to the CPU).
  Image device(MemoryType::DmaBuffer, ImageFormat::InterBgr, DataType::Uint8, {kWidth, kHeight});
  ASSERT_TRUE(Buffer::copy(*device.buffer(), *host.image.buffer(), CopyPolicy::AutoDstStride));
  Image packed(MemoryType::DmaBuffer, ImageFormat::InterBgr, DataType::Uint8, {kWidth, kHeight},
               {kPitch * static_cast<std::size_t>(kHeight), kPitch, 3});
  ASSERT_TRUE(packed.valid());
  ASSERT_TRUE(Buffer::copy(*packed.buffer(), *host.image.buffer(), CopyPolicy::DirectCopy));
  // A caller's frame wrapped by fd, as from a decoder: RGA reads it without staging.
  const auto fd = device.memory()->mutable_device();
  auto frame = Image::wrap(MemoryType::DmaBuffer, reinterpret_cast<void*>(fd), ImageFormat::InterBgr, {kWidth, kHeight},
                           DataType::Uint8, device.stride()[1]);
  ASSERT_TRUE(frame);
  const hrt::Array<const Image*, 4> sources{&host.image, &device, &packed, &*frame};
  // Interior, offset 1 (RGA shifts it to 0), right and bottom edges, a 7x9 box upscaled past RGA's 16x,
  // a box thinner than the input aspect, and the whole image.
  const Rect crops[] = {{301, 157, 333, 219}, {1, 1, 200, 100},   {1001, 517, 269, 200},
                        {640, 300, 7, 9},     {100, 40, 50, 600}, {0, 0, kWidth, kHeight}};
  for (std::size_t s = 0; s < sources.size(); ++s) {
    const auto* source = sources[s];
    for (const auto& crop : crops) {
      for (const auto resize :
           {ResizeType::Normal, ResizeType::LetterBox, ResizeType::PadOutside, ResizeType::TextLine}) {
        for (const auto format : {ImageFormat::InterRgb, ImageFormat::PlanarRgb}) {
          SCOPED_TRACE(testing::Message() << "source " << s << " crop " << crop.x << ',' << crop.y << ' ' << crop.width
                                          << 'x' << crop.height << " resize " << static_cast<int>(resize) << " format "
                                          << static_cast<int>(format));
          ImagePreprocessorParam param;
          param.resize_type = resize;
          param.dst_image_format = format;
          param.default_color = {114, 114, 114};
          param.pad_scale = 1.2F;
          const auto type = format == ImageFormat::InterRgb ? DataType::Uint8 : DataType::Float32;
          Image expected(MemoryType::Cpu, format, type, {128, 96});
          Image actual(MemoryType::DmaBuffer, format, type, {128, 96});
          const auto reference = cpu->process(expected, host.image, crop, param);
          const auto result = rga->process(actual, *source, crop, param);
          ASSERT_TRUE(reference);
          ASSERT_TRUE(result);
          EXPECT_EQ(result->dst_roi, reference->dst_roi);
          const auto row = [](const Image& image, int y) -> const std::byte* {
            return static_cast<const std::byte*>(image.memory()->cpu()) +
                   static_cast<std::size_t>(y) * image.stride()[image.stride().size() - 2];
          };
          float worst = 0.0F;
          for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 96; ++y) {
              for (int x = 0; x < 128; ++x) {
                float want = 0.0F;
                float got = 0.0F;
                if (type == DataType::Uint8) {
                  const auto offset = static_cast<std::size_t>(x * 3 + c);
                  want = static_cast<float>(reinterpret_cast<const std::uint8_t*>(row(expected, y))[offset]);
                  got = static_cast<float>(reinterpret_cast<const std::uint8_t*>(row(actual, y))[offset]);
                } else {
                  want = planar_at(expected, c, x, y);
                  got = planar_at(actual, c, x, y);
                }
                worst = std::max(worst, std::abs(got - want));
              }
            }
          }
          // RGA and OpenCV bilinear sampling differ by a level or two on these gradients.
          EXPECT_LE(worst, 3.0F);
        }
      }
    }
  }
}

TEST(ImagePreprocessorTest, ConvertsNv12Input) {
  // Mid-gray NV12: Y = U = V = 128.
  hrt::Vector<std::uint8_t> nv12(4 * 4 * 3 / 2, 128);
  auto src = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, nv12.data(), hrt::core::ImageFormat::Nv12, {4, 4});
  ASSERT_TRUE(src);
  hrt::core::Image dst(hrt::core::MemoryType::Cpu, hrt::core::ImageFormat::PlanarRgb, hrt::core::DataType::Float32,
                       {4, 4});
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu);
  const hrt::preprocess::ImagePreprocessorParam param;

  ASSERT_TRUE(preprocessor->process(dst, *src, {}, param));
  EXPECT_NEAR(planar_at(dst, 0, 1, 1), 128.0F, 2.0F);
  EXPECT_NEAR(planar_at(dst, 2, 3, 3), 128.0F, 2.0F);
}

// DVPP only runs the fused crop, resize and border step, on uint8 images whose rows VPC can address.
TEST(ImagePreprocessorTest, DvppMatchesCpuForCropsAndResizes) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto context = Context::create(DeviceType::Ascend);
  if (context == nullptr)
    GTEST_SKIP() << "Ascend device unavailable";
  context->set_current();
  auto dvpp = ImagePreprocessor::create(ImagePreprocessorType::Dvpp);
  if (!dvpp)
    GTEST_SKIP() << "DVPP unavailable";
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  auto stream = context->create_stream();
  constexpr int kWidth = 64;
  constexpr int kHeight = 48;
  auto src = uniform_bgr(kWidth, kHeight, {0, 0, 0});
  for (std::size_t i = 0; i < src.pixels.size(); ++i)
    src.pixels[i] = static_cast<std::uint8_t>((i * 37) % 251);
  Image device(MemoryType::Dvpp, ImageFormat::InterBgr, DataType::Uint8, {kWidth, kHeight});
  ASSERT_TRUE(device.valid());
  ASSERT_TRUE(Buffer::copy(*device.buffer(), *src.image.buffer(), CopyPolicy::AutoDstStride, stream.get()));
  stream->synchronize();
  Image probe(MemoryType::Dvpp, ImageFormat::InterRgb, DataType::Uint8, {32, 24});
  ASSERT_TRUE(probe.valid());
  if (!dvpp->supports(ImageOperation::Fused, probe, device))
    GTEST_SKIP() << "VPC cannot run on this device";

  // 1:1 crops: VPC copies the pixels, so every byte must equal the CPU result. Scaled cases only
  // check that the geometry and channel order agree, as VPC resamples differently from OpenCV.
  for (const auto size : {Size{kWidth, kHeight}, Size{32, 24}}) {
    for (const auto crop : {Rect{}, Rect{8, 6, 32, 24}, Rect{0, 0, 16, 48}}) {
      SCOPED_TRACE(testing::Message() << "size " << size.width << 'x' << size.height << " crop " << crop.x << ','
                                      << crop.y << ' ' << crop.width << 'x' << crop.height);
      ImagePreprocessorParam param;
      param.dst_image_format = ImageFormat::InterRgb;
      param.normalize_type = NormalizeType::None;
      param.default_color = {114, 114, 114};
      Image expected(MemoryType::Cpu, ImageFormat::InterRgb, DataType::Uint8, size);
      Image actual(MemoryType::Cpu, ImageFormat::InterRgb, DataType::Uint8, size);
      const auto reference = cpu->process(expected, src.image, crop, param);
      const auto result = dvpp->process(actual, device, crop, param, stream.get());
      ASSERT_TRUE(reference);
      ASSERT_TRUE(result);
      stream->synchronize();
      EXPECT_EQ(result->src_roi, reference->src_roi);
      EXPECT_EQ(result->dst_roi, reference->dst_roi);
      const auto* want = static_cast<const std::uint8_t*>(expected.memory()->cpu());
      const auto* got = static_cast<const std::uint8_t*>(actual.memory()->cpu());
      ASSERT_NE(want, nullptr);
      ASSERT_NE(got, nullptr);
      const auto scaled =
          result->dst_roi.width != result->src_roi.width || result->dst_roi.height != result->src_roi.height;
      double error = 0;
      for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width * 3; ++x) {
          const auto a = want[static_cast<std::size_t>(y) * expected.stride()[1] + static_cast<std::size_t>(x)];
          const auto b = got[static_cast<std::size_t>(y) * actual.stride()[1] + static_cast<std::size_t>(x)];
          if (!scaled) {
            ASSERT_EQ(a, b);
          }
          error += std::abs(static_cast<double>(a) - static_cast<double>(b));
        }
      }
      error /= static_cast<double>(size.width) * size.height * 3;
      EXPECT_LT(error, 24.0);
    }
  }
}

#ifdef __APPLE__
TEST(ImagePreprocessorTest, MetalFallsBackForNv12Input) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto metal = ImagePreprocessor::create(ImagePreprocessorType::Metal);
  if (!metal)
    GTEST_SKIP() << "Metal GPU unavailable";
  hrt::Array<std::uint8_t, 24> pixels{};
  std::fill_n(pixels.begin(), 16, std::uint8_t{80});
  std::fill(pixels.begin() + 16, pixels.end(), std::uint8_t{128});
  auto source = Image::wrap(MemoryType::Cpu, pixels.data(), ImageFormat::Nv12, {4, 4});
  ASSERT_TRUE(source);
  Image expected(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {4, 4});
  Image actual(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {4, 4});
  ImagePreprocessorParam param;
  param.std = {255, 255, 255};
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  ASSERT_TRUE(cpu->process(expected, *source, {}, param));
  ASSERT_TRUE(metal->process(actual, *source, {}, param));
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        EXPECT_FLOAT_EQ(planar_at(actual, c, x, y), planar_at(expected, c, x, y));
}

TEST(ImagePreprocessorTest, MetalMatchesCpuAcrossGeometryStridesAndFrameChanges) {
  using namespace hrt::core;
  using namespace hrt::preprocess;
  auto metal = ImagePreprocessor::create(ImagePreprocessorType::Metal);
  if (!metal)
    GTEST_SKIP() << "Metal GPU unavailable";
  auto cpu = ImagePreprocessor::create(ImagePreprocessorType::Cpu);
  auto invalid_source = uniform_bgr(8, 8, {10, 20, 30});
  Image invalid_target(MemoryType::Cpu, ImageFormat::PlanarRgb, DataType::Float32, {8, 8});
  EXPECT_FALSE(metal->process(invalid_target, invalid_source.image, {-1, 0, 8, 8}, ImagePreprocessorParam{}));
  for (const auto size : {Size{13, 7}, Size{31, 17}, Size{9, 15}, Size{13, 7}}) {
    const auto pitch = static_cast<std::size_t>(size.width * 3 + 16);
    auto src = uniform_bgr(size.width, size.height, {0, 0, 0}, pitch);
    for (int y = 0; y < size.height; ++y)
      for (int x = 0; x < size.width; ++x)
        for (int c = 0; c < 3; ++c)
          src.pixels[static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x * 3 + c)] =
              static_cast<std::uint8_t>((x * 17 + y * 13 + c * 31) % 256);
    const auto source_format = size.height == 15 ? ImageFormat::InterRgb : ImageFormat::InterBgr;
    auto source = Image::wrap(MemoryType::Cpu, src.pixels.data(), source_format, size, DataType::Uint8, pitch);
    ASSERT_TRUE(source);
    const int side = size.width == 31 ? 16 : 8;
    const auto row = static_cast<std::size_t>(side + 4) * sizeof(float);
    const auto plane = row * static_cast<std::size_t>(side);
    for (const auto format : {ImageFormat::PlanarRgb, ImageFormat::PlanarBgr}) {
      Image expected(MemoryType::Cpu, format, DataType::Float32, {side, side});
      Image actual(MemoryType::Cpu, format, DataType::Float32, {side, side}, {plane * 3, plane, row});
      for (const auto resize : {ResizeType::Normal, ResizeType::LetterBox, ResizeType::KeepRatio,
                                ResizeType::CenterCrop, ResizeType::PadOutside, ResizeType::TextLine}) {
        for (const auto interpolation :
             {InterpolationMode::Linear, InterpolationMode::Nearest, InterpolationMode::Cubic}) {
          for (const auto normalization : {NormalizeType::Full, NormalizeType::Roi, NormalizeType::None}) {
            ImagePreprocessorParam param;
            param.dst_image_format = format;
            param.resize_type = resize;
            param.resize_short_size = side;
            param.interpolation_mode = interpolation;
            param.normalize_type = normalization;
            param.mean = {1, 2, 3};
            param.std = {255, 127, 63};
            param.default_color = {11, 22, 33};
            const Rect crop{1, 1, size.width - 2, size.height - 2};
            const auto reference = cpu->process(expected, *source, crop, param);
            const auto result = metal->process(actual, *source, crop, param);
            ASSERT_TRUE(reference);
            ASSERT_TRUE(result);
            EXPECT_EQ(result->src_roi, reference->src_roi);
            EXPECT_EQ(result->dst_roi, reference->dst_roi);
            for (int c = 0; c < 3; ++c)
              for (int y = 0; y < side; ++y)
                for (int x = 0; x < side; ++x) {
                  const auto& roi = reference->dst_roi;
                  const bool inside = x >= roi.x && x < roi.x + roi.width && y >= roi.y && y < roi.y + roi.height;
                  const bool normalized =
                      normalization == NormalizeType::Full || (normalization == NormalizeType::Roi && inside);
                  const auto tolerance = interpolation == InterpolationMode::Linear
                                             ? 1.01F / (normalized ? param.std[static_cast<std::size_t>(c)] : 1.0F)
                                             : 1.0e-6F;
                  EXPECT_NEAR(planar_at(actual, c, x, y), planar_at(expected, c, x, y), tolerance);
                }
          }
        }
      }
    }
  }
}
#endif

TEST(ImagePreprocessorTest, CreatesByDevice) {
  EXPECT_EQ(hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Cpu)->type(),
            hrt::preprocess::ImagePreprocessorType::Cpu);
  // Host target memory picks the CPU.
  auto preprocessor = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Auto,
                                                                 hrt::core::MemoryType::Cpu);
  ASSERT_NE(preprocessor, nullptr);
  EXPECT_EQ(preprocessor->type(), hrt::preprocess::ImagePreprocessorType::Cpu);
  // Auto picks the device of the target memory when this build has it.
  const auto dvpp = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Dvpp);
  const auto for_dvpp = hrt::preprocess::ImagePreprocessor::create(hrt::preprocess::ImagePreprocessorType::Auto,
                                                                   hrt::core::MemoryType::Dvpp);
  ASSERT_NE(for_dvpp, nullptr);
  EXPECT_EQ(for_dvpp->type(), dvpp != nullptr ? hrt::preprocess::ImagePreprocessorType::Dvpp
                                              : hrt::preprocess::ImagePreprocessorType::Cpu);
}

TEST(ImagePreprocessorTest, ParsesDevice) {
  EXPECT_EQ(kitzoo::json::Json::parse(R"({"device":"metal"})").get<hrt::preprocess::ImagePreprocessorParam>().device,
            hrt::preprocess::ImagePreprocessorType::Metal);
  auto param = kitzoo::json::Json::parse(R"({"device": "cuda"})").get<hrt::preprocess::ImagePreprocessorParam>();
  EXPECT_EQ(param.device, hrt::preprocess::ImagePreprocessorType::Cuda);
  EXPECT_EQ(kitzoo::json::Json::parse(R"({"device": "auto"})").get<hrt::preprocess::ImagePreprocessorParam>().device,
            hrt::preprocess::ImagePreprocessorType::Auto);
}

} // namespace
