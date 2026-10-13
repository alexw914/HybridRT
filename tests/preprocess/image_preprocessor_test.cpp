// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/preprocess/image_preprocessor_test.cpp
// Description: Tests preprocessing parameters and resize geometry.
// -----------------------------------------------------------------------------

#include <hrt/preprocess/image_preprocessor.hpp>

#include <cstdint>
#include <gtest/gtest.h>

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
                float expected = padding ? param.default_color[static_cast<std::size_t>(input_channel)]
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
  auto param = kitzoo::json::Json::parse(R"({"device": "cuda"})").get<hrt::preprocess::ImagePreprocessorParam>();
  EXPECT_EQ(param.device, hrt::preprocess::ImagePreprocessorType::Cuda);
  EXPECT_EQ(kitzoo::json::Json::parse(R"({"device": "auto"})").get<hrt::preprocess::ImagePreprocessorParam>().device,
            hrt::preprocess::ImagePreprocessorType::Auto);
}

} // namespace
