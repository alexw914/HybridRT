// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/engine/openvino_engine_test.cpp
// Description: Tests OpenVINO CPU inference on IR models.
// -----------------------------------------------------------------------------

#include <hrt/engine/engine.hpp>
#include <hrt/model/image_model.hpp>

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <utility>

namespace {

auto identity_ir_param(std::string_view config = "") -> hrt::hrtc::EngineParam {
  constexpr std::string_view xml = R"xml(<?xml version="1.0"?>
<net name="relu" version="11"><layers>
  <layer id="0" name="x" type="Parameter" version="opset1"><data shape="1,4" element_type="f32"/><output><port id="0" precision="FP32" names="x"><dim>1</dim><dim>4</dim></port></output></layer>
  <layer id="1" name="relu" type="ReLU" version="opset1"><input><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port></input><output><port id="1" precision="FP32" names="y"><dim>1</dim><dim>4</dim></port></output></layer>
  <layer id="2" name="y" type="Result" version="opset1"><input><port id="0" precision="FP32" names="y"><dim>1</dim><dim>4</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/><edge from-layer="1" from-port="1" to-layer="2" to-port="0"/></edges></net>)xml";
  return {.engine_type = "openvino",
          .config = hrt::String(config),
          .proto = std::make_shared<const hrt::hrtc::Bytes>(xml.begin(), xml.end()),
          .data = {}};
}

TEST(OpenvinoEngineTest, RunsReluIrModel) {
  auto engine = hrt::engine::Engine::create("OpenVINO");
  ASSERT_NE(engine, nullptr);
  ASSERT_TRUE(engine->initialize(identity_ir_param(R"({"num_threads": 2})")));
  EXPECT_EQ(engine->input_names(), hrt::Vector<hrt::String>{"x"});
  EXPECT_EQ(engine->output_names(), hrt::Vector<hrt::String>{"y"});
  ASSERT_EQ(engine->input_protos().size(), 1U);
  ASSERT_EQ(engine->output_protos().size(), 1U);
  EXPECT_EQ(engine->input_protos()[0].shape, (hrt::core::Shape{1, 4}));
  EXPECT_EQ(engine->output_protos()[0].data_type, hrt::core::DataType::Float32);

  auto input = std::make_shared<hrt::core::Buffer>(engine->input_protos()[0]);
  auto output = std::make_shared<hrt::core::Buffer>(engine->output_protos()[0]);
  ASSERT_TRUE(engine->set_inputs({input}));
  ASSERT_TRUE(engine->set_outputs({output}));
  const hrt::Array<float, 4> values{-2.0F, -0.5F, 0.0F, 3.0F};
  std::copy(values.begin(), values.end(), static_cast<float*>(input->memory()->mutable_cpu()));
  ASSERT_TRUE(engine->inference());
  const auto* result = static_cast<const float*>(output->memory()->cpu());
  EXPECT_FLOAT_EQ(result[0], 0.0F);
  EXPECT_FLOAT_EQ(result[1], 0.0F);
  EXPECT_FLOAT_EQ(result[2], 0.0F);
  EXPECT_FLOAT_EQ(result[3], 3.0F);

  ASSERT_TRUE(engine->initialize(identity_ir_param()));
  ASSERT_TRUE(engine->set_inputs({input}));
  ASSERT_TRUE(engine->set_outputs({output}));
  ASSERT_TRUE(engine->inference());
  EXPECT_FLOAT_EQ(result[3], 3.0F);
}

TEST(OpenvinoEngineTest, RunsIrXmlWithBinWeights) {
  constexpr std::string_view xml = R"xml(<?xml version="1.0"?>
<net name="add_weights" version="11">
  <layers>
    <layer id="0" name="x" type="Parameter" version="opset1">
      <data shape="1,4" element_type="f32"/><output><port id="0" precision="FP32" names="x"><dim>1</dim><dim>4</dim></port></output>
    </layer>
    <layer id="1" name="weights" type="Const" version="opset1">
      <data element_type="f32" shape="1,4" offset="0" size="16"/><output><port id="0" precision="FP32" names="weights"><dim>1</dim><dim>4</dim></port></output>
    </layer>
    <layer id="2" name="sum" type="Add" version="opset1">
      <input><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port><port id="1" precision="FP32"><dim>1</dim><dim>4</dim></port></input>
      <output><port id="2" precision="FP32" names="y"><dim>1</dim><dim>4</dim></port></output>
    </layer>
    <layer id="3" name="y" type="Result" version="opset1">
      <input><port id="0" precision="FP32" names="y"><dim>1</dim><dim>4</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
  </edges>
</net>)xml";
  const hrt::Array<float, 4> weights{1.0F, 2.0F, 3.0F, 4.0F};
  hrt::hrtc::Bytes bin(sizeof(weights));
  std::memcpy(bin.data(), weights.data(), sizeof(weights));
  const hrt::hrtc::EngineParam param{.engine_type = "openvino",
                                     .config = {},
                                     .proto = std::make_shared<const hrt::hrtc::Bytes>(xml.begin(), xml.end()),
                                     .data = std::make_shared<const hrt::hrtc::Bytes>(std::move(bin))};
  auto engine = hrt::engine::Engine::create("openvino");
  ASSERT_NE(engine, nullptr);
  ASSERT_TRUE(engine->initialize(param));
  auto input = std::make_shared<hrt::core::Buffer>(engine->input_protos()[0]);
  auto output = std::make_shared<hrt::core::Buffer>(engine->output_protos()[0]);
  const hrt::Array<float, 4> values{10.0F, 20.0F, 30.0F, 40.0F};
  std::copy(values.begin(), values.end(), static_cast<float*>(input->memory()->mutable_cpu()));
  ASSERT_TRUE(engine->set_inputs({input}));
  ASSERT_TRUE(engine->set_outputs({output}));
  ASSERT_TRUE(engine->inference());
  const auto* result = static_cast<const float*>(output->memory()->cpu());
  for (std::size_t i = 0; i < weights.size(); ++i)
    EXPECT_FLOAT_EQ(result[i], values[i] + weights[i]);
}

TEST(OpenvinoEngineTest, RejectsNonIrModelAndBuffer) {
  auto engine = hrt::engine::Engine::create(hrt::engine::EngineType::OpenVino);
  ASSERT_NE(engine, nullptr);
  const hrt::hrtc::EngineParam bad{.engine_type = "openvino",
                                   .config = {},
                                   .proto = {},
                                   .data = std::make_shared<const hrt::hrtc::Bytes>(hrt::hrtc::Bytes{1, 2, 3})};
  EXPECT_FALSE(engine->initialize(bad));
  ASSERT_TRUE(engine->initialize(identity_ir_param()));
  auto wrong = std::make_shared<hrt::core::Buffer>(hrt::core::MemoryType::Cpu, hrt::core::DataType::Float32,
                                                   hrt::core::Shape{1, 3});
  EXPECT_FALSE(engine->set_inputs({wrong}));
}

TEST(OpenvinoEngineTest, ChecksConfig) {
  auto engine = hrt::engine::Engine::create("openvino");
  ASSERT_NE(engine, nullptr);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(engine->initialize(identity_ir_param(R"({"unknown": 1})")), "");
  EXPECT_DEATH(engine->initialize(identity_ir_param(R"({"num_threads": -1})")), "");
}

TEST(OpenvinoEngineTest, RunsImageModelPipeline) {
  constexpr std::string_view xml = R"xml(<?xml version="1.0"?>
<net name="identity_image" version="11"><layers>
  <layer id="0" name="x" type="Parameter" version="opset1"><data shape="1,3,2,2" element_type="f32"/><output><port id="0" precision="FP32" names="x"><dim>1</dim><dim>3</dim><dim>2</dim><dim>2</dim></port></output></layer>
  <layer id="1" name="relu" type="ReLU" version="opset1"><input><port id="0" precision="FP32"><dim>1</dim><dim>3</dim><dim>2</dim><dim>2</dim></port></input><output><port id="1" precision="FP32" names="y"><dim>1</dim><dim>3</dim><dim>2</dim><dim>2</dim></port></output></layer>
  <layer id="2" name="y" type="Result" version="opset1"><input><port id="0" precision="FP32" names="y"><dim>1</dim><dim>3</dim><dim>2</dim><dim>2</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/><edge from-layer="1" from-port="1" to-layer="2" to-port="0"/></edges></net>)xml";
  const auto config =
      kitzoo::json::Json{{"preprocess", {{"dst_image_format", "planar_rgb"}, {"std", {255, 255, 255}}}},
                         {"postprocess", {{"type", "GeneralClassify"}, {"classify_postprocess_type", "none"}}}};
  const hrt::hrtc::AlgoParam algo{
      .name = "openvino_classify",
      .engines = {{.engine_type = "openvino",
                   .config = {},
                   .proto = std::make_shared<const hrt::hrtc::Bytes>(xml.begin(), xml.end()),
                   .data = {}}},
      .config = config.dump()};
  hrt::SharedPtr<hrt::core::Context> context = hrt::core::Context::create(hrt::core::DeviceType::Cpu);
  auto model = hrt::model::ImageModel<hrt::postprocess::ClassifyResult>::create(algo, context);
  ASSERT_NE(model, nullptr);

  hrt::Array<std::uint8_t, 12> pixels{};
  for (std::size_t i = 0; i < pixels.size(); i += 3) {
    pixels[i] = 10;
    pixels[i + 1] = 20;
    pixels[i + 2] = 30;
  }
  auto image =
      hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, pixels.data(), hrt::core::ImageFormat::InterBgr, {2, 2});
  ASSERT_TRUE(image);
  const auto result = model->process_sync(*image);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1U);
  ASSERT_EQ(result->front().size(), 12U);
  EXPECT_FLOAT_EQ((*result)[0][0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][4], 20.0F / 255.0F);
  EXPECT_FLOAT_EQ((*result)[0][11], 10.0F / 255.0F);
}

} // namespace
