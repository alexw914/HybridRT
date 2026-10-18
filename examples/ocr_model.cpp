// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: examples/ocr_model.cpp
// Description: Runs a two-algo OCR model from JSON or .hrtc and writes JSON text results.
// -----------------------------------------------------------------------------

#include <hrt/model/ocr_model.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>

#include <opencv2/imgcodecs.hpp>

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hybridrt_example_ocr_model", "Run sequential OCR from a manifest or package");
  auto add = options.add_options();
  add("m,model", ".json or .hrtc", kitzoo::core::value<hrt::String>());
  add("i,image", "Input image", kitzoo::core::value<hrt::String>());
  add("d,detection", "Detection algo", kitzoo::core::value<hrt::String>()->default_value("ppocrv6_tiny_det"));
  add("r,recognition", "Recognition algo", kitzoo::core::value<hrt::String>()->default_value("ppocrv6_tiny_rec"));
  add("p,provider", "Optional ORT override: cpu or coreml", kitzoo::core::value<hrt::String>());
  add("o,output", "JSON results", kitzoo::core::value<hrt::String>()->default_value("ocr.json"));
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  if (args.count("help") != 0 || args.count("model") == 0 || args.count("image") == 0) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0 ? 0 : 1;
  }
  const auto detection = args["detection"].as<hrt::String>();
  const auto recognition = args["recognition"].as<hrt::String>();
  const std::filesystem::path path = args["model"].as<hrt::String>();
  std::optional<hrt::hrtc::Package> package;
  if (path.extension() == ".json") {
    const kitzoo::json::Reader reader(path);
    const auto& json = reader.raw();
    if (!reader.is_parse_success() || !json.is_object() || !json.contains("algos") || !json["algos"].is_array())
      return 1;
    package.emplace();
    for (const auto& entry : json["algos"]) {
      if (!entry.is_object() || !entry.contains("name") || !entry["name"].is_string() ||
          (entry["name"] != detection && entry["name"] != recognition))
        continue;
      auto algo = hrt::hrtc::load_algo(entry);
      if (!algo)
        return 1;
      package->algos.push_back(std::move(*algo));
    }
  } else {
    package = hrt::hrtc::load(path);
  }
  if (!package || package->find(detection) == nullptr || package->find(recognition) == nullptr) {
    KZ_LOG_ERROR("OCR detection or recognition algo is missing");
    return 1;
  }
  if (args.count("provider") != 0) {
    const auto provider = args["provider"].as<hrt::String>();
    if (provider != "cpu" && provider != "coreml") {
      KZ_LOG_ERROR("provider must be cpu or coreml");
      return 1;
    }
    for (auto& algo : package->algos) {
      for (auto& backend : algo.engines) {
        if (backend.engine_type != "ort")
          continue;
        const kitzoo::json::Reader reader(backend.config.empty() ? "{}" : backend.config);
        auto config = reader.raw();
        if (!reader.is_parse_success() || !config.is_object())
          return 1;
        config["execution_provider"] = provider;
        config["provider_options"] =
            provider == "coreml" ? kitzoo::json::Json{{"ModelFormat", "MLProgram"}, {"RequireStaticInputShapes", "1"}}
                                 : kitzoo::json::Json::object();
        backend.config = config.dump();
      }
    }
  }
  const auto bgr = cv::imread(args["image"].as<hrt::String>());
  if (bgr.empty()) {
    KZ_LOG_ERROR("cannot read input image");
    return 1;
  }
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  if (!image)
    return 1;
  const auto model = hrt::model::OcrModel::create(*package, detection, recognition);
  const auto result = model->process_sync(*image);
  if (!result)
    return 1;
  auto output = kitzoo::json::Json::array();
  for (const auto& text : *result) {
    output.push_back({{"rect", {text.rect.x, text.rect.y, text.rect.width, text.rect.height}},
                      {"text", text.text},
                      {"detection_score", text.detection_score},
                      {"recognition_score", text.recognition_score}});
  }
  std::error_code error;
  kitzoo::os::write_file(args["output"].as<hrt::String>(), output.dump(2), error);
  if (error) {
    KZ_LOG_ERROR("cannot write OCR results: {}", error.message());
    return 1;
  }
  KZ_LOG_INFO("OCR: {} text regions", result->size());
  return 0;
}
