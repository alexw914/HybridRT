// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tools/hrt_ocr.cpp
// Description: Runs OCR (a detection and a recognition algo of an .hrtc package
//              or a JSON manifest) on every image of a directory and writes the
//              texts as JSON (assets/template/ocr_results.json).
// -----------------------------------------------------------------------------

#include <hrt/model/ocr_model.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>
#include <kitzoo/time/time.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/imgcodecs.hpp>

namespace {

// Keeps the field order of assets/template/ocr_results.json.
using Json = nlohmann::ordered_json;

// 4 decimals keep the file diffable; floats would otherwise print as 0.9100000262.
auto round4(float value) -> double {
  return std::round(static_cast<double>(value) * 1e4) / 1e4;
}

auto to_json(const hrt::postprocess::OcrResult& result) -> Json {
  Json results = Json::array();
  for (const auto& text : result) {
    results.push_back({{"text", text.text},
                       {"x", round4(text.rect.x)},
                       {"y", round4(text.rect.y)},
                       {"width", round4(text.rect.width)},
                       {"height", round4(text.rect.height)},
                       {"detection_score", round4(text.detection_score)},
                       {"recognition_score", round4(text.recognition_score)}});
  }
  return results;
}

// One image entry; failures are recorded in "error".
auto run(hrt::model::OcrModel& model, const std::filesystem::path& path) -> Json {
  Json entry = {
      {"file", path.filename().string()}, {"width", 0}, {"height", 0}, {"error", nullptr}, {"results", Json::array()}};
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  const auto image =
      bgr.empty() ? std::nullopt
                  : hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                           {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
  if (!image) {
    entry["error"] = "cannot read the image";
    return entry;
  }
  const auto result = model.process_sync(*image);
  if (!result) {
    entry["error"] = "inference failed";
    return entry;
  }
  entry["width"] = bgr.cols;
  entry["height"] = bgr.rows;
  entry["results"] = to_json(*result);
  return entry;
}

} // namespace

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hrt-ocr", "Run OCR of an .hrtc package or manifest on an image directory");
  auto add = options.add_options();
  add("m,model", ".hrtc package or .json manifest", kitzoo::core::value<hrt::String>());
  add("d,detection", "Detection algo", kitzoo::core::value<hrt::String>()->default_value("ppocrv6_tiny_det"));
  add("r,recognition", "Recognition algo", kitzoo::core::value<hrt::String>()->default_value("ppocrv6_tiny_rec"));
  add("i,input", "Input image directory", kitzoo::core::value<hrt::String>());
  add("e,engine", "Engine type, e.g. ort (default: the first usable one)", kitzoo::core::value<hrt::String>());
  add("o,output", "Output JSON file (default: <input>_ocr.json beside the input directory)",
      kitzoo::core::value<hrt::String>());
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  if (args.count("help") != 0U || args.count("model") == 0U || args.count("input") == 0U) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0U ? 0 : 1;
  }

  const std::filesystem::path model = args["model"].as<hrt::String>();
  const std::filesystem::path input = args["input"].as<hrt::String>();
  const auto package = model.extension() == ".json" ? hrt::hrtc::load_json(model) : hrt::hrtc::load(model);
  if (!package)
    return 1;
  hrt::Vector<hrt::hrtc::AlgoParam> algos;
  for (const auto& name : {args["detection"].as<hrt::String>(), args["recognition"].as<hrt::String>()}) {
    const auto* found = package->find(name);
    if (found == nullptr) {
      KZ_LOG_ERROR("no algo {} in {}", name, model.string());
      return 1;
    }
    algos.push_back(*found);
  }
  Json engine = nullptr;
  if (args.count("engine") != 0U) {
    const auto type = hrt::engine::engine_type_from(args["engine"].as<hrt::String>());
    for (auto& algo : algos) {
      std::erase_if(algo.engines,
                    [&](const auto& param) { return hrt::engine::engine_type_from(param.engine_type) != type; });
      if (algo.engines.empty()) {
        KZ_LOG_ERROR("algo {} has no {} engine", algo.name, args["engine"].as<hrt::String>());
        return 1;
      }
    }
    engine = algos.front().engines.front().engine_type;
  }
  const auto problems = hrt::model::check_ocr_config(algos[0], algos[1]);
  for (const auto& problem : problems)
    KZ_LOG_ERROR("{}", problem);
  if (!problems.empty())
    return 1;
  std::error_code ec;
  auto paths = kitzoo::os::list_directory(input, ec);
  if (ec || !kitzoo::os::is_dir(input)) {
    KZ_LOG_ERROR("{} is not a directory", input.string());
    return 1;
  }
  std::erase_if(paths,
                [](const auto& path) { return !kitzoo::os::is_regular(path) || !cv::haveImageReader(path.string()); });
  std::ranges::sort(paths);

  // OcrModel is not thread-safe; images run one after another.
  const auto ocr = hrt::model::OcrModel::create(algos[0], algos[1]);
  Json output = {{"detection", algos[0].name},
                 {"recognition", algos[1].name},
                 {"task", "ocr"},
                 {"engine", engine},
                 {"images", Json::array()}};
  auto& images = output["images"];
  KZ_LOG_INFO("total images: {}", paths.size());
  const kitzoo::time::Stopwatch watch;
  std::size_t ran = 0;
  for (const auto& path : paths) {
    auto entry = run(*ocr, path);
    if (entry["error"].is_null())
      ++ran;
    else
      KZ_LOG_ERROR("{}: {}", path.string(), entry["error"].get<hrt::String>());
    images.push_back(std::move(entry));
  }
  const auto elapsed = watch.elapsed_as<std::chrono::duration<double, std::milli>>().count();
  KZ_LOG_INFO("total {:.1f} ms, {:.2f} ms per image, {}/{} images inferred", elapsed,
              paths.empty() ? 0.0 : elapsed / static_cast<double>(paths.size()), ran, paths.size());

  auto dir = std::filesystem::absolute(input, ec).lexically_normal();
  if (!dir.has_filename())
    dir = dir.parent_path();
  const auto path = args.count("output") != 0U ? std::filesystem::path(args["output"].as<hrt::String>())
                                               : dir.parent_path() / (dir.filename().string() + "_ocr.json");
  ec.clear();
  kitzoo::os::write_file(path, output.dump(2, ' ', false, Json::error_handler_t::replace) + "\n", ec);
  if (ec) {
    KZ_LOG_ERROR("cannot write {}: {}", path.string(), ec.message());
    return 1;
  }
  KZ_LOG_INFO("{} images of OCR written to {}", images.size(), path.string());
  return ran == paths.size() ? 0 : 1;
}
