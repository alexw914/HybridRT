// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: examples/image_model.cpp
// Description: Runs one algorithm of an .hrtc package or a JSON manifest on an
//              image file and saves the drawn results.
// -----------------------------------------------------------------------------

#include <hrt/core/view.hpp>
#include <hrt/model/image_model.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/log/logger.hpp>

#include <opencv2/imgcodecs.hpp>

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hybridrt_example_image_model", "Run an algorithm of an .hrtc package or a manifest");
  auto add = options.add_options();
  add("m,model", ".hrtc package or .json manifest", kitzoo::core::value<hrt::String>());
  add("i,image", "Input image", kitzoo::core::value<hrt::String>());
  add("o,output", "Output image with drawn results", kitzoo::core::value<hrt::String>()->default_value("result.jpg"));
  add("a,algo", "Algo name (default: the first one)", kitzoo::core::value<hrt::String>());
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  if (args.count("help") != 0U || args.count("model") == 0U || args.count("image") == 0U) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0U ? 0 : 1;
  }

  // A manifest builds the model from its algo entry, a package from the packed algo.
  const std::filesystem::path model_path = args["model"].as<hrt::String>();
  const auto wanted = args.count("algo") != 0U ? args["algo"].as<hrt::String>() : hrt::String{};
  const bool from_json = model_path.extension() == ".json";
  std::optional<hrt::hrtc::Package> package;
  const hrt::hrtc::AlgoParam* packed = nullptr;
  kitzoo::json::Json entry;
  hrt::String name;
  std::optional<hrt::model::Task> task;
  if (from_json) {
    const kitzoo::json::Reader manifest(model_path);
    // Reads `key` of `json` when both are present, without throwing.
    const auto field = [](const kitzoo::json::Json& json, const char* key) -> const kitzoo::json::Json* {
      const auto it = json.is_object() ? json.find(key) : json.end();
      return it != json.end() ? &*it : nullptr;
    };
    const auto* algos = field(manifest.raw(), "algos");
    for (const auto& item : algos != nullptr && algos->is_array() ? *algos : kitzoo::json::Json::array()) {
      const auto* algo_name = field(item, "name");
      if (algo_name != nullptr && algo_name->is_string() && (wanted.empty() || *algo_name == wanted)) {
        entry = item;
        name = algo_name->get<hrt::String>();
        break;
      }
    }
    const auto* config = field(entry, "config");
    const auto* type = config != nullptr ? field(*config, "postprocess") : nullptr;
    type = type != nullptr ? field(*type, "type") : nullptr;
    if (type != nullptr && type->is_string())
      task = hrt::model::task_of(type->get_ref<const hrt::String&>());
  } else {
    package = hrt::hrtc::load(model_path);
    if (package && !package->algos.empty())
      packed = wanted.empty() ? &package->algos.front() : package->find(wanted);
    name = packed != nullptr ? packed->name : hrt::String{};
    task = packed != nullptr ? hrt::model::task_of(*packed) : std::nullopt;
  }
  if (name.empty()) {
    KZ_LOG_ERROR("no algo {}in {}", wanted.empty() ? hrt::String{} : wanted + " ", model_path.string());
    return 1;
  }

  const auto path = args["image"].as<hrt::String>();
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    KZ_LOG_ERROR("cannot read {}", path);
    return 1;
  }
  const auto image = hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                            {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);

  if (!image || !task) {
    KZ_LOG_ERROR("unsupported image or postprocess type of algo {}", name);
    return 1;
  }
  const auto run = [&]<typename Result>() -> bool {
    // The first engine of the algo that runs on this machine.
    const auto model =
        from_json ? hrt::model::ImageModel<Result>::create(entry) : hrt::model::ImageModel<Result>::create(*packed);
    const auto result = model->process_sync(*image);
    if (!result)
      return false;
    KZ_LOG_INFO("{} on {}x{} {}", name, bgr.cols, bgr.rows, path);
    hrt::core::draw(bgr, *result);
    return true;
  };
  bool ok = false;
  switch (*task) {
  case hrt::model::Task::Detect:
    ok = run.operator()<hrt::postprocess::DetectResultMap>();
    break;
  case hrt::model::Task::Rotated:
    ok = run.operator()<hrt::postprocess::RotatedResultMap>();
    break;
  case hrt::model::Task::Classify:
    ok = run.operator()<hrt::postprocess::ClassifyResult>();
    break;
  case hrt::model::Task::Keypoint:
    ok = run.operator()<hrt::postprocess::KeypointResultMap>();
    break;
  case hrt::model::Task::Segment:
    ok = run.operator()<hrt::postprocess::SegmentResultMap>();
    break;
  }
  if (!ok)
    return 1;

  const auto output = args["output"].as<hrt::String>();
  if (!cv::imwrite(output, bgr)) {
    KZ_LOG_ERROR("cannot write {}", output);
    return 1;
  }
  KZ_LOG_INFO("saved {}", output);
  return 0;
}
