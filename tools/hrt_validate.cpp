// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tools/hrt_validate.cpp
// Description: Runs one algo of an .hrtc package or a JSON manifest on every
//              image of a directory and writes the results as JSON
//              (assets/template/results.json).
// -----------------------------------------------------------------------------

#include <hrt/core/view.hpp>
#include <hrt/model/image_model.hpp>

#include <kitzoo/core/cli.hpp>
#include <kitzoo/core/scopeguard.hpp>
#include <kitzoo/log/logger.hpp>
#include <kitzoo/os/filesys.hpp>
#include <kitzoo/thread/thread_pool.hpp>
#include <kitzoo/time/time.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <numeric>
#include <opencv2/imgcodecs.hpp>

namespace {

// Keeps the field order of assets/template/results.json.
using Json = nlohmann::ordered_json;

constexpr std::size_t kTopK = 5;

constexpr std::size_t kThreads = 8;

// 4 decimals keep the file diffable; floats would otherwise print as 0.9100000262.
auto round4(float value) -> double {
  return std::round(static_cast<double>(value) * 1e4) / 1e4;
}

// Every task fills the same fields; unused ones stay empty.
auto make_result(const hrt::postprocess::ObjectBox& box) -> Json {
  return {{"class_id", box.type},
          {"score", round4(box.score)},
          {"x", round4(box.rect.x)},
          {"y", round4(box.rect.y)},
          {"width", round4(box.rect.width)},
          {"height", round4(box.rect.height)},
          {"angle", 0.0},
          {"keypoints", Json::array()},
          {"mask", nullptr},
          {"cls", Json::array()}};
}

// Row-major run lengths alternating 0 and 1, starting with 0.
auto to_json(const hrt::postprocess::Mask& mask) -> Json {
  Json rle = Json::array();
  std::uint8_t value = 0;
  std::size_t count = 0;
  for (const auto pixel : mask.data) {
    if ((pixel != 0) != (value != 0)) {
      rle.push_back(count);
      value ^= 1U;
      count = 0;
    }
    ++count;
  }
  rle.push_back(count);
  return {{"rect", {mask.rect.x, mask.rect.y, mask.rect.width, mask.rect.height}}, {"rle", std::move(rle)}};
}

auto to_json(const hrt::postprocess::DetectResultMap& result, const hrt::core::Rect& /*region*/) -> Json {
  Json results = Json::array();
  for (const auto& [cls, boxes] : result)
    for (const auto& box : boxes)
      results.push_back(make_result(box));
  return results;
}

// x and y are the top-left corner of the unrotated box.
auto to_json(const hrt::postprocess::RotatedResultMap& result, const hrt::core::Rect& /*region*/) -> Json {
  Json results = Json::array();
  for (const auto& [cls, boxes] : result) {
    for (const auto& box : boxes) {
      const auto& rect = box.rect;
      auto item = make_result({{rect.center.x - (rect.size.width / 2.0F), rect.center.y - (rect.size.height / 2.0F),
                                rect.size.width, rect.size.height},
                               box.score,
                               box.type,
                               box.track_id});
      item["angle"] = round4(rect.angle);
      results.push_back(std::move(item));
    }
  }
  return results;
}

// One result over `region`; class_id and score are the top 1 of output 0.
auto to_json(const hrt::postprocess::ClassifyResult& result, const hrt::core::Rect& region) -> Json {
  hrt::postprocess::ObjectBox box{{static_cast<float>(region.x), static_cast<float>(region.y),
                                   static_cast<float>(region.width), static_cast<float>(region.height)},
                                  0.0F,
                                  -1,
                                  -1};
  Json cls = Json::array();
  for (std::size_t output = 0; output < result.size(); ++output) {
    const auto& scores = result[output];
    hrt::Vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0U);
    const auto k = std::min(kTopK, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                      [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
    Json topk = Json::array();
    for (std::size_t i = 0; i < k; ++i)
      topk.push_back({order[i], round4(scores[order[i]])});
    Json all = Json::array();
    for (const auto score : scores)
      all.push_back(round4(score));
    if (output == 0 && k > 0) {
      box.type = static_cast<std::int64_t>(order[0]);
      box.score = scores[order[0]];
    }
    cls.push_back({{"output", output}, {"topk", std::move(topk)}, {"scores", std::move(all)}});
  }
  auto item = make_result(box);
  item["cls"] = std::move(cls);
  return Json::array({std::move(item)});
}

auto to_json(const hrt::postprocess::KeypointResultMap& result, const hrt::core::Rect& /*region*/) -> Json {
  Json results = Json::array();
  for (const auto& [cls, objects] : result) {
    for (const auto& object : objects) {
      auto item = make_result(object.box);
      for (const auto& point : object.keypoints)
        item["keypoints"].push_back(
            {round4(point.point.x), round4(point.point.y), round4(point.score), static_cast<int>(point.visible)});
      results.push_back(std::move(item));
    }
  }
  return results;
}

auto to_json(const hrt::postprocess::SegmentResultMap& result, const hrt::core::Rect& /*region*/) -> Json {
  Json results = Json::array();
  for (const auto& [cls, objects] : result) {
    for (const auto& object : objects) {
      auto item = make_result(object.box);
      item["mask"] = to_json(object.mask);
      results.push_back(std::move(item));
    }
  }
  return results;
}

// Runs `algo` on every readable image in `dir`; all images are submitted to
// kThreads workers first, then the results are collected in name order while
// progress is logged every second. Failures are recorded in "error" and make
// the run fail. A non-empty `draw_dir` receives each image with its results
// drawn.
template <typename Result>
auto validate(const hrt::hrtc::AlgoParam& algo, const hrt::model::ImageModelOptions& options,
              const std::filesystem::path& dir, const std::filesystem::path& draw_dir, Json& images) -> bool {
  const auto model = hrt::model::ImageModel<Result>::create(algo, options);
  std::error_code ec;
  auto paths = kitzoo::os::list_directory(dir, ec);
  if (ec) {
    KZ_LOG_ERROR("cannot list {}: {}", dir.string(), ec.message());
    return false;
  }
  std::erase_if(paths,
                [](const auto& path) { return !kitzoo::os::is_regular(path) || !cv::haveImageReader(path.string()); });

  KZ_LOG_INFO("total images: {}", paths.size());
  std::atomic<std::size_t> done = 0;
  std::atomic<std::size_t> ran = 0;
  const auto run = [&](const std::filesystem::path& path) -> Json {
    KZ_SCOPE_EXIT {
      ++done;
    };
    Json entry = {{"file", path.filename().string()},
                  {"width", 0},
                  {"height", 0},
                  {"error", nullptr},
                  {"results", Json::array()}};
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    const auto image =
        bgr.empty() ? std::nullopt
                    : hrt::core::Image::wrap(hrt::core::MemoryType::Cpu, bgr.data, hrt::core::ImageFormat::InterBgr,
                                             {bgr.cols, bgr.rows}, hrt::core::DataType::Uint8, bgr.step);
    const auto result = image ? model->process_sync(*image) : std::nullopt;
    if (!image) {
      entry["error"] = "cannot read the image";
      return entry;
    }
    if (!result) {
      entry["error"] = "inference failed";
      return entry;
    }
    ++ran;
    entry["width"] = bgr.cols;
    entry["height"] = bgr.rows;
    entry["results"] = to_json(*result, {0, 0, bgr.cols, bgr.rows});
    if (!draw_dir.empty()) {
      hrt::core::draw(bgr, *result);
      cv::imwrite((draw_dir / path.filename()).string(), bgr);
    }
    return entry;
  };

  const kitzoo::time::Stopwatch watch;
  kitzoo::thread::ThreadPool pool(kThreads, "validate");
  hrt::Vector<std::future<Json>> pending;
  pending.reserve(paths.size());
  for (const auto& path : paths)
    pending.push_back(pool.submit_task(run, path));
  kitzoo::time::Stopwatch tick;
  const auto progress = [&] {
    if (tick.elapsed() < std::chrono::seconds(1))
      return;
    KZ_LOG_INFO("task {}/{}", done.load(), paths.size());
    tick.reset();
  };
  bool ok = true;
  for (std::size_t i = 0; i < pending.size(); ++i) {
    while (pending[i].wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
      progress();
    progress();
    auto entry = pending[i].get();
    if (!entry["error"].is_null()) {
      KZ_LOG_ERROR("{}: {}", paths[i].string(), entry["error"].get<hrt::String>());
      ok = false;
    }
    images.push_back(std::move(entry));
  }
  const auto elapsed = watch.elapsed_as<std::chrono::duration<double, std::milli>>().count();
  KZ_LOG_INFO("total {:.1f} ms, {:.2f} ms per image, {}/{} images inferred", elapsed,
              paths.empty() ? 0.0 : elapsed / static_cast<double>(paths.size()), ran.load(), paths.size());
  return ok;
}

} // namespace

auto main(int argc, char** argv) -> int {
  kitzoo::core::Options options("hrt-validate", "Run one algo of an .hrtc package or manifest on an image directory");
  auto add = options.add_options();
  add("m,model", ".hrtc package or .json manifest", kitzoo::core::value<hrt::String>());
  add("n,name", "Algo name", kitzoo::core::value<hrt::String>());
  add("i,input", "Input image directory", kitzoo::core::value<hrt::String>());
  add("e,engine", "Engine type, e.g. ort (default: the first usable one)", kitzoo::core::value<hrt::String>());
  add("preprocess-num", "Concurrent preprocessors, 1 to 64", kitzoo::core::value<std::size_t>()->default_value("3"));
  add("engine-num", "Concurrent engines, 1 to 64", kitzoo::core::value<std::size_t>()->default_value("2"));
  add("postprocess-num", "Concurrent postprocessors, 1 to 64", kitzoo::core::value<std::size_t>()->default_value("3"));
  add("o,output", "Output JSON file (default: <input>_<name>.json beside the input directory)",
      kitzoo::core::value<hrt::String>());
  add("r,result-path", "Directory for images with drawn results (default: none saved)",
      kitzoo::core::value<hrt::String>());
  add("h,help", "Print help");
  const auto args = options.parse(argc, argv);
  const auto in_range = [&](const char* key) -> bool {
    const auto count = args[key].as<std::size_t>();
    return count >= 1 && count <= 64;
  };
  if (args.count("help") != 0U || args.count("model") == 0U || args.count("name") == 0U || args.count("input") == 0U ||
      !in_range("preprocess-num") || !in_range("engine-num") || !in_range("postprocess-num")) {
    KZ_LOG_INFO("\n{}", options.help());
    return args.count("help") != 0U ? 0 : 1;
  }

  const hrt::model::ImageModelOptions model_options{args["preprocess-num"].as<std::size_t>(),
                                                    args["engine-num"].as<std::size_t>(),
                                                    args["postprocess-num"].as<std::size_t>()};
  const std::filesystem::path model = args["model"].as<hrt::String>();
  const auto name = args["name"].as<hrt::String>();
  const std::filesystem::path input = args["input"].as<hrt::String>();
  const auto package = model.extension() == ".json" ? hrt::hrtc::load_json(model) : hrt::hrtc::load(model);
  if (!package)
    return 1;
  const auto* found = package->find(name);
  if (found == nullptr) {
    KZ_LOG_ERROR("no algo {} in {}", name, model.string());
    return 1;
  }
  auto algo = *found;
  Json engine = nullptr;
  if (args.count("engine") != 0U) {
    const auto type = hrt::engine::engine_type_from(args["engine"].as<hrt::String>());
    std::erase_if(algo.engines,
                  [&](const auto& param) { return hrt::engine::engine_type_from(param.engine_type) != type; });
    if (algo.engines.empty()) {
      KZ_LOG_ERROR("algo {} has no {} engine", name, args["engine"].as<hrt::String>());
      return 1;
    }
    engine = algo.engines.front().engine_type;
  }
  const auto task = hrt::model::task_of(algo);
  if (!task) {
    KZ_LOG_ERROR("algo {} has no known postprocess type", name);
    return 1;
  }
  if (!kitzoo::os::is_dir(input)) {
    KZ_LOG_ERROR("{} is not a directory", input.string());
    return 1;
  }

  std::filesystem::path draw_dir;
  if (args.count("result-path") != 0U) {
    draw_dir = args["result-path"].as<hrt::String>();
    std::error_code ec;
    kitzoo::os::make_directory(draw_dir, ec, true);
    if (ec) {
      KZ_LOG_ERROR("cannot create {}: {}", draw_dir.string(), ec.message());
      return 1;
    }
  }

  // task_of succeeded, so the config parses and has a postprocess type.
  const kitzoo::json::Reader config{algo.config};
  const auto& postprocess = config.raw()["postprocess"];
  constexpr std::array<const char*, 5> kTasks = {"detect", "rotated", "classify", "keypoint", "segment"};
  Json output = {{"model", name},
                 {"task", kTasks.at(static_cast<std::size_t>(*task))},
                 {"type", postprocess["type"].get<hrt::String>()},
                 {"engine", engine},
                 {"images", Json::array()}};
  auto& images = output["images"];
  bool ok = false;
  switch (*task) {
  case hrt::model::Task::Detect:
    ok = validate<hrt::postprocess::DetectResultMap>(algo, model_options, input, draw_dir, images);
    break;
  case hrt::model::Task::Rotated:
    ok = validate<hrt::postprocess::RotatedResultMap>(algo, model_options, input, draw_dir, images);
    break;
  case hrt::model::Task::Classify:
    ok = validate<hrt::postprocess::ClassifyResult>(algo, model_options, input, draw_dir, images);
    break;
  case hrt::model::Task::Keypoint:
    ok = validate<hrt::postprocess::KeypointResultMap>(algo, model_options, input, draw_dir, images);
    break;
  case hrt::model::Task::Segment:
    ok = validate<hrt::postprocess::SegmentResultMap>(algo, model_options, input, draw_dir, images);
    break;
  }

  std::error_code ec;
  auto dir = std::filesystem::absolute(input, ec).lexically_normal();
  if (!dir.has_filename())
    dir = dir.parent_path();
  const auto path = args.count("output") != 0U ? std::filesystem::path(args["output"].as<hrt::String>())
                                               : dir.parent_path() / (dir.filename().string() + "_" + name + ".json");
  ec.clear();
  kitzoo::os::write_file(path, output.dump(2, ' ', false, Json::error_handler_t::replace) + "\n", ec);
  if (ec) {
    KZ_LOG_ERROR("cannot write {}: {}", path.string(), ec.message());
    return 1;
  }
  KZ_LOG_INFO("{} images of {} written to {}", images.size(), name, path.string());
  return ok ? 0 : 1;
}
