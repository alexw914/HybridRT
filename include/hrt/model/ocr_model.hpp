// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/model/ocr_model.hpp
// Description: Declares sequential text detection and recognition over two algos.
// -----------------------------------------------------------------------------

#ifndef HRT_MODEL_OCR_MODEL_HPP
#define HRT_MODEL_OCR_MODEL_HPP

#include <hrt/hrtc/hrtc.hpp>
#include <hrt/postprocess/ocr.hpp>

namespace hrt::model {

// Checks both stages without loading engines; unknown keys and mismatched roles are errors.
auto check_ocr_config(const hrtc::AlgoParam& detection, const hrtc::AlgoParam& recognition) -> Vector<String>;

// One engine per stage, reusable buffers, no pools. Not thread-safe.
class OcrModel {
public:
  // Invalid config or no usable engine aborts, like ImageModel::create.
  static auto create(const hrtc::AlgoParam& detection, const hrtc::AlgoParam& recognition) -> UniquePtr<OcrModel>;

  static auto create(const hrtc::Package& package, std::string_view detection, std::string_view recognition)
      -> UniquePtr<OcrModel>;

  ~OcrModel();

  OcrModel(const OcrModel&) = delete;
  auto operator=(const OcrModel&) -> OcrModel& = delete;

  // Host or device image, results in source pixels. Empty detection is a successful empty result.
  auto process_sync(const core::Image& src) -> std::optional<postprocess::OcrResult>;

private:
  OcrModel();

  struct Impl;
  UniquePtr<Impl> impl_;
};

} // namespace hrt::model

#endif // HRT_MODEL_OCR_MODEL_HPP
