// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/capi/hrt.h
// Description: Declares the C API over image models for language bindings.
// -----------------------------------------------------------------------------

#ifndef HRT_CAPI_HRT_H
#define HRT_CAPI_HRT_H

// C99 declarations require leading return types, including when included from C++.
// NOLINTBEGIN(modernize-use-trailing-return-type)

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(HRT_C_API_BUILD)
#define HRT_API __declspec(dllexport)
#else
#define HRT_API __declspec(dllimport)
#endif
#else
#define HRT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Values are part of the ABI; new ones are only appended.
typedef enum HrtStatus {
  HRT_STATUS_OK = 0,
  HRT_STATUS_INVALID_ARGUMENT = 1, // Null pointer, out-of-range option or unsupported image.
  HRT_STATUS_NOT_FOUND = 2,        // Package unreadable or algo missing.
  HRT_STATUS_INVALID_CONFIG = 3,   // The algo config fails check_algo_config.
  HRT_STATUS_UNAVAILABLE = 4,      // No engine of the algo is compiled in with a usable device.
  HRT_STATUS_PROCESS_FAILED = 5,   // Preprocessing, inference or postprocessing failed.
} HrtStatus;

typedef enum HrtTask {
  HRT_TASK_DETECT = 0,
  HRT_TASK_ROTATED = 1,
  HRT_TASK_CLASSIFY = 2,
  HRT_TASK_KEYPOINT = 3,
  HRT_TASK_SEGMENT = 4,
  HRT_TASK_EMBEDDING = 5,
} HrtTask;

// Same values as hrt::core::ImageFormat. Planes are laid out as: interleaved
// {h, w, c}, planar {3, h, w}, gray {1, h, w}, and YUV 4:2:0 {h * 3 / 2, w, 1}.
typedef enum HrtImageFormat {
  HRT_IMAGE_FORMAT_UNKNOWN = 0,
  HRT_IMAGE_FORMAT_INTER_BGR = 1,
  HRT_IMAGE_FORMAT_INTER_RGB = 2,
  HRT_IMAGE_FORMAT_PLANAR_BGR = 3,
  HRT_IMAGE_FORMAT_PLANAR_RGB = 4,
  HRT_IMAGE_FORMAT_NV12 = 5,
  HRT_IMAGE_FORMAT_NV21 = 6,
  HRT_IMAGE_FORMAT_YU12 = 7,
  HRT_IMAGE_FORMAT_YV12 = 8,
  HRT_IMAGE_FORMAT_INTER_RGBA = 9,
  HRT_IMAGE_FORMAT_GRAY = 10,
} HrtImageFormat;

// Same values as kitzoo::log::Level.
typedef enum HrtLogLevel {
  HRT_LOG_LEVEL_TRACE = 0,
  HRT_LOG_LEVEL_DEBUG = 1,
  HRT_LOG_LEVEL_INFO = 2,
  HRT_LOG_LEVEL_WARN = 3,
  HRT_LOG_LEVEL_ERROR = 4,
  HRT_LOG_LEVEL_FATAL = 5,
  HRT_LOG_LEVEL_OFF = 6,
} HrtLogLevel;

typedef struct HrtPackage HrtPackage;
typedef struct HrtModel HrtModel;
typedef struct HrtResult HrtResult;
typedef struct HrtOcrModel HrtOcrModel;
typedef struct HrtOcrResult HrtOcrResult;

typedef struct HrtOcrText {
  float x; // Axis-aligned text box in source pixels.
  float y;
  float width;
  float height;
  const char* text; // UTF-8, owned by the result; byte length is text_size.
  size_t text_size;
  float detection_score;
  float recognition_score;
} HrtOcrText;

// One engine per stage. Calls using the same model or result must not overlap.
HRT_API HrtStatus hrt_ocr_model_create(const HrtPackage* package, const char* detection, const char* recognition,
                                       HrtOcrModel** model);

HRT_API void hrt_ocr_model_destroy(HrtOcrModel* model);

HRT_API HrtStatus hrt_ocr_result_create(HrtOcrResult** result);

HRT_API void hrt_ocr_result_destroy(HrtOcrResult* result);

HRT_API size_t hrt_ocr_result_count(const HrtOcrResult* result);

// Valid until the next process call or result destruction; null for an invalid index.
HRT_API const HrtOcrText* hrt_ocr_result_text(const HrtOcrResult* result, size_t index);

// Calls each stage serves at once, 1 to 64.
typedef struct HrtModelOptions {
  size_t preprocess;
  size_t engine;
  size_t postprocess;
} HrtModelOptions;

typedef struct HrtRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
} HrtRect;

// 8-bit host pixels owned by the caller, read only during hrt_model_process.
typedef struct HrtImage {
  const void* data;
  HrtImageFormat format;
  int32_t width;
  int32_t height;
  size_t row_pitch; // Bytes between rows of a plane; 0 means packed rows.
} HrtImage;

// On failure clears the previous result. Input is borrowed for this call only.
HRT_API HrtStatus hrt_ocr_model_process(HrtOcrModel* model, const HrtImage* image, HrtOcrResult* result);

typedef struct HrtBox {
  float x;
  float y;
  float width;
  float height;
  float score;
  int64_t class_id;
  int64_t track_id; // -1 means unassigned.
} HrtBox;

// Angle in degrees, clockwise in image coordinates (y down).
typedef struct HrtRotatedBox {
  float center_x;
  float center_y;
  float width;
  float height;
  float angle;
  float score;
  int64_t class_id;
  int64_t track_id;
} HrtRotatedBox;

typedef struct HrtKeypoint {
  float x;
  float y;
  float score;
  int32_t visible; // As the model outputs it: 0 unlabeled, 1 invisible, 2 visible.
} HrtKeypoint;

// Binary, row-major pixels over `rect` in source coordinates; 1 marks the class.
typedef struct HrtMask {
  HrtRect rect;
  const uint8_t* data;
} HrtMask;

// Views into an HrtResult, valid until it is processed into again or
// destroyed. Objects are ordered by class ID; arrays a task does not produce
// are null. Fields are only appended.
typedef struct HrtResultData {
  HrtTask task;
  size_t count;                       // Objects; for classify and embedding, engine outputs.
  const HrtBox* boxes;                // Detect, keypoint and segment.
  const HrtRotatedBox* rotated_boxes; // Rotated.
  const size_t* keypoint_offsets;     // Keypoint: count + 1 offsets into keypoints.
  const HrtKeypoint* keypoints;
  const HrtMask* masks;        // Segment; semantic results cover the whole input region.
  const size_t* score_offsets; // Classify, embedding: count + 1 offsets into scores.
  const float* scores;         // Classify scores of each output by class ID; embedding: the vector, count 1.
} HrtResultData;

typedef void (*HrtLogCallback)(HrtLogLevel level, const char* message, size_t length, void* user_data);

HRT_API const char* hrt_version(void);

// Message of the last failed call on this thread; empty when none failed.
HRT_API const char* hrt_last_error(void);

HRT_API void hrt_set_log_level(HrtLogLevel level);

// Sends log records to `callback` from the logging thread instead of the
// console; null restores the console. Safe while other threads log; once it
// returns, the previous callback is no longer called. `callback` must not
// call hrt_set_log_callback.
HRT_API void hrt_set_log_callback(HrtLogCallback callback, void* user_data);

// Whether `engine_type` (e.g. "ort", "trt") is compiled into this build.
HRT_API int hrt_engine_available(const char* engine_type);

// A .json manifest (paths relative to the working directory) or an .hrtc package.
HRT_API HrtStatus hrt_package_load(const char* path, HrtPackage** package);

HRT_API HrtStatus hrt_package_load_memory(const void* data, size_t size, HrtPackage** package);

HRT_API void hrt_package_destroy(HrtPackage* package);

HRT_API size_t hrt_package_algo_count(const HrtPackage* package);

// Null when `index` is out of range.
HRT_API const char* hrt_package_algo_name(const HrtPackage* package, size_t index);

HRT_API HrtModelOptions hrt_model_options_default(void);

// Loads `algo` (null: the first one) with the first engine, in package order,
// that runs on this machine; `options` may be null. The package may be
// destroyed afterwards. Config, option and engine availability problems are
// returned; an engine whose outputs do not fit the config still aborts, as in
// hrt::model::ImageModel::create.
HRT_API HrtStatus hrt_model_create(const HrtPackage* package, const char* algo, const HrtModelOptions* options,
                                   HrtModel** model);

HRT_API void hrt_model_destroy(HrtModel* model);

HRT_API HrtTask hrt_model_task(const HrtModel* model);

HRT_API const char* hrt_model_algo_name(const HrtModel* model);

HRT_API HrtStatus hrt_result_create(HrtResult** result);

HRT_API void hrt_result_destroy(HrtResult* result);

HRT_API const HrtResultData* hrt_result_data(const HrtResult* result);

// Thread-safe for distinct results; results are in source coordinates.
// `roi` (null: the whole image) must lie inside the image unless the
// preprocess resize_type is pad_outside. Reusing `result` keeps its storage.
HRT_API HrtStatus hrt_model_process(HrtModel* model, const HrtImage* image, const HrtRect* roi, HrtResult* result);

// Static engine batch size; null returns zero.
HRT_API size_t hrt_model_batch(const HrtModel* model);

// Runs any positive count, including one, in groups of up to hrt_model_batch().
// images and results contain count entries; rois is null or contains count entries.
// Results must be distinct, valid handles. Output order matches input order.
// Unused samples in a partial engine batch are not decoded. Inputs remain alive
// until this synchronous call returns. Failure leaves existing results unchanged.
// Thread-safe for distinct results, as with hrt_model_process.
HRT_API HrtStatus hrt_model_process_batch(HrtModel* model, const HrtImage* images, const HrtRect* rois, size_t count,
                                          HrtResult* const* results);

#ifdef __cplusplus
} // extern "C"
#endif

// NOLINTEND(modernize-use-trailing-return-type)

#endif // HRT_CAPI_HRT_H
