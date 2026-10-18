// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt-sys/src/lib.rs
// Description: Declares the include/hrt.h types and functions for Rust.
// -----------------------------------------------------------------------------

//! Raw declarations of the HybridRT C API; see `include/hrt.h` for the
//! contracts. The `hybridrt` crate wraps them safely.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub type HrtStatus = c_int;
pub const HRT_STATUS_OK: HrtStatus = 0;
pub const HRT_STATUS_INVALID_ARGUMENT: HrtStatus = 1;
pub const HRT_STATUS_NOT_FOUND: HrtStatus = 2;
pub const HRT_STATUS_INVALID_CONFIG: HrtStatus = 3;
pub const HRT_STATUS_UNAVAILABLE: HrtStatus = 4;
pub const HRT_STATUS_PROCESS_FAILED: HrtStatus = 5;

pub type HrtTask = c_int;
pub const HRT_TASK_DETECT: HrtTask = 0;
pub const HRT_TASK_ROTATED: HrtTask = 1;
pub const HRT_TASK_CLASSIFY: HrtTask = 2;
pub const HRT_TASK_KEYPOINT: HrtTask = 3;
pub const HRT_TASK_SEGMENT: HrtTask = 4;

pub type HrtImageFormat = c_int;
pub const HRT_IMAGE_FORMAT_UNKNOWN: HrtImageFormat = 0;
pub const HRT_IMAGE_FORMAT_INTER_BGR: HrtImageFormat = 1;
pub const HRT_IMAGE_FORMAT_INTER_RGB: HrtImageFormat = 2;
pub const HRT_IMAGE_FORMAT_PLANAR_BGR: HrtImageFormat = 3;
pub const HRT_IMAGE_FORMAT_PLANAR_RGB: HrtImageFormat = 4;
pub const HRT_IMAGE_FORMAT_NV12: HrtImageFormat = 5;
pub const HRT_IMAGE_FORMAT_NV21: HrtImageFormat = 6;
pub const HRT_IMAGE_FORMAT_YU12: HrtImageFormat = 7;
pub const HRT_IMAGE_FORMAT_YV12: HrtImageFormat = 8;
pub const HRT_IMAGE_FORMAT_INTER_RGBA: HrtImageFormat = 9;
pub const HRT_IMAGE_FORMAT_GRAY: HrtImageFormat = 10;

pub type HrtLogLevel = c_int;
pub const HRT_LOG_LEVEL_TRACE: HrtLogLevel = 0;
pub const HRT_LOG_LEVEL_DEBUG: HrtLogLevel = 1;
pub const HRT_LOG_LEVEL_INFO: HrtLogLevel = 2;
pub const HRT_LOG_LEVEL_WARN: HrtLogLevel = 3;
pub const HRT_LOG_LEVEL_ERROR: HrtLogLevel = 4;
pub const HRT_LOG_LEVEL_FATAL: HrtLogLevel = 5;
pub const HRT_LOG_LEVEL_OFF: HrtLogLevel = 6;

#[repr(C)]
pub struct HrtPackage {
    _private: [u8; 0],
}

#[repr(C)]
pub struct HrtModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct HrtResult {
    _private: [u8; 0],
}

#[repr(C)]
pub struct HrtOcrModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct HrtOcrResult {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HrtOcrText {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub text: *const c_char,
    pub text_size: usize,
    pub detection_score: f32,
    pub recognition_score: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HrtModelOptions {
    pub preprocess: usize,
    pub engine: usize,
    pub postprocess: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HrtRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HrtImage {
    pub data: *const c_void,
    pub format: HrtImageFormat,
    pub width: i32,
    pub height: i32,
    pub row_pitch: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HrtBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    pub class_id: i64,
    pub track_id: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HrtRotatedBox {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
    pub score: f32,
    pub class_id: i64,
    pub track_id: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HrtKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub visible: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HrtMask {
    pub rect: HrtRect,
    pub data: *const u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HrtResultData {
    pub task: HrtTask,
    pub count: usize,
    pub boxes: *const HrtBox,
    pub rotated_boxes: *const HrtRotatedBox,
    pub keypoint_offsets: *const usize,
    pub keypoints: *const HrtKeypoint,
    pub masks: *const HrtMask,
    pub score_offsets: *const usize,
    pub scores: *const f32,
}

pub type HrtLogCallback =
    Option<unsafe extern "C" fn(level: HrtLogLevel, message: *const c_char, length: usize, user_data: *mut c_void)>;

extern "C" {
    pub fn hrt_ocr_model_create(
        package: *const HrtPackage,
        detection: *const c_char,
        recognition: *const c_char,
        model: *mut *mut HrtOcrModel,
    ) -> HrtStatus;
    pub fn hrt_ocr_model_destroy(model: *mut HrtOcrModel);
    pub fn hrt_ocr_result_create(result: *mut *mut HrtOcrResult) -> HrtStatus;
    pub fn hrt_ocr_result_destroy(result: *mut HrtOcrResult);
    pub fn hrt_ocr_result_count(result: *const HrtOcrResult) -> usize;
    pub fn hrt_ocr_result_text(result: *const HrtOcrResult, index: usize) -> *const HrtOcrText;
    pub fn hrt_ocr_model_process(
        model: *mut HrtOcrModel,
        image: *const HrtImage,
        result: *mut HrtOcrResult,
    ) -> HrtStatus;
    pub fn hrt_version() -> *const c_char;
    pub fn hrt_last_error() -> *const c_char;
    pub fn hrt_set_log_level(level: HrtLogLevel);
    pub fn hrt_set_log_callback(callback: HrtLogCallback, user_data: *mut c_void);
    pub fn hrt_engine_available(engine_type: *const c_char) -> c_int;

    pub fn hrt_package_load(path: *const c_char, package: *mut *mut HrtPackage) -> HrtStatus;
    pub fn hrt_package_load_memory(data: *const c_void, size: usize, package: *mut *mut HrtPackage) -> HrtStatus;
    pub fn hrt_package_destroy(package: *mut HrtPackage);
    pub fn hrt_package_algo_count(package: *const HrtPackage) -> usize;
    pub fn hrt_package_algo_name(package: *const HrtPackage, index: usize) -> *const c_char;

    pub fn hrt_model_options_default() -> HrtModelOptions;
    pub fn hrt_model_create(
        package: *const HrtPackage,
        algo: *const c_char,
        options: *const HrtModelOptions,
        model: *mut *mut HrtModel,
    ) -> HrtStatus;
    pub fn hrt_model_destroy(model: *mut HrtModel);
    pub fn hrt_model_task(model: *const HrtModel) -> HrtTask;
    pub fn hrt_model_algo_name(model: *const HrtModel) -> *const c_char;

    pub fn hrt_result_create(result: *mut *mut HrtResult) -> HrtStatus;
    pub fn hrt_result_destroy(result: *mut HrtResult);
    pub fn hrt_result_data(result: *const HrtResult) -> *const HrtResultData;
    pub fn hrt_model_process(
        model: *mut HrtModel,
        image: *const HrtImage,
        roi: *const HrtRect,
        result: *mut HrtResult,
    ) -> HrtStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    // Layouts of include/hrt.h on 64-bit targets.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn matches_c_layouts() {
        assert_eq!(size_of::<HrtModelOptions>(), 24);
        assert_eq!(size_of::<HrtRect>(), 16);
        assert_eq!(size_of::<HrtImage>(), 32);
        assert_eq!(size_of::<HrtBox>(), 40);
        assert_eq!(align_of::<HrtBox>(), 8);
        assert_eq!(size_of::<HrtRotatedBox>(), 40);
        assert_eq!(size_of::<HrtKeypoint>(), 16);
        assert_eq!(size_of::<HrtMask>(), 24);
        assert_eq!(size_of::<HrtResultData>(), 72);
        assert_eq!(size_of::<HrtOcrText>(), 40);
        assert_eq!(align_of::<HrtOcrText>(), 8);
    }
}
