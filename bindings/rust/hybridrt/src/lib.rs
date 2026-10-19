// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/src/lib.rs
// Description: Wraps packages, image models, host images and results of the
//              C API safely.
// -----------------------------------------------------------------------------

//! Safe bindings of the HybridRT C API (`include/hrt.h`).
//!
//! ```no_run
//! use hybridrt::{Image, ImageFormat, Model, ModelOptions, Package};
//!
//! let package = Package::load("model.hrtc")?;
//! let model = Model::new(&package, Some("yolov8n_detect"), ModelOptions::default())?;
//! let pixels = vec![0u8; 640 * 480 * 3];
//! let image = Image::new(&pixels, ImageFormat::InterBgr, 640, 480)?;
//! let results = model.run(&image, None)?;
//! for object in results.boxes() {
//!     println!("{} {} {:?}", object.class_id, object.score, (object.x, object.y, object.width, object.height));
//! }
//! # Ok::<(), hybridrt::Error>(())
//! ```

use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::RwLock;

use hybridrt_sys as sys;

mod ocr;
pub use ocr::{OcrModel, OcrText};

pub use sys::HrtBox as ObjectBox;
pub use sys::HrtKeypoint as Keypoint;
pub use sys::HrtRect as Rect;
pub use sys::HrtRotatedBox as RotatedBox;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    InvalidArgument, // Null pointer, out-of-range option or unsupported image.
    NotFound,        // Package unreadable or algo missing.
    InvalidConfig,   // The algo config fails check_config.
    Unavailable,     // No engine of the algo is compiled in with a usable device.
    ProcessFailed,   // Preprocessing, inference or postprocessing failed.
    Unknown,         // A status this crate does not know.
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    status: Status,
    message: String,
}

impl Error {
    fn new(status: Status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// The error of a failed call on this thread, from `hrt_last_error`.
    fn last(status: sys::HrtStatus) -> Self {
        let status = match status {
            sys::HRT_STATUS_INVALID_ARGUMENT => Status::InvalidArgument,
            sys::HRT_STATUS_NOT_FOUND => Status::NotFound,
            sys::HRT_STATUS_INVALID_CONFIG => Status::InvalidConfig,
            sys::HRT_STATUS_UNAVAILABLE => Status::Unavailable,
            sys::HRT_STATUS_PROCESS_FAILED => Status::ProcessFailed,
            _ => Status::Unknown,
        };
        // SAFETY: hrt_last_error returns a valid C string owned by this thread.
        let message = unsafe { CStr::from_ptr(sys::hrt_last_error()) };
        Self::new(status, message.to_string_lossy())
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(status: sys::HrtStatus) -> Result<()> {
    if status == sys::HRT_STATUS_OK {
        Ok(())
    } else {
        Err(Error::last(status))
    }
}

fn c_string(text: &str, what: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error::new(Status::InvalidArgument, format!("{what} contains a NUL byte")))
}

/// # Safety
/// `pointer` must be null or point at `len` valid values that outlive `'a`.
unsafe fn slice<'a, T>(pointer: *const T, len: usize) -> &'a [T] {
    if pointer.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(pointer, len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Detect,
    Rotated,
    Classify,
    Keypoint,
    Segment,
    Embedding,
}

impl Task {
    fn from_raw(task: sys::HrtTask) -> Self {
        match task {
            sys::HRT_TASK_ROTATED => Self::Rotated,
            sys::HRT_TASK_CLASSIFY => Self::Classify,
            sys::HRT_TASK_KEYPOINT => Self::Keypoint,
            sys::HRT_TASK_SEGMENT => Self::Segment,
            sys::HRT_TASK_EMBEDDING => Self::Embedding,
            _ => Self::Detect,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
    Off,
}

impl LogLevel {
    fn from_raw(level: sys::HrtLogLevel) -> Self {
        match level {
            sys::HRT_LOG_LEVEL_TRACE => Self::Trace,
            sys::HRT_LOG_LEVEL_DEBUG => Self::Debug,
            sys::HRT_LOG_LEVEL_INFO => Self::Info,
            sys::HRT_LOG_LEVEL_WARN => Self::Warn,
            sys::HRT_LOG_LEVEL_ERROR => Self::Error,
            sys::HRT_LOG_LEVEL_FATAL => Self::Fatal,
            _ => Self::Off,
        }
    }

    fn to_raw(self) -> sys::HrtLogLevel {
        match self {
            Self::Trace => sys::HRT_LOG_LEVEL_TRACE,
            Self::Debug => sys::HRT_LOG_LEVEL_DEBUG,
            Self::Info => sys::HRT_LOG_LEVEL_INFO,
            Self::Warn => sys::HRT_LOG_LEVEL_WARN,
            Self::Error => sys::HRT_LOG_LEVEL_ERROR,
            Self::Fatal => sys::HRT_LOG_LEVEL_FATAL,
            Self::Off => sys::HRT_LOG_LEVEL_OFF,
        }
    }
}

pub fn version() -> &'static str {
    // SAFETY: hrt_version returns a static C string.
    unsafe { CStr::from_ptr(sys::hrt_version()) }
        .to_str()
        .unwrap_or_default()
}

/// Whether `engine_type` (e.g. "ort", "trt") is compiled into the library.
pub fn engine_available(engine_type: &str) -> bool {
    // SAFETY: the argument is a valid C string.
    CString::new(engine_type).is_ok_and(|name| unsafe { sys::hrt_engine_available(name.as_ptr()) } != 0)
}

pub fn set_log_level(level: LogLevel) {
    // SAFETY: plain value argument.
    unsafe { sys::hrt_set_log_level(level.to_raw()) }
}

type LogCallback = dyn Fn(LogLevel, &str) + Send + Sync;

static LOG_CALLBACK: RwLock<Option<Box<LogCallback>>> = RwLock::new(None);

unsafe extern "C" fn forward_log(level: sys::HrtLogLevel, message: *const c_char, length: usize, _: *mut c_void) {
    let text = String::from_utf8_lossy(slice(message.cast::<u8>(), length));
    if let Ok(callback) = LOG_CALLBACK.read() {
        if let Some(callback) = callback.as_ref() {
            callback(LogLevel::from_raw(level), &text);
        }
    }
}

/// Sends library log records to `callback` from the logging thread instead of
/// the console; safe while other threads log. The callback must not panic or
/// change the callback.
pub fn set_log_callback(callback: impl Fn(LogLevel, &str) + Send + Sync + 'static) {
    *LOG_CALLBACK.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Box::new(callback));
    // SAFETY: forward_log stays valid for the program's lifetime.
    unsafe { sys::hrt_set_log_callback(Some(forward_log), ptr::null_mut()) }
}

/// Restores console logging.
pub fn clear_log_callback() {
    // SAFETY: a null callback restores the console.
    unsafe { sys::hrt_set_log_callback(None, ptr::null_mut()) }
    *LOG_CALLBACK.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

/// An .hrtc package or a .json manifest whose relative paths resolve against
/// the working directory.
pub struct Package {
    raw: NonNull<sys::HrtPackage>,
}

// SAFETY: a loaded package is only read.
unsafe impl Send for Package {}
unsafe impl Sync for Package {}

impl Package {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = path
            .to_str()
            .ok_or_else(|| Error::new(Status::InvalidArgument, format!("{} is not UTF-8", path.display())))?;
        let path = c_string(text, "the package path")?;
        let mut raw = ptr::null_mut();
        // SAFETY: valid C string and out pointer.
        check(unsafe { sys::hrt_package_load(path.as_ptr(), &mut raw) })?;
        Self::wrap(raw)
    }

    /// An .hrtc package in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: the bytes are valid for the call and the out pointer is valid.
        check(unsafe { sys::hrt_package_load_memory(bytes.as_ptr().cast(), bytes.len(), &mut raw) })?;
        Self::wrap(raw)
    }

    fn wrap(raw: *mut sys::HrtPackage) -> Result<Self> {
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| Error::new(Status::Unknown, "the library returned no package"))
    }

    pub fn algos(&self) -> Vec<String> {
        // SAFETY: the package is valid; names live as long as it.
        let count = unsafe { sys::hrt_package_algo_count(self.raw.as_ptr()) };
        (0..count)
            .map(|index| unsafe { CStr::from_ptr(sys::hrt_package_algo_name(self.raw.as_ptr(), index)) })
            .map(|name| name.to_string_lossy().into_owned())
            .collect()
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        // SAFETY: the package is owned and destroyed once.
        unsafe { sys::hrt_package_destroy(self.raw.as_ptr()) }
    }
}

/// How many calls each stage serves at once, 1 to 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOptions {
    pub preprocess: usize,
    pub engine: usize,
    pub postprocess: usize,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            preprocess: 1,
            engine: 1,
            postprocess: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    InterBgr,
    InterRgb,
    PlanarBgr,
    PlanarRgb,
    Nv12,
    Nv21,
    Yu12,
    Yv12,
    InterRgba,
    Gray,
}

impl ImageFormat {
    fn to_raw(self) -> sys::HrtImageFormat {
        match self {
            Self::InterBgr => sys::HRT_IMAGE_FORMAT_INTER_BGR,
            Self::InterRgb => sys::HRT_IMAGE_FORMAT_INTER_RGB,
            Self::PlanarBgr => sys::HRT_IMAGE_FORMAT_PLANAR_BGR,
            Self::PlanarRgb => sys::HRT_IMAGE_FORMAT_PLANAR_RGB,
            Self::Nv12 => sys::HRT_IMAGE_FORMAT_NV12,
            Self::Nv21 => sys::HRT_IMAGE_FORMAT_NV21,
            Self::Yu12 => sys::HRT_IMAGE_FORMAT_YU12,
            Self::Yv12 => sys::HRT_IMAGE_FORMAT_YV12,
            Self::InterRgba => sys::HRT_IMAGE_FORMAT_INTER_RGBA,
            Self::Gray => sys::HRT_IMAGE_FORMAT_GRAY,
        }
    }

    /// Rows and bytes per row: interleaved {h, w * c}, planar {3 * h, w},
    /// gray {h, w} and YUV 4:2:0 {h * 3 / 2, w}.
    fn rows(self, width: usize, height: usize) -> (usize, usize) {
        match self {
            Self::InterBgr | Self::InterRgb => (height, width * 3),
            Self::InterRgba => (height, width * 4),
            Self::PlanarBgr | Self::PlanarRgb => (height * 3, width),
            Self::Gray => (height, width),
            Self::Nv12 | Self::Nv21 | Self::Yu12 | Self::Yv12 => (height * 3 / 2, width),
        }
    }
}

/// 8-bit host pixels borrowed for processing.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    data: &'a [u8],
    format: ImageFormat,
    width: i32,
    height: i32,
    row_pitch: usize,
}

impl<'a> Image<'a> {
    /// Packed rows.
    pub fn new(data: &'a [u8], format: ImageFormat, width: u32, height: u32) -> Result<Self> {
        Self::with_row_pitch(data, format, width, height, 0)
    }

    /// `row_pitch` is the bytes between rows of a plane; 0 means packed rows.
    pub fn with_row_pitch(
        data: &'a [u8],
        format: ImageFormat,
        width: u32,
        height: u32,
        row_pitch: usize,
    ) -> Result<Self> {
        let invalid = |message: String| Error::new(Status::InvalidArgument, message);
        let (Ok(w), Ok(h)) = (i32::try_from(width), i32::try_from(height)) else {
            return Err(invalid(format!("image size {width}x{height} is too large")));
        };
        if w == 0 || h == 0 {
            return Err(invalid("image size must be positive".into()));
        }
        let yuv = matches!(
            format,
            ImageFormat::Nv12 | ImageFormat::Nv21 | ImageFormat::Yu12 | ImageFormat::Yv12
        );
        if yuv && (width % 2 != 0 || height % 2 != 0) {
            return Err(invalid(format!("{format:?} needs an even size, got {width}x{height}")));
        }
        let (rows, row_bytes) = format.rows(width as usize, height as usize);
        let pitch = if row_pitch == 0 { row_bytes } else { row_pitch };
        if pitch < row_bytes {
            return Err(invalid(format!(
                "row pitch {pitch} is smaller than a row ({row_bytes})"
            )));
        }
        let needed = (rows - 1) * pitch + row_bytes;
        if data.len() < needed {
            return Err(invalid(format!(
                "{format:?} {width}x{height} needs {needed} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            format,
            width: w,
            height: h,
            row_pitch,
        })
    }

    fn to_raw(self) -> sys::HrtImage {
        sys::HrtImage {
            data: self.data.as_ptr().cast(),
            format: self.format.to_raw(),
            width: self.width,
            height: self.height,
            row_pitch: self.row_pitch,
        }
    }
}

/// Binary, row-major pixels over `rect` in source coordinates; 1 marks the class.
#[derive(Clone, Copy, Debug)]
pub struct Mask<'a> {
    pub rect: Rect,
    pub data: &'a [u8],
}

/// Results of one call, in source coordinates and ordered by class ID;
/// reusing it for the next call keeps its storage.
pub struct Results {
    raw: NonNull<sys::HrtResult>,
}

// SAFETY: the result is plain data, written only through &mut.
unsafe impl Send for Results {}
unsafe impl Sync for Results {}

impl Results {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: valid out pointer.
        check(unsafe { sys::hrt_result_create(&mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| Error::new(Status::Unknown, "the library returned no result"))
    }

    fn data(&self) -> &sys::HrtResultData {
        // SAFETY: the data of a valid result lives as long as it.
        unsafe { &*sys::hrt_result_data(self.raw.as_ptr()) }
    }

    pub fn task(&self) -> Task {
        Task::from_raw(self.data().task)
    }

    /// Objects; for classify and embedding, engine outputs.
    pub fn len(&self) -> usize {
        self.data().count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Detect, keypoint and segment objects.
    pub fn boxes(&self) -> &[ObjectBox] {
        let data = self.data();
        // SAFETY: boxes holds count entries when present.
        unsafe { slice(data.boxes, data.count) }
    }

    pub fn rotated_boxes(&self) -> &[RotatedBox] {
        let data = self.data();
        // SAFETY: rotated_boxes holds count entries when present.
        unsafe { slice(data.rotated_boxes, data.count) }
    }

    /// Keypoints of object `index`; None for other tasks or out of range.
    pub fn keypoints(&self, index: usize) -> Option<&[Keypoint]> {
        let data = self.data();
        if data.keypoint_offsets.is_null() || index >= data.count {
            return None;
        }
        // SAFETY: keypoint_offsets holds count + 1 offsets into keypoints.
        unsafe {
            let offsets = slice(data.keypoint_offsets, data.count + 1);
            let (begin, end) = (offsets[index], offsets[index + 1]);
            Some(
                slice(data.keypoints, offsets[data.count])
                    .get(begin..end)
                    .unwrap_or_default(),
            )
        }
    }

    /// Mask of segment object `index`; None for other tasks or out of range.
    pub fn mask(&self, index: usize) -> Option<Mask<'_>> {
        let data = self.data();
        if data.masks.is_null() || index >= data.count {
            return None;
        }
        // SAFETY: masks holds count entries, each over width * height bytes.
        unsafe {
            let mask = &*data.masks.add(index);
            let size = mask.rect.width.max(0) as usize * mask.rect.height.max(0) as usize;
            Some(Mask {
                rect: mask.rect,
                data: slice(mask.data, size),
            })
        }
    }

    /// Classify scores of engine output `output` by class ID; for embedding,
    /// output 0 is the embedding.
    pub fn scores(&self, output: usize) -> Option<&[f32]> {
        let data = self.data();
        if data.score_offsets.is_null() || output >= data.count {
            return None;
        }
        // SAFETY: score_offsets holds count + 1 offsets into scores.
        unsafe {
            let offsets = slice(data.score_offsets, data.count + 1);
            slice(data.scores, offsets[data.count]).get(offsets[output]..offsets[output + 1])
        }
    }
}

impl Drop for Results {
    fn drop(&mut self) {
        // SAFETY: the result is owned and destroyed once.
        unsafe { sys::hrt_result_destroy(self.raw.as_ptr()) }
    }
}

/// One algo of a package, run by the first of its engines that works on this
/// machine. Calls may run from several threads at once.
pub struct Model {
    raw: NonNull<sys::HrtModel>,
    task: Task,
    algo: String,
}

// SAFETY: hrt_model_process is thread-safe for distinct results.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Loads `algo`, or the first algo of the package; the package may be
    /// dropped afterwards. An engine whose outputs do not fit the postprocess
    /// config still aborts the process, as in C++.
    pub fn new(package: &Package, algo: Option<&str>, options: ModelOptions) -> Result<Self> {
        let algo = algo.map(|name| c_string(name, "the algo name")).transpose()?;
        let options = sys::HrtModelOptions {
            preprocess: options.preprocess,
            engine: options.engine,
            postprocess: options.postprocess,
        };
        let mut raw = ptr::null_mut();
        // SAFETY: valid package, optional C string, options and out pointer.
        check(unsafe {
            sys::hrt_model_create(
                package.raw.as_ptr(),
                algo.as_ref().map_or(ptr::null(), |name| name.as_ptr()),
                &options,
                &mut raw,
            )
        })?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::new(Status::Unknown, "the library returned no model"))?;
        // SAFETY: the model is valid; its name lives as long as it.
        let (task, algo) = unsafe {
            (
                Task::from_raw(sys::hrt_model_task(raw.as_ptr())),
                CStr::from_ptr(sys::hrt_model_algo_name(raw.as_ptr()))
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        Ok(Self { raw, task, algo })
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn algo(&self) -> &str {
        &self.algo
    }

    /// Runs on `image`; `roi` (None: the whole image) must lie inside it unless
    /// the preprocess resize_type is pad_outside.
    pub fn process(&self, image: &Image<'_>, roi: Option<Rect>, results: &mut Results) -> Result<()> {
        let raw = image.to_raw();
        let roi = roi.as_ref().map_or(ptr::null(), |rect| rect as *const Rect);
        // SAFETY: the image borrows its pixels for the call; the result is exclusively borrowed.
        check(unsafe { sys::hrt_model_process(self.raw.as_ptr(), &raw, roi, results.raw.as_ptr()) })
    }

    pub fn run(&self, image: &Image<'_>, roi: Option<Rect>) -> Result<Results> {
        let mut results = Results::new()?;
        self.process(image, roi, &mut results)?;
        Ok(results)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the model is owned and destroyed once.
        unsafe { sys::hrt_model_destroy(self.raw.as_ptr()) }
    }
}
