// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/src/ocr.rs
// Description: Wraps sequential OCR with exclusive access and owned text results.
// -----------------------------------------------------------------------------

use crate::{c_string, check, sys, Error, Image, Package, Result, Status};
use std::ptr::{self, NonNull};

#[derive(Clone, Debug)]
pub struct OcrText {
    pub rect: [f32; 4], // x, y, width, height in source pixels.
    pub text: String,
    pub detection_score: f32,
    pub recognition_score: f32,
}

struct OcrResults(NonNull<sys::HrtOcrResult>);

impl Drop for OcrResults {
    fn drop(&mut self) {
        // SAFETY: this wrapper owns the result and destroys it once.
        unsafe { sys::hrt_ocr_result_destroy(self.0.as_ptr()) }
    }
}

/// One engine per stage, without pools. `run` requires exclusive access.
pub struct OcrModel {
    raw: NonNull<sys::HrtOcrModel>,
    results: OcrResults,
}

impl OcrModel {
    /// Both names refer to algos in the same JSON/.hrtc package.
    pub fn new(package: &Package, detection: &str, recognition: &str) -> Result<Self> {
        let detection = c_string(detection, "the detection algo")?;
        let recognition = c_string(recognition, "the recognition algo")?;
        let mut result = ptr::null_mut();
        // SAFETY: valid out pointer; the result is owned below.
        check(unsafe { sys::hrt_ocr_result_create(&mut result) })?;
        let results = OcrResults(
            NonNull::new(result).ok_or_else(|| Error::new(Status::Unknown, "the library returned no OCR result"))?,
        );
        let mut raw = ptr::null_mut();
        // SAFETY: valid package, C strings and out pointer; the library retains its own resources.
        check(unsafe {
            sys::hrt_ocr_model_create(package.raw.as_ptr(), detection.as_ptr(), recognition.as_ptr(), &mut raw)
        })?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::new(Status::Unknown, "the library returned no OCR model"))?;
        Ok(Self { raw, results })
    }

    /// Results own their text and coordinates and survive later calls.
    pub fn run(&mut self, image: &Image<'_>) -> Result<Vec<OcrText>> {
        let native = image.to_raw();
        // SAFETY: exclusive model/result access; image bytes live through the call.
        check(unsafe { sys::hrt_ocr_model_process(self.raw.as_ptr(), &native, self.results.0.as_ptr()) })?;
        // SAFETY: valid result owned by this model.
        let count = unsafe { sys::hrt_ocr_result_count(self.results.0.as_ptr()) };
        let mut texts = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: index is below count; the result is not modified while copying it.
            let value = unsafe { &*sys::hrt_ocr_result_text(self.results.0.as_ptr(), index) };
            // SAFETY: the C API provides text_size readable bytes, valid through this copy.
            let bytes = unsafe { std::slice::from_raw_parts(value.text.cast::<u8>(), value.text_size) };
            let text = std::str::from_utf8(bytes)
                .map_err(|_| Error::new(Status::ProcessFailed, "OCR returned invalid UTF-8"))?
                .to_owned();
            texts.push(OcrText {
                rect: [value.x, value.y, value.width, value.height],
                text,
                detection_score: value.detection_score,
                recognition_score: value.recognition_score,
            });
        }
        Ok(texts)
    }
}

impl Drop for OcrModel {
    fn drop(&mut self) {
        // SAFETY: the model is owned and destroyed once; no calls can overlap &mut run.
        unsafe { sys::hrt_ocr_model_destroy(self.raw.as_ptr()) }
    }
}
