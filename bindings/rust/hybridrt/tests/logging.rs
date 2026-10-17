// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/tests/logging.rs
// Description: Tests forwarding library logs to a callback; it reconfigures
//              the process-wide logger, so it runs as its own test binary.
// -----------------------------------------------------------------------------

mod common;

use std::sync::{Arc, Mutex};

use common::{manifest, uniform_bgr, CLASSIFY};
use hybridrt::{
    clear_log_callback, engine_available, set_log_callback, Image, ImageFormat, LogLevel, Model, ModelOptions, Package,
    Rect,
};

#[test]
fn forwards_logs_to_callback() {
    if !engine_available("ort") {
        return;
    }
    let model = Model::new(
        &Package::load(manifest("logging", &[("classify", CLASSIFY, "ort")])).unwrap(),
        None,
        ModelOptions::default(),
    )
    .unwrap();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&errors);
    set_log_callback(move |level, message| {
        if level >= LogLevel::Error {
            sink.lock().unwrap().push(message.to_owned());
        }
    });
    let pixels = uniform_bgr();
    let image = Image::new(&pixels, ImageFormat::InterBgr, 8, 8).unwrap();
    assert!(model
        .run(
            &image,
            Some(Rect {
                x: 4,
                y: 4,
                width: 8,
                height: 8
            })
        )
        .is_err());
    clear_log_callback();
    assert!(errors.lock().unwrap().iter().any(|message| message.contains("leaves")));
    assert!(model.run(&image, None).is_ok());
}
