// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/tests/template.rs
// Description: Runs template models on bus.jpg when the fixtures exist; the
//              template's paths are relative to the repository root, which
//              becomes this test binary's working directory.
// -----------------------------------------------------------------------------

mod common;

use hybridrt::{engine_available, Image, ImageFormat, Model, ModelOptions, Package, Task};

#[test]
fn runs_template_models_when_available() {
    let root = common::source_root();
    let models = root.join("assets/test/onnx");
    let Ok(jpeg) = std::fs::read(root.join("assets/test/images/bus.jpg")) else {
        return;
    };
    if !engine_available("ort") || !models.is_dir() {
        return;
    }
    let mut decoder = zune_jpeg::JpegDecoder::new(&jpeg);
    let rgb = decoder.decode().unwrap();
    let (width, height) = decoder.dimensions().unwrap();
    let image = Image::new(&rgb, ImageFormat::InterRgb, width as u32, height as u32).unwrap();

    std::env::set_current_dir(&root).unwrap();
    let package = Package::load("assets/template/hrtc_template.json").unwrap();
    let run = |algo: &str, model_file: &str| {
        if !models.join(model_file).is_file() {
            return None;
        }
        let model = Model::new(&package, Some(algo), ModelOptions::default()).unwrap();
        Some((model.task(), model.run(&image, None).unwrap()))
    };

    if let Some((task, results)) = run("yolov8n_detect", "GeneralDetect.onnx") {
        assert_eq!(task, Task::Detect);
        assert!(results.boxes().iter().any(|object| object.class_id == 0));
    }
    // Three people and the bus, as the C++ segment test reports.
    if let Some((task, results)) = run("yolo26n_e2e_segment", "E2ESegment.onnx") {
        assert_eq!(task, Task::Segment);
        let classes: Vec<i64> = results.boxes().iter().map(|object| object.class_id).collect();
        assert_eq!(classes, [0, 0, 0, 5]);
        for index in 0..results.len() {
            let mask = results.mask(index).unwrap();
            assert_eq!(mask.data.len(), (mask.rect.width * mask.rect.height) as usize);
            assert_eq!(mask.data.iter().max(), Some(&1));
        }
    }
    if let Some((task, results)) = run("yolov8n_pose", "GeneralYoloPose.onnx") {
        assert_eq!(task, Task::Keypoint);
        assert!(!results.is_empty());
        assert!((0..results.len()).all(|index| results.keypoints(index).unwrap().len() == 17));
    }
    if let Some((task, results)) = run("yolov8n_obb", "GeneralRotated.onnx") {
        assert_eq!(task, Task::Rotated);
        assert_eq!(results.rotated_boxes().len(), results.len());
    }
    if let Some((task, results)) = run("yolov8n_classify", "GeneralClassify.onnx") {
        assert_eq!(task, Task::Classify);
        assert_eq!(results.scores(0).unwrap().len(), 1000);
    }
}
