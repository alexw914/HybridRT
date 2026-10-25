// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/tests/ocr.rs
// Description: Tests OCR errors, both recognition models and owned results.
// -----------------------------------------------------------------------------

mod common;

use hybridrt::{engine_available, Image, ImageFormat, OcrModel, Package, Status};

#[test]
fn runs_ocr_models_when_available() {
    let root = common::source_root();
    let Ok(jpeg) = std::fs::read(root.join("assets/test/images/ocr_document.jpg")) else {
        return;
    };
    if !engine_available("ort") {
        return;
    }
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("assets/template/hrtc_template.json")).unwrap()).unwrap();
    let algos = config["algos"].as_array_mut().unwrap();
    algos.retain(|algo| algo["name"].as_str().unwrap().starts_with("ppocrv6_"));
    for algo in algos.iter_mut() {
        let backend = &mut algo["config"]["engine"]["backends"][0];
        let model = root.join(backend["data"].as_str().unwrap());
        if !model.is_file() {
            return;
        }
        backend["data"] = model.to_str().unwrap().into();
        if let Some(dictionary) = algo["config"]["postprocess"].get_mut("dictionary") {
            let path = root.join(dictionary.as_str().unwrap());
            if !path.is_file() {
                return;
            }
            *dictionary = path.to_str().unwrap().into();
        }
    }
    let manifest = common::temp_dir("ocr").join("ocr.json");
    std::fs::write(&manifest, serde_json::to_vec(&config).unwrap()).unwrap();
    let package = Package::load(&manifest).unwrap();
    let error = OcrModel::new(&package, "missing", "ppocrv6_tiny_rec").err().unwrap();
    assert_eq!(error.status(), Status::NotFound);
    let error = OcrModel::new(&package, "ppocrv6_tiny_rec", "ppocrv6_tiny_rec")
        .err()
        .unwrap();
    assert_eq!(error.status(), Status::InvalidConfig);
    let mut decoder = zune_jpeg::JpegDecoder::new(&jpeg);
    let rgb = decoder.decode().unwrap();
    let (width, height) = decoder.dimensions().unwrap();
    let image = Image::new(&rgb, ImageFormat::InterRgb, width as u32, height as u32).unwrap();
    for recognition in ["ppocrv6_tiny_rec", "ppocrv6_tiny_rec_batch8"] {
        let mut model = OcrModel::new(&package, "ppocrv6_tiny_det", recognition).unwrap();
        let result = model.run(&image).unwrap();
        assert!(result.iter().any(|item| item.text.contains("BOARDING")));
        assert!(result.iter().any(|item| item.text.contains("登机牌")));
        let white = vec![255; 64 * 64 * 3];
        let blank = Image::new(&white, ImageFormat::InterRgb, 64, 64).unwrap();
        assert!(model.run(&blank).unwrap().is_empty());
        drop(model);
        assert!(result.iter().all(|item| item.rect.iter().all(|v| v.is_finite())));
    }
}
