// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/tests/model.rs
// Description: Tests packages, model errors, image layouts, results and
//              threads on an identity model.
// -----------------------------------------------------------------------------

mod common;

use common::{assert_scores, expected_scores, manifest, uniform_bgr, CLASSIFY};
use hybridrt::{
    engine_available, version, Image, ImageFormat, Model, ModelOptions, Package, Rect, Results, Status, Task,
};

#[test]
fn reports_version_and_engines() {
    assert!(!version().is_empty());
    assert!(!engine_available("unknown"));
    assert!(!engine_available("o\0rt"));
}

#[test]
fn reports_load_and_create_errors() {
    let missing = common::temp_dir("load_errors").join("missing.hrtc");
    assert_eq!(Package::load(&missing).err().unwrap().status(), Status::NotFound);
    assert_eq!(
        Package::from_bytes(&[1, 2, 3]).err().unwrap().status(),
        Status::NotFound
    );

    let bad = CLASSIFY.replace(r#""std""#, r#""unknown_key": 1, "std""#);
    let path = manifest(
        "create_errors",
        &[
            ("classify", CLASSIFY, "ort"),
            ("bad", &bad, "ort"),
            ("absent", CLASSIFY, "unknown"),
        ],
    );
    let package = Package::load(path).unwrap();
    assert_eq!(package.algos(), ["classify", "bad", "absent"]);
    let zero = ModelOptions {
        engine: 0,
        ..ModelOptions::default()
    };
    for (algo, options, status, message) in [
        ("missing", ModelOptions::default(), Status::NotFound, "missing"),
        ("bad", ModelOptions::default(), Status::InvalidConfig, "unknown_key"),
        ("classify", zero, Status::InvalidArgument, "stage counts"),
        ("absent", ModelOptions::default(), Status::Unavailable, "no engine"),
    ] {
        let error = Model::new(&package, Some(algo), options).err().unwrap();
        assert_eq!(error.status(), status, "{error}");
        assert!(error.message().contains(message), "{error}");
    }
    assert_eq!(
        Model::new(&package, Some("a\0b"), ModelOptions::default())
            .err()
            .unwrap()
            .status(),
        Status::InvalidArgument
    );
}

#[test]
fn checks_image_bounds() {
    let pixels = uniform_bgr();
    assert!(Image::new(&pixels, ImageFormat::InterBgr, 8, 8).is_ok());
    let short = Image::new(&pixels[..191], ImageFormat::InterBgr, 8, 8).err().unwrap();
    assert!(short.message().contains("needs 192 bytes"), "{short}");
    assert!(Image::new(&pixels, ImageFormat::InterBgr, 0, 8).is_err());
    assert!(Image::with_row_pitch(&pixels, ImageFormat::InterBgr, 8, 8, 4).is_err());
    assert!(Image::new(&pixels, ImageFormat::Nv12, 7, 8).is_err());
    // The last row needs no padding after it.
    assert!(Image::with_row_pitch(&pixels, ImageFormat::Gray, 8, 2, 184).is_ok());
    assert!(Image::with_row_pitch(&pixels, ImageFormat::Gray, 8, 2, 185).is_err());
}

#[test]
fn runs_identity_model_on_image_layouts() {
    if !engine_available("ort") {
        return;
    }
    let path = manifest("layouts", &[("classify", CLASSIFY, "ort")]);
    let package = Package::load(path).unwrap();
    let model = Model::new(&package, None, ModelOptions::default()).unwrap();
    drop(package);
    assert_eq!(model.task(), Task::Classify);
    assert_eq!(model.algo(), "classify");
    let expected = expected_scores();

    let bgr = uniform_bgr();
    let mut results = Results::new().unwrap();
    for _ in 0..2 {
        model
            .process(
                &Image::new(&bgr, ImageFormat::InterBgr, 8, 8).unwrap(),
                None,
                &mut results,
            )
            .unwrap();
        assert_eq!(results.task(), Task::Classify);
        assert_eq!(results.len(), 1);
        assert!(results.boxes().is_empty());
        assert_scores(results.scores(0).unwrap(), &expected);
        assert!(results.scores(1).is_none());
        assert!(results.keypoints(0).is_none());
        assert!(results.mask(0).is_none());
    }

    let rgb: Vec<u8> = bgr.chunks(3).flat_map(|pixel| [pixel[2], pixel[1], pixel[0]]).collect();
    let run = |pixels: &[u8], format, pitch, roi| {
        let image = Image::with_row_pitch(pixels, format, 8, 8, pitch).unwrap();
        model.run(&image, roi).unwrap().scores(0).unwrap().to_vec()
    };
    assert_scores(&run(&rgb, ImageFormat::InterRgb, 0, None), &expected);
    let planar: Vec<u8> = [10u8, 20, 30].iter().flat_map(|&value| [value; 64]).collect();
    assert_scores(&run(&planar, ImageFormat::PlanarBgr, 0, None), &expected);
    // Rows padded to 32 bytes, and a region inside them.
    let padded: Vec<u8> = bgr.chunks(24).flat_map(|row| [row, &[0u8; 8]].concat()).collect();
    assert_scores(
        &run(
            &padded,
            ImageFormat::InterBgr,
            32,
            Some(Rect {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
            }),
        ),
        &expected,
    );

    let outside = Rect {
        x: 4,
        y: 4,
        width: 8,
        height: 8,
    };
    let error = model
        .run(&Image::new(&bgr, ImageFormat::InterBgr, 8, 8).unwrap(), Some(outside))
        .err()
        .unwrap();
    assert_eq!(error.status(), Status::ProcessFailed);
    assert!(error.message().contains("leaves"), "{error}");
}

#[test]
fn returns_embedding() {
    if !engine_available("ort") {
        return;
    }
    let config = CLASSIFY.replace(
        r#"{"type": "GeneralClassify", "classify_postprocess_type": "none"}"#,
        r#"{"type": "Embedding"}"#,
    );
    let path = manifest("embedding", &[("embedding", &config, "ort")]);
    let package = Package::load(path).unwrap();
    let model = Model::new(&package, None, ModelOptions::default()).unwrap();
    assert_eq!(model.task(), Task::Embedding);
    let bgr = uniform_bgr();
    let mut results = Results::new().unwrap();
    model
        .process(
            &Image::new(&bgr, ImageFormat::InterBgr, 8, 8).unwrap(),
            None,
            &mut results,
        )
        .unwrap();
    assert_eq!(results.task(), Task::Embedding);
    assert_eq!(results.len(), 1);
    assert_scores(results.scores(0).unwrap(), &expected_scores());
}

#[test]
fn runs_calls_from_threads() {
    if !engine_available("ort") {
        return;
    }
    let path = manifest("threads", &[("classify", CLASSIFY, "ort")]);
    let options = ModelOptions {
        preprocess: 2,
        engine: 2,
        postprocess: 2,
    };
    let model = Model::new(&Package::load(path).unwrap(), None, options).unwrap();
    std::thread::scope(|scope| {
        for value in 0..4u8 {
            let model = &model;
            scope.spawn(move || {
                let pixels = vec![value; 8 * 8 * 3];
                let image = Image::new(&pixels, ImageFormat::InterBgr, 8, 8).unwrap();
                let mut results = Results::new().unwrap();
                for _ in 0..4 {
                    model.process(&image, None, &mut results).unwrap();
                    assert_scores(results.scores(0).unwrap(), &[f32::from(value) / 255.0; 12]);
                }
            });
        }
    });
}

#[test]
fn runs_batches_and_reuses_results() {
    if !engine_available("ort") {
        return;
    }
    for batch in [1, 4, 8] {
        let path = manifest(&format!("batch{batch}"), &[("classify", CLASSIFY, "ort")]);
        std::fs::write(
            path.parent().unwrap().join("identity.onnx"),
            common::identity_onnx_batch(batch),
        )
        .unwrap();
        let model = Model::new(&Package::load(path).unwrap(), None, ModelOptions::default()).unwrap();
        assert_eq!(model.batch(), batch as usize);
        let pixels: Vec<_> = (0..9u8).map(|i| vec![i * 20; 8 * 8 * 3]).collect();
        let images: Vec<_> = pixels
            .iter()
            .map(|p| Image::new(p, ImageFormat::InterBgr, 8, 8).unwrap())
            .collect();
        for count in [1, batch as usize, batch as usize + 1] {
            let mut results = model.run_batch(&images[..count], None).unwrap();
            for (i, result) in results.iter().enumerate() {
                assert_scores(result.scores(0).unwrap(), &[i as f32 * 20.0 / 255.0; 12]);
            }
            let rois = vec![
                Rect {
                    x: 1,
                    y: 1,
                    width: 4,
                    height: 4
                };
                count
            ];
            model
                .process_batch(&images[..count], Some(&rois), &mut results)
                .unwrap();
            for (i, result) in results.iter().enumerate() {
                assert_scores(result.scores(0).unwrap(), &[i as f32 * 20.0 / 255.0; 12]);
            }
            assert!(model.process_batch(&images[..count], Some(&[]), &mut results).is_err());
        }
        assert!(model.run_batch(&[], None).is_err());
    }
}
