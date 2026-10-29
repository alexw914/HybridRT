// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/tests/common/mod.rs
// Description: Writes an identity ONNX model and manifests for the tests.
// -----------------------------------------------------------------------------

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

/// The repository root, which the template's fixture paths are relative to.
pub fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

pub fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("hybridrt_rust").join(name);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn number(field: u64, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    varint(&mut out, field << 3);
    varint(&mut out, value);
    out
}

fn bytes(field: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    varint(&mut out, (field << 3) | 2);
    varint(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    out
}

fn value_info(name: &str, dims: &[u64]) -> Vec<u8> {
    let shape: Vec<u8> = dims.iter().flat_map(|&dim| bytes(1, &number(1, dim))).collect();
    let tensor = [number(1, 1), bytes(2, &shape)].concat();
    [bytes(1, name.as_bytes()), bytes(2, &bytes(1, &tensor))].concat()
}

/// y = Identity(x) over float {1, 3, 2, 2}, opset 13, as tests/engine/onnx_model.hpp writes it.
pub fn identity_onnx() -> Vec<u8> {
    identity_onnx_batch(1)
}

pub fn identity_onnx_batch(batch: u64) -> Vec<u8> {
    let dims = [batch, 3, 2, 2];
    let node = [bytes(1, b"x"), bytes(2, b"y"), bytes(4, b"Identity")].concat();
    let graph = [
        bytes(1, &node),
        bytes(2, b"g"),
        bytes(11, &value_info("x", &dims)),
        bytes(12, &value_info("y", &dims)),
    ]
    .concat();
    [number(1, 7), bytes(7, &graph), bytes(8, &number(2, 13))].concat()
}

pub const CLASSIFY: &str = r#""preprocess": {"dst_image_format": "planar_rgb", "std": [255, 255, 255]},
  "postprocess": {"type": "GeneralClassify", "classify_postprocess_type": "none"}"#;

/// A manifest of (algo name, config sections, engine type) over the identity model.
pub fn manifest(name: &str, algos: &[(&str, &str, &str)]) -> PathBuf {
    let dir = temp_dir(name);
    let model = dir.join("identity.onnx");
    fs::write(&model, identity_onnx()).unwrap();
    let model = model.to_str().unwrap().replace('\\', "/");
    let entries: Vec<String> = algos
        .iter()
        .map(|(algo, config, engine)| {
            format!(
                r#"{{"name": "{algo}", "config": {{{config},
  "engine": {{"backends": [{{"engine_type": "{engine}", "data": "{model}"}}]}}}}}}"#
            )
        })
        .collect();
    let path = dir.join("manifest.json");
    fs::write(
        &path,
        format!(
            r#"{{"name": "rust", "version": "1.0.0", "algos": [{}]}}"#,
            entries.join(",")
        ),
    )
    .unwrap();
    path
}

/// An 8x8 BGR image of (10, 20, 30).
pub fn uniform_bgr() -> Vec<u8> {
    [10u8, 20, 30].repeat(64)
}

/// The identity classify scores of uniform_bgr: planar RGB / 255.
pub fn expected_scores() -> Vec<f32> {
    [30.0f32, 20.0, 10.0]
        .iter()
        .flat_map(|value| [value / 255.0; 4])
        .collect()
}

pub fn assert_scores(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
    }
}
