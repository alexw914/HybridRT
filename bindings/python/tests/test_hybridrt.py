# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/tests/test_hybridrt.py
# Description: Tests the Python package: errors, image layouts, results,
#              threads, logging and template models.
# -----------------------------------------------------------------------------

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import hybridrt as hrt

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "assets" / "template" / "hrtc_template.json"
BUS = ROOT / "assets" / "test" / "images" / "bus.jpg"

needs_ort = pytest.mark.skipif(not hrt.engine_available("ort"), reason="ONNX Runtime is not in this build")


def _varint(value):
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number, value):
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    payload = value.encode() if isinstance(value, str) else value
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _value_info(name, dims):
    shape = b"".join(_field(1, _field(1, dim)) for dim in dims)
    return _field(1, name) + _field(2, _field(1, _field(1, 1) + _field(2, shape)))


def identity_onnx(dims=(1, 3, 2, 2)):
    """y = Identity(x) over float tensors of dims, opset 13, as tests/engine/onnx_model.hpp writes it."""
    node = _field(1, "x") + _field(2, "y") + _field(4, "Identity")
    graph = _field(1, node) + _field(2, "g") + _field(11, _value_info("x", dims)) + _field(12, _value_info("y", dims))
    return _field(1, 7) + _field(7, graph) + _field(8, _field(2, 13))


CLASSIFY = {
    "preprocess": {"dst_image_format": "planar_rgb", "std": [255, 255, 255]},
    "postprocess": {"type": "GeneralClassify", "classify_postprocess_type": "none"},
}


@pytest.fixture
def manifest(tmp_path):
    """Writes a manifest of {algo name: (config, engine_type)} over the identity model."""
    model = tmp_path / "identity.onnx"
    model.write_bytes(identity_onnx())

    def write(algos):
        entries = []
        for name, (config, engine_type) in algos.items():
            config = dict(config, engine={"backends": [{"engine_type": engine_type, "data": str(model)}]})
            entries.append({"name": name, "config": config})
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"name": "python", "version": "1.0.0", "algos": entries}))
        return path

    return write


def uniform_bgr(width=8, height=8):
    image = np.empty((height, width, 3), np.uint8)
    image[:] = (10, 20, 30)
    return image


def test_reports_version_and_engines():
    assert hrt.version()
    assert not hrt.engine_available("unknown")


def test_reports_load_and_create_errors(manifest, tmp_path):
    with pytest.raises(hrt.HybridRTError) as error:
        hrt.Package(tmp_path / "missing.hrtc")
    assert error.value.status == hrt.Status.NOT_FOUND
    with pytest.raises(hrt.HybridRTError):
        hrt.Package(b"\x01\x02\x03")

    bad = dict(CLASSIFY, preprocess=dict(CLASSIFY["preprocess"], unknown_key=1))
    path = manifest({"classify": (CLASSIFY, "ort"), "bad": (bad, "ort"), "absent": (CLASSIFY, "unknown")})
    with hrt.Package(path) as package:
        assert package.algos == ["classify", "bad", "absent"]
        cases = [
            ("missing", {}, hrt.Status.NOT_FOUND, "missing"),
            ("bad", {}, hrt.Status.INVALID_CONFIG, "unknown_key"),
            ("classify", {"engine": 0}, hrt.Status.INVALID_ARGUMENT, "stage counts"),
            ("absent", {}, hrt.Status.UNAVAILABLE, "no engine"),
        ]
        for algo, options, status, message in cases:
            with pytest.raises(hrt.HybridRTError) as error:
                package.model(algo, **options)
            assert error.value.status == status
            assert message in str(error.value)


@needs_ort
def test_runs_classify_on_image_layouts(manifest):
    model = hrt.Model(manifest({"classify": (CLASSIFY, "ort")}))
    assert model.task == hrt.Task.CLASSIFY
    assert model.algo == "classify"
    expected = np.repeat(np.array([30, 20, 10], np.float32) / 255, 4)

    bgr = uniform_bgr()
    result = model(bgr)
    assert isinstance(result, hrt.ClassifyResult)
    assert len(result) == 1
    np.testing.assert_allclose(result.scores[0], expected, rtol=1e-6)

    # RGB channels, a strided view, a planar copy and a region.
    np.testing.assert_allclose(model(bgr[:, :, ::-1], format="inter_rgb").scores[0], expected, rtol=1e-6)
    wide = np.zeros((8, 16, 3), np.uint8)
    wide[:, ::2] = (10, 20, 30)
    np.testing.assert_allclose(model(wide[:, ::2]).scores[0], expected, rtol=1e-6)
    np.testing.assert_allclose(model(bgr.transpose(2, 0, 1), format="planar_bgr").scores[0], expected, rtol=1e-6)
    padded = np.zeros((12, 12, 3), np.uint8)
    padded[2:10, 2:10] = bgr
    np.testing.assert_allclose(model(padded, roi=(2, 2, 8, 8)).scores[0], expected, rtol=1e-6)

    with pytest.raises(TypeError):
        model(bgr.astype(np.float32))
    with pytest.raises(ValueError):
        model(bgr, format="bgr")
    with pytest.raises(ValueError):
        model(bgr[:, :, :2])
    model.close()
    with pytest.raises(ValueError):
        model(bgr)


@needs_ort
def test_runs_calls_from_threads(manifest):
    model = hrt.Model(manifest({"classify": (CLASSIFY, "ort")}), preprocess=2, engine=2, postprocess=2)
    images = [np.full((8, 8, 3), value, np.uint8) for value in range(16)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(model, images))
    for value, result in enumerate(results):
        np.testing.assert_allclose(result.scores[0], np.full(12, value / 255, np.float32), rtol=1e-6)


@needs_ort
def test_reports_process_errors_through_logging(manifest, caplog):
    model = hrt.Model(manifest({"classify": (CLASSIFY, "ort")}))
    hrt.use_python_logging()
    try:
        with caplog.at_level(logging.ERROR, logger="hybridrt"):
            with pytest.raises(hrt.HybridRTError) as error:
                model(uniform_bgr(), roi=(4, 4, 8, 8))
    finally:
        hrt.set_log_callback(None)
    assert error.value.status == hrt.Status.PROCESS_FAILED
    assert "leaves" in str(error.value)
    assert any("leaves" in record.getMessage() for record in caplog.records)
    assert len(model(uniform_bgr()).scores) == 1


@pytest.fixture
def template(monkeypatch):
    """Models of the template, whose fixture paths are relative to the repository root, and bus.jpg as BGR."""
    cv2 = pytest.importorskip("cv2")
    if not hrt.engine_available("ort") or not BUS.is_file():
        pytest.skip("ONNX Runtime or bus.jpg is unavailable")
    monkeypatch.chdir(ROOT)
    config = json.loads(TEMPLATE.read_text())
    models = {algo["name"]: algo["config"]["engine"]["backends"][0]["data"] for algo in config["algos"]}
    # Loading a manifest reads every model file it names, so the test loads it once.
    package = hrt.Package(TEMPLATE)

    def load(name):
        if not (ROOT / models[name]).is_file():
            pytest.skip(f"model of {name} is unavailable")
        return package.model(name)

    yield load, cv2.imread(str(BUS))
    package.close()


def test_runs_template_models(template):
    load, bus = template
    height, width = bus.shape[:2]

    detect = load("yolov8n_detect")(bus)
    assert isinstance(detect, hrt.DetectResult)
    assert detect.boxes.shape == (len(detect), 4)
    assert 0 in detect.class_ids
    assert np.all(detect.boxes[:, 2:] > 0)

    # Three people and the bus, as the C++ segment test reports.
    segment = load("yolo26n_e2e_segment")(bus)
    assert isinstance(segment, hrt.SegmentResult)
    assert list(segment.class_ids) == [0, 0, 0, 5]
    for mask in segment.masks:
        x, y, w, h = mask.rect
        assert mask.data.shape == (h, w)
        assert 0 <= x and 0 <= y and x + w <= width and y + h <= height
        assert mask.data.max() == 1

    pose = load("yolov8n_pose")(bus)
    assert isinstance(pose, hrt.KeypointResult)
    assert pose.keypoints.shape == (len(pose), 17, 3)
    assert pose.visible.shape == (len(pose), 17)

    rotated = load("yolov8n_obb")(bus)
    assert isinstance(rotated, hrt.RotatedResult)
    assert rotated.boxes.shape == (len(rotated), 5)

    classify = load("yolov8n_classify")(bus)
    assert len(classify.scores) == 1
    assert classify.scores[0].shape == (1000,)
