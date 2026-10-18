# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: bindings/python/tests/test_ocr.py
# Description: Tests OCR ownership, errors and real CPU/CoreML model cascades.
# -----------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import hybridrt as hrt

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def ocr_package(tmp_path, monkeypatch):
    if not hrt.engine_available("ort"):
        pytest.skip("ONNX Runtime unavailable")
    config = json.loads((ROOT / "assets/template/hrtc_template.json").read_text())
    config["algos"] = [a for a in config["algos"] if a["name"].startswith("ppocrv6_")]
    for algo in config["algos"]:
        model = ROOT / algo["config"]["engine"]["backends"][0]["data"]
        dictionary = algo["config"]["postprocess"].get("dictionary")
        if not model.is_file() or (dictionary and not (ROOT / dictionary).is_file()):
            pytest.skip("OCR fixtures unavailable")
    monkeypatch.chdir(ROOT)

    def load(provider):
        for algo in config["algos"]:
            algo["config"]["engine"]["backends"][0]["config"] = {
                "execution_provider": provider,
                "provider_options": {"ModelFormat": "MLProgram", "RequireStaticInputShapes": "1"}
                if provider == "coreml" else {},
            }
        path = tmp_path / "ocr.json"
        path.write_text(json.dumps(config))
        return hrt.Package(path)

    return load


def test_reports_ocr_errors(ocr_package):
    with ocr_package("cpu") as package:
        with pytest.raises(hrt.HybridRTError) as error:
            hrt.OcrModel(package, detection="missing")
        assert error.value.status == hrt.Status.NOT_FOUND
        with pytest.raises(hrt.HybridRTError) as error:
            hrt.OcrModel(package, detection="ppocrv6_tiny_rec")
        assert error.value.status == hrt.Status.INVALID_CONFIG


@pytest.mark.parametrize("recognition", ["ppocrv6_tiny_rec", "ppocrv6_small_rec"])
@pytest.mark.parametrize("provider", ["cpu", pytest.param("coreml", marks=pytest.mark.skipif(
    sys.platform != "darwin", reason="CoreML requires macOS"))])
def test_runs_ocr_models(ocr_package, recognition, provider):
    cv2 = pytest.importorskip("cv2")
    path = ROOT / "assets/test/images/ocr_document.png"
    if not path.is_file():
        pytest.skip("OCR test image unavailable")
    image = cv2.imread(str(path))
    package = ocr_package(provider)
    with hrt.OcrModel(package, recognition=recognition) as model:
        package.close()
        result = model(image)
        assert len(result) > 0
        assert any("BOARDING" in text.text for text in result)
        assert any("登机牌" in text.text for text in result)
        for item in result:
            x, y, width, height = item.rect
            assert item.rect.shape == (4,)
            assert np.isfinite(item.rect).all()
            assert x >= 0 and y >= 0 and width > 0 and height > 0
            assert x + width < image.shape[1]
            assert y + height < image.shape[0]
            assert 0 <= item.detection_score <= 1
            assert 0 <= item.recognition_score <= 1
        saved = result[0].rect.copy()
        assert model(np.full((64, 64, 3), 255, np.uint8)) == []
        np.testing.assert_array_equal(result[0].rect, saved)
    with pytest.raises(ValueError):
        model(image)
